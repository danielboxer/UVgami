# import copy
import functools
import multiprocessing
import time
import traceback
from collections import deque, namedtuple

import bpy
import numpy

from .batch import BatchProcess, last_meaningful_line
from .job import Join, Result, TransferReport
from .logger import logger
from .ops.grid import add_grid, make_grid_img, make_grid_mat
from .ops.uv import pack_objects, show_seams
from .part_overrides import write_overrides
from .progress_bar import progress_bar
from .seams.uv_transfer import AMBIGUOUS_GEOMETRY
from .similar import write_twin_output
from .utils.geometry import set_origin
from .utils.io import import_obj
from .utils.mesh import (
    add_shading_modifiers,
    check_collection,
    check_exists,
    edit_restore,
    mark_not_unwrapped,
    move_to_collection,
)
from .utils.paths import clear_io_dir, get_io_dir_paths, get_preferences
from .utils.ui import popup, set_status, switch_solid_color, tag_redraw

STATUS_SECONDS = 5
DISPATCH_SECONDS = 0.1
# a finished piece waits for the tick
QUEUED_DISPATCH_SECONDS = 0.02
# reserved from the start, or the bar moves backward when the transfer begins
TRANSFER_PROGRESS_SHARE = 0.4
# a sidebar rebuild mid click drops the click
PANEL_REDRAW_SECONDS = 1.0


# a slider moved mid-run changes nothing
class Settings:
    def __init__(self, props):
        for prop in props.bl_rna.properties:
            if prop.identifier == "rna_type":
                continue
            value = getattr(props, prop.identifier)
            if prop.type == "POINTER":
                value = Settings(value)
            elif isinstance(value, set):
                value = set(value)
            setattr(self, prop.identifier, value)
        # bl_rna skips the plain python properties like avoid_seams
        for cls in type(props).__mro__:
            for name, attribute in vars(cls).items():
                if isinstance(attribute, property) and not hasattr(self, name):
                    setattr(self, name, getattr(props, name))


PendingTransfer = namedtuple(
    "PendingTransfer", ["job", "output", "pack_index", "name", "started_at"]
)
# an object being preseeded, before its pieces exist
Preparing = namedtuple("Preparing", ["name", "cancel"])
STOP_SECONDS = 180


class UnwrapManager:
    def __init__(self):
        self._queue = deque()
        self._running = []
        self._shared_processes = []
        # argv batches, one engine process for a whole queue
        self._batch_processes = []
        self._pack_output_objects = []
        self._pack_topology = {}
        self.engine = None
        self.engine_ctx = None
        self.is_active = False
        self.in_uv_editor = False
        self._dispatch_handle = None
        self.pieces_still_arriving = 0
        # _reset_session must not clear this, the builders add and remove entries
        self.preparing = []
        self.summary = []
        self.summary_failed = False
        self._reset_session()

    # called from __init__ too, end_session can run before a session starts
    def _reset_session(self):
        self.props = None
        self.moved_to_invalid = False
        # (unwrap, result) per settled piece
        self.results = []
        self._to_import = []
        self.pending_transfers = []
        self.cancelled_transfers = set()
        self.transfer_uv_failed = False
        self.transfer_uv_fail_detail = ""
        self.transfer_uv_reason_known = False
        self.transfer_uv_split_count = 0
        self.error_code = 0
        self.error_stderr = ""
        self.error_messages = []
        self.current_viewer = None
        self.is_viewer_active = False
        self.exit_viewer = False
        self.viewer_done = False
        self._pack_output_objects = []
        self._pack_topology = {}
        self._drawn_panel_state = None
        self._panel_drawn_at = 0.0
        self.session_started_at = time.monotonic()
        # (done, running, remaining) fractions, where the bar starts
        self.progress = numpy.array((0.0, 0.0, 1.0))

    @property
    def active(self):
        return self._running + list(self._queue)

    def add(self, unwrap):
        self._queue.append(unwrap)

    def remove_unwrap(self, unwrap):
        if unwrap in self._running:
            self._running.remove(unwrap)
        elif unwrap in self._queue:
            self._queue.remove(unwrap)

    # the queue ui lists every piece before any input file exists
    def claimed_paths(self):
        claimed = {unwrap.path for unwrap in self.active}
        claimed.update(unwrap.path for unwrap, _ in self.results)
        return claimed

    # a session that adds to a running one keeps the running one's files
    def prepare_input_dir(self):
        input_path, output_path = get_io_dir_paths()
        if not self.is_active:
            clear_io_dir(input_path)
            clear_io_dir(output_path)
        return input_path

    def start(self, engine, engine_ctx, uv_editor=False):
        self._reset_session()
        self.engine = engine
        self.engine_ctx = engine_ctx
        self.props = Settings(bpy.context.scene.uvgami)
        self._fill_slots()
        if get_preferences().show_progress_bar:
            progress_bar.start(uv_editor)
        self.in_uv_editor = uv_editor
        self.is_active = True
        self.clear_summary()
        self._dispatch_handle = functools.partial(self._dispatch)
        bpy.app.timers.register(self._dispatch_handle)

    def _fill_slots(self):
        props = self.props
        engine = self.engine
        if engine.batches_queue(props):
            if self.pieces_still_arriving > 0:
                # a batch needs the whole queue at once
                return
            if any(u.batch_process is not None for u in self._running):
                return
            if len(self._queue) > 1:
                self._start_batch_process(engine, props)
                return
            # a single queued mesh runs the normal solo path
        max_concurrent = 1 if engine.batches_queue(props) else props.max_cores
        # pieces export in queue order
        for unwrap in list(self._queue):
            if len(self._running) >= max_concurrent:
                break
            if not unwrap.is_exported:
                break
            # a copy twin never runs, its representative settles it
            if unwrap.copy_of is not None:
                continue
            self._queue.remove(unwrap)
            shared_args = engine.build_shared_args(
                self.engine_ctx,
                unwrap.path,
                props,
                self._shared_threads(unwrap, props),
            )
            if shared_args is None:
                unwrap.start_unwrap()
            else:
                self._send_to_shared_process(unwrap, shared_args)
            self._running.append(unwrap)

    # def piece_props(self, unwrap):
    #     if unwrap.priority_override is None:
    #         return self.props
    #     props = copy.copy(self.props)
    #     props.priority = unwrap.priority_override
    #     return props

    # a lone mesh gets every core
    def _shared_threads(self, unwrap, props):
        pieces = unwrap.join_job.expected if unwrap.join_job is not None else 1
        if pieces < props.max_cores:
            return 0
        return max(1, multiprocessing.cpu_count() // props.max_cores)

    # spawning a process costs 50ms inside Blender
    def _send_to_shared_process(self, unwrap, args):
        # a part priority override runs with other args
        process = next(
            (p for p in self._shared_processes if p.is_idle and p.args == args), None
        )
        if process is None:
            process = BatchProcess(args, self.engine.build_env(self.engine_ctx))
            self._shared_processes.append(process)
        process.send(unwrap.path, unwrap)
        unwrap.join_batch(process)

    # a closed stdin ends a shared process, a batch process runs its argv to the end
    def _close_processes(self):
        for process in self._shared_processes:
            process.close()
        self._shared_processes.clear()
        for process in self._batch_processes:
            if process.process.poll() is None:
                self.engine.stop(process.process, self.engine_ctx)
            process.close()
        self._batch_processes.clear()

    def _start_batch_process(self, engine, props):
        unwraps = [u for u in self._queue if u.copy_of is None]
        if not unwraps:
            return
        self._queue = deque(u for u in self._queue if u not in unwraps)
        args = engine.build_batch_args(
            self.engine_ctx, [u.path for u in unwraps], props
        )
        batch_process = BatchProcess(
            args,
            engine.build_env(self.engine_ctx),
            {unwrap.path.stem: unwrap for unwrap in unwraps},
        )
        self._batch_processes.append(batch_process)
        for unwrap in unwraps:
            unwrap.join_batch(batch_process)
            self._running.append(unwrap)

    def _dispatch(self):
        if not self.is_active:
            return None

        try:
            completed = []
            failed = []
            requeued = []

            for unwrap in list(self._running):
                unwrap.read_output()
                unwrap.update_progress()

                if unwrap.viewing:
                    unwrap.update_viewer()

                if unwrap.is_stopped:
                    self.engine.request_early_stop(unwrap.process)
                    if unwrap.stop_requested_at is None:
                        unwrap.stop_requested_at = time.monotonic()
                    elif time.monotonic() - unwrap.stop_requested_at > STOP_SECONDS:
                        unwrap.stop_process()
                        failed.append(
                            (unwrap, self._fail, "Stop timed out (force killed)")
                        )
                        # the poll below must not fail it a second time
                        continue

                timeout_minutes = self.props.unwrap_timeout
                if (
                    timeout_minutes > 0
                    and not unwrap.is_stopped
                    and unwrap.started_at is not None
                    and time.monotonic() - unwrap.started_at > timeout_minutes * 60
                ):
                    if self.engine.supports_early_stop:
                        # force killed after STOP_SECONDS by the stop branch above
                        unwrap.is_stopped = True
                        self.error_messages.append(
                            f"{unwrap.input_name}: timed out,"
                            " stopped with a partial result"
                        )
                    else:
                        unwrap.stop_process()
                        elapsed = (time.monotonic() - unwrap.started_at) / 60
                        failed.append(
                            (
                                unwrap,
                                self._fail,
                                f"Timed out after {elapsed:.1f} minutes",
                            )
                        )
                        # the poll below must not fail it a second time
                        continue

                ret_code = unwrap.poll_engine()
                if ret_code is not None:
                    if ret_code == 0 and unwrap.output_path.is_file():
                        completed.append(unwrap)
                    elif ret_code == 0:
                        failed.append((unwrap, self._fail, "Engine produced no output"))
                    else:
                        # a batched mesh that never started goes back to the queue
                        stem = unwrap.path.stem
                        if (
                            unwrap.batch_process is not None
                            and unwrap.batch_process.should_retry(stem)
                            and unwrap.path.is_file()
                        ):
                            requeued.append(unwrap)
                        else:
                            failed.append((unwrap, self._handle_failure, ret_code))

            if self._running or self._queue:
                logger.mark_unwrapping()
            logger.update_time()
            self._update_progress_bar()

            for unwrap in completed:
                self._settle_step(unwrap, "finishing", self._process_completion)

            for unwrap, step, detail in failed:
                self._settle_step(unwrap, "handling the failure of", step, detail)

            self._drain_imports()
            self._finish_transfers()

            for unwrap in requeued:
                if unwrap in self._running:
                    self._running.remove(unwrap)
                unwrap.leave_batch()
                self._queue.append(unwrap)
            if requeued:
                print(f"UVgami: requeued {len(requeued)} mesh(es) after batch ended")

            self._fill_slots()

            if (
                not self._running
                and not self._queue
                and self.pieces_still_arriving == 0
                and not self.pending_transfers
            ):
                self.finish_session()
                return None

        except Exception as e:
            from .handler import handle_error

            handle_error(e, "MIDDLE")
            return None

        return QUEUED_DISPATCH_SECONDS if self._queue else DISPATCH_SECONDS

    # a piece with a uv transfer splits its bar between the engine and the transfer
    def _update_progress_bar(self):
        active = self.active
        pieces = active + [unwrap for unwrap, _ in self.results]
        if not pieces:
            return
        engine = numpy.array(
            [unwrap.progress for unwrap in active]
            + [(1.0, 0.0, 0.0)] * len(self.results)
        )
        jobs = [unwrap.transfer_uvs_job for unwrap in pieces]
        has_transfer = numpy.array(
            [job is not None and not job.settled for job in jobs]
        )
        # a job waiting on the rest of its group has not started reporting
        reported = {entry.job: entry.job.progress for entry in self.pending_transfers}
        transferred = numpy.array([reported.get(job, 0.0) for job in jobs])
        share = TRANSFER_PROGRESS_SHARE * has_transfer
        progress = engine * (1 - share)[:, None]
        progress[:, 0] += share * transferred
        progress[:, 2] += share * (1 - transferred)
        progress = progress.mean(axis=0)
        # the bar's gpu batches are rebuilt on every update
        if get_preferences().show_progress_bar and not numpy.array_equal(
            progress, self.progress
        ):
            progress_bar.update(progress)
            tag_redraw(("WINDOW",))
        self.progress = progress
        now = time.monotonic()
        if now - self._panel_drawn_at < PANEL_REDRAW_SECONDS:
            return
        state = self._panel_state()
        if state != self._drawn_panel_state:
            self._drawn_panel_state = state
            self._panel_drawn_at = now
            tag_redraw(("UI",))

    @property
    def elapsed_seconds(self):
        return int(time.monotonic() - self.session_started_at)

    # everything the queue ui draws, as a comparable value
    def _panel_state(self):
        return (
            self.elapsed_seconds,
            len(self.results),
            self.is_viewer_active,
            tuple(p.name for p in self.preparing),
            tuple(t.name for t in self.pending_transfers),
            tuple(
                (
                    unwrap.path.stem,
                    unwrap.is_running,
                    unwrap.is_exported,
                    unwrap.is_viewable,
                    unwrap.is_stalled,
                )
                for unwrap in self.active
            ),
        )

    # every piece ends here as finished, invalid, or cancelled
    def record_result(self, unwrap, result):
        if unwrap.result is not None:
            return
        unwrap.result = result
        self.results.append((unwrap, result))

        self.remove_unwrap(unwrap)
        unwrap.release_engine()
        unwrap.cleanup()

        group = unwrap.join_job
        if group is None:
            if result is Result.FINISHED:
                self._to_import.append(unwrap)
        else:
            group.record(unwrap, result)
            if group.is_settled() and group.finished and not group.discard:
                self._to_import.append(group)

        for twin in unwrap.twins:
            self._settle_twin(twin, result)

    # an unexported twin has no metadata, the exporter settles it
    def _settle_twin(self, twin, result):
        if twin.result is not None:
            return
        if result is Result.FINISHED:
            if not twin.is_exported:
                return
            write_twin_output(
                twin.copy_of.output_path, twin.output_path, twin.copy_matrix
            )
        self.record_result(twin, result)

    def _drain_imports(self):
        for item in self._to_import:
            is_group = isinstance(item, Join)
            unwrap = item.finished[-1] if is_group else item
            try:
                if is_group and len(item.finished) > 1:
                    path, added_edges = item.finish()
                else:
                    path, added_edges = unwrap.output_path, unwrap.added_edges
                self._import_and_finalize(unwrap, path, added_edges)
                # a mid-session ctrl z would otherwise discard the result
                if self._running or self._queue or self.pieces_still_arriving:
                    bpy.ops.ed.undo_push(message="UVgami Unwrap")
            except Exception:
                error_list = traceback.format_exc().split("\n")[:-1]
                logger.add_data("errors", "Error finishing unwrap:")
                for line in error_list:
                    logger.add_data("errors", line)
                    print(line)
                self.error_messages.append(
                    f"Error finishing {unwrap.input_name}, see the console"
                )
        self._to_import.clear()

    # an exception here leaves the piece's group waiting forever
    def _settle_step(self, unwrap, doing, step, *args):
        try:
            step(unwrap, *args)
        except Exception:
            message = f"Error {doing} {unwrap.input_name}"
            logger.add_data("errors", f"{message}:")
            for line in traceback.format_exc().split("\n")[:-1]:
                logger.add_data("errors", line)
                print(line)
            self.error_messages.append(f"{message}, see the console")
            self.record_result(unwrap, Result.INVALID)

    def _process_completion(self, unwrap):
        if unwrap.viewing:
            self.viewer_done = True
            tag_redraw()
        self.record_result(unwrap, Result.FINISHED)

    def _import_and_finalize(self, unwrap, path, added_edges):
        props = self.props

        output = import_obj(path, f"{unwrap.input_name}_unwrapped")

        set_origin(output, unwrap.origin)

        for m_name in unwrap.materials:
            output.data.materials.append(
                None if m_name is None else bpy.data.materials.get(m_name)
            )

        # before the preserve job, so untriangulated faces carry these along
        self._restore_face_data(unwrap, output, "material_index", "material_indices")
        self._restore_face_data(unwrap, output, "use_smooth", "face_smooth")
        self._restore_overrides(unwrap, output)
        add_shading_modifiers(output, unwrap.shading_modifiers)

        if unwrap.preserve_job is not None:
            unwrap.preserve_job.finish(output, added_edges)

        if unwrap.hide_job is not None:
            unwrap.hide_job.finish()

        symmetrize_job = unwrap.symmetrize_job
        if symmetrize_job is not None:
            symmetrize_job.finish(output, unwrap.matrix)

        pieces = unwrap.join_job.finished if unwrap.join_job is not None else [unwrap]
        if any(u.preseeded for u in pieces):
            # imported here: ops.island imports this module back
            from .ops.island import finish_preseed, read_mesh, rectify_islands

            ranges = None
            if len(pieces) > 1:
                ranges = []
                start = 0
                for u in pieces:
                    # material_indices is the piece's own face count
                    stop = start + len(u.material_indices)
                    if u.preseeded:
                        ranges.append((start, stop))
                    start = stop
                if start != len(output.data.polygons):
                    ranges = None
            reads = read_mesh(output.data)
            reads = finish_preseed(output, reads, ranges)
            rectify_islands(output, reads)
            self._pack_topology[output] = (reads.faces, reads.edges)

        if props.pack_after_unwrap:
            self._pack_output_objects.append(output)

        self._restore_vertex_groups(unwrap, output)

        # vertex groups restore by index, and the weld renumbers vertices
        weld_job = unwrap.segment_weld_job
        if weld_job is not None and unwrap.transfer_uvs_job is None:
            weld_job.finish(output)
            # its vertex indices are from before the weld
            self._pack_topology.pop(output, None)

        edit_restore([output], show_seams)

        if unwrap.transfer_uvs_job is not None:
            job = unwrap.transfer_uvs_job
            # output is deleted by the transfer
            pack_index = None
            if props.pack_after_unwrap:
                for i, obj in enumerate(self._pack_output_objects):
                    if obj == output:
                        pack_index = i
                        break
            group = unwrap.join_job
            missing_pieces = group is not None and len(group.finished) < group.expected
            if missing_pieces and not job.allows_missing_pieces:
                failed = group.expected - len(group.finished)
                self.transfer_uv_reason_known = True
                report = TransferReport(
                    False, 0, f"{failed} of {group.expected} parts were not unwrapped"
                )
                self._settle_transfer(job, output, pack_index, report)
                return
            report = job.start(output)
            if report is None:
                # _finish_transfers settles it when the worker is done
                self.pending_transfers.append(
                    PendingTransfer(
                        job, output, pack_index, unwrap.input_name, time.monotonic()
                    )
                )
                # or it sits in the scene beside the original
                for collection in output.users_collection:
                    collection.objects.unlink(output)
                return
            self._settle_transfer(job, output, pack_index, report)
            return

        self._add_auto_grid(props, output)

        collection = check_collection("UVgami Unwrapped", bpy.context.scene.collection)
        move_to_collection(output, collection)

    def _settle_transfer(self, job, output, pack_index, report):
        job.settled = True
        props = self.props
        input_mesh = job.input_mesh
        if report.applied:
            self.transfer_uv_split_count += report.split_count
            if pack_index is not None and job.repack_input:
                self._pack_output_objects[pack_index] = input_mesh
            replacement = job.replacement
            if replacement is None:
                self._add_auto_grid(props, input_mesh)
                return
            if pack_index is not None:
                self._pack_output_objects[pack_index] = replacement
            output = replacement
        else:
            # no hide job exists when a transfer job is in the slot
            if check_exists(input_mesh):
                input_mesh.hide_set(True)
            self.transfer_uv_failed = True
            self.transfer_uv_fail_detail = report.detail
            if report.reason == AMBIGUOUS_GEOMETRY:
                self.transfer_uv_reason_known = True
            logger.add_data(
                "errors",
                f"UV transfer failed ({report.detail}), keeping output",
            )
            if not check_exists(output):
                return

        self._add_auto_grid(props, output)

        collection = check_collection("UVgami Unwrapped", bpy.context.scene.collection)
        move_to_collection(output, collection)

    def _finish_transfers(self):
        timeout_minutes = self.props.unwrap_timeout
        for entry in list(self.pending_transfers):
            job, output, pack_index = entry.job, entry.output, entry.pack_index
            if (
                timeout_minutes > 0
                and time.monotonic() - entry.started_at > timeout_minutes * 60
            ):
                # cancel deletes the two objects the settle path works on
                self.cancel_transfer(entry)
                self.transfer_uv_failed = True
                self.transfer_uv_reason_known = True
                self.transfer_uv_fail_detail = f"{entry.name} timed out"
                continue
            try:
                report = job.poll()
            except Exception:
                error_list = traceback.format_exc().split("\n")[:-1]
                logger.add_data("errors", "Error finishing UV transfer:")
                for line in error_list:
                    logger.add_data("errors", line)
                    print(line)
                report = TransferReport(False, 0, "see the console")
            if report is None:
                continue
            self.pending_transfers.remove(entry)
            try:
                self._settle_transfer(job, output, pack_index, report)
            except Exception:
                error_list = traceback.format_exc().split("\n")[:-1]
                logger.add_data("errors", "Error finishing UV transfer:")
                for line in error_list:
                    logger.add_data("errors", line)
                    print(line)

    # the input mesh is still there after a transfer deletes the output
    def _add_auto_grid(self, props, obj):
        if props.auto_grid:
            add_grid(obj, make_grid_mat(make_grid_img()))
            switch_solid_color("TEXTURE")

    # in piece order when the mesh was separated
    def _restore_face_data(self, unwrap, output, attribute, field):
        pieces = unwrap.join_job.finished if unwrap.join_job is not None else [unwrap]
        values = []
        for piece in pieces:
            values.extend(getattr(piece, field))
        if len(values) == len(output.data.polygons):
            output.data.polygons.foreach_set(attribute, values)

    # pack reads the uv scale override off the output
    def _restore_overrides(self, unwrap, output):
        pieces = unwrap.join_job.finished if unwrap.join_job is not None else [unwrap]
        overrides = {
            name: [value for piece in pieces for value in piece.overrides[name]]
            for name in pieces[0].overrides
        }
        if all(len(v) == len(output.data.polygons) for v in overrides.values()):
            write_overrides(output.data, overrides)

    # one add per distinct weight, a call per vertex is slow on a dense mesh
    def _restore_vertex_groups(self, unwrap, output):
        pieces = unwrap.join_job.finished if unwrap.join_job is not None else [unwrap]
        combined = {}
        offset = 0
        for piece in pieces:
            for name, (indices, weights) in piece.vertex_groups.items():
                combined.setdefault(name, []).append((indices + offset, weights))
            offset += piece.vertex_count

        vertex_count = len(output.data.vertices)
        for name, parts in combined.items():
            indices = numpy.concatenate([indices for indices, _ in parts])
            weights = numpy.concatenate([weights for _, weights in parts])
            inside = indices < vertex_count
            indices, weights = indices[inside], weights[inside]
            order = numpy.argsort(weights, kind="stable")
            values, starts = numpy.unique(weights[order], return_index=True)
            group = output.vertex_groups.new(name=name)
            for weight, run in zip(values, numpy.split(indices[order], starts[1:])):
                group.add(run.tolist(), float(weight), "REPLACE")

    # returns the log line
    def _move_to_invalid(self, unwrap, msg):
        self.moved_to_invalid = True
        return mark_not_unwrapped(import_obj(unwrap.path), msg)

    # a timeout, force kill or missing output, with no engine exit code to describe
    def _fail(self, unwrap, msg):
        logger.add_data("errors", self._move_to_invalid(unwrap, msg))
        self.record_result(unwrap, Result.INVALID)

    def _handle_failure(self, unwrap, ret_code):
        # windows reports exit codes unsigned
        if ret_code >= 2**31:
            ret_code -= 2**32

        described = self.engine.describe_failure(ret_code)
        if described is None:
            self.error_code = ret_code
            tail = unwrap.get_stderr_tail()
            last = last_meaningful_line(tail)
            if last:
                self.error_stderr = last
            if tail:
                print(f"UVgami engine stderr (exit {ret_code}):")
                for line in tail:
                    print(line)
            msg = f"Unknown Engine Error ({ret_code})"
            logger.add_data("errors", self._move_to_invalid(unwrap, msg))
        else:
            msg, move_to_invalid = described
            if move_to_invalid:
                logged = self._move_to_invalid(unwrap, msg)
                logger.add_data("errors", f"{logged} (exit {ret_code})")
            else:
                # one code can cover several causes, the cause is in stderr
                last = last_meaningful_line(unwrap.get_stderr_tail())
                self.error_messages.append(f"{msg} ({last})" if last else msg)

        self.record_result(unwrap, Result.INVALID)

    def _reached_scene(self, unwrap):
        group = unwrap.join_job
        # a discarded or unsettled group is never imported
        if group is not None and (group.discard or not group.is_settled()):
            return False
        # a cancelled transfer deletes the output it was about to read
        return unwrap.transfer_uvs_job not in self.cancelled_transfers

    # a finished piece that never reached the scene counts as cancelled
    def _result_counts(self):
        counts = dict.fromkeys(Result, 0)
        for unwrap, result in self.results:
            if result is Result.FINISHED and not self._reached_scene(unwrap):
                result = Result.CANCELLED
            counts[result] += 1
        return counts

    # its piece already recorded FINISHED, so the counts have to take it back
    def cancel_transfer(self, entry):
        entry.job.cancel()
        self.pending_transfers.remove(entry)
        self.cancelled_transfers.add(entry.job)

    def cancel_session(self):
        # a still pending transfer would otherwise count its piece as finished
        for entry in list(self.pending_transfers):
            self.cancel_transfer(entry)
        for unwrap in list(self.active):
            if unwrap.join_job is not None:
                # finished pieces get discarded instead of joined
                unwrap.join_job.discard = True
            unwrap.stop_process()
            self.record_result(unwrap, Result.CANCELLED)
        self.finish_session()

    def finish_session(self):
        props = self.props
        if props.pack_after_unwrap and self._pack_output_objects:
            valid_objects = [o for o in self._pack_output_objects if check_exists(o)]
            if valid_objects:
                if props.combine_uvs:
                    pack_objects(valid_objects, props, self._pack_topology)
                else:
                    for obj in valid_objects:
                        pack_objects([obj], props, self._pack_topology)

        counts = self._result_counts()
        # one undo step for the whole session
        bpy.ops.ed.undo_push(message="UVgami Unwrap")
        self.end_session()
        self.log_final_status()

        if counts[Result.CANCELLED] != len(self.results):
            msg = []

            # problems that don't fail a mesh but shouldn't read as a clean run
            had_error = bool(
                self.transfer_uv_failed or self.error_code or self.error_messages
            )

            # headline first, the banner shows it alone on the top row
            finished, invalid = counts[Result.FINISHED], counts[Result.INVALID]
            cancelled, stopped = counts[Result.CANCELLED], counts[Result.STOPPED]
            # the counts are per loose part, not per mesh
            if finished and invalid:
                msg.append(f"{invalid} of {finished + invalid} parts failed")
            elif finished and stopped:
                msg.append(f"{stopped} of {finished + stopped} parts stopped")
            elif finished and had_error:
                msg.append("UV unwrap finished with errors")
            elif finished:
                msg.append("UV unwrap complete")
            elif stopped and not invalid:
                msg.append("UV unwrap stopped")
            else:
                msg.append("UV unwrap failed")

            if invalid:
                logger.add_data("errors", "Some meshes failed to unwrap")
            if finished and cancelled:
                objects = "object" if cancelled == 1 else "objects"
                msg.append(f"{cancelled} {objects} cancelled")
            if self.moved_to_invalid:
                msg.append("Check 'UVgami Not Unwrapped'.")

            if self.transfer_uv_failed:
                detail = self.transfer_uv_fail_detail or "unknown reason"
                line = f"UV transfer failed: {detail}."
                if not self.transfer_uv_reason_known:
                    line += " This can happen with cuts or symmetry enabled."
                msg.append(line)

            if self.transfer_uv_split_count > 0:
                count = self.transfer_uv_split_count
                msg.append(f"UV transfer split {count} face(s) crossed by a seam.")

            if self.error_code != 0:
                err_msg = f"An unknown error occurred: {self.error_code}"
                if self.error_stderr:
                    err_msg += f" ({self.error_stderr})"
                msg.append(err_msg)
                logger.add_data("errors", err_msg)

            # a whole queue can fail with the same message
            for err in dict.fromkeys(self.error_messages):
                msg.append(err)
                logger.add_data("errors", err)

            self.summary = msg
            self.summary_failed = bool(invalid or had_error)
            self._show_status()

            if get_preferences().show_popup:
                popup(msg, "UVgami", "INFO")
        else:
            self.clear_summary()

    # a run with problems stays up until the next one
    def _show_status(self):
        text = f"UVgami: {self.summary[0]}"
        if self.summary_failed:
            set_status(f"{text}", "ERROR")
        else:
            set_status(text)
            bpy.app.timers.register(
                functools.partial(set_status, None), first_interval=STATUS_SECONDS
            )

    def log_final_status(self):
        # after the last import, transfer and pack
        logger.update_time()
        counts = self._result_counts()
        cancelled = counts[Result.CANCELLED] == len(self.results)
        logger.change_status("Cancelled" if cancelled else "Complete")
        logger.get_latest().piece_counts = counts
        logger.write_latest()

    def clear_summary(self):
        self.summary = []
        self.summary_failed = False
        set_status(None)

    def _unregister_dispatch(self):
        if self._dispatch_handle is not None:
            if bpy.app.timers.is_registered(self._dispatch_handle):
                bpy.app.timers.unregister(self._dispatch_handle)
            self._dispatch_handle = None

    # tolerant, end_session clears the list before the builder gets here
    def drop_preparing(self, entry):
        if entry in self.preparing:
            self.preparing.remove(entry)

    # a timer that outlives end_session would go negative
    def finished_adding(self):
        self.pieces_still_arriving = max(0, self.pieces_still_arriving - 1)

    # safe to call more than once
    def end_session(self):
        # late import: ops.viewer imports the manager
        from .ops.viewer import stop_viewer_draw

        for entry in list(self.preparing):
            entry.cancel()
        # a file load kills the builder timers that would have cleared these
        self.preparing.clear()
        self.pieces_still_arriving = 0
        for unwrap in list(self._running):
            # release_engine leaves a batch or shared process running
            unwrap.stop_process()
            unwrap.release_engine()
            unwrap.cleanup()
        for unwrap in list(self._queue):
            unwrap.cleanup()
        for transfer in self.pending_transfers:
            transfer.job.cancel()
        self.pending_transfers.clear()
        self._running.clear()
        self._queue.clear()
        self._close_processes()
        self._unregister_dispatch()
        progress_bar.remove()
        # the viewer modal dies with a file load
        stop_viewer_draw()
        self.exit_viewer = True
        self.is_viewer_active = False
        self.is_active = False
        self._pack_output_objects.clear()
        self._pack_topology.clear()
        for path in get_io_dir_paths():
            clear_io_dir(path)
        # erasing the progress bar needs a repaint
        tag_redraw()

    # a file load kills the timer that would have cleaned these up
    def shutdown(self):
        self.end_session()
        self.clear_summary()
        logger.reset()


manager = UnwrapManager()
