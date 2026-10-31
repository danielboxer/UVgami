# import copy
import functools
import multiprocessing
import time
from collections import deque, namedtuple

import bpy
import mathutils
import numpy

from .batch import BatchProcess, last_meaningful_line
from .hard_surface import finish_hard_surface
from .job import Join, Result, TransferReport
from .logger import logger
from .no_bpy import faces_by_edge
from .no_bpy.uv_transfer import AMBIGUOUS_GEOMETRY
from .ops.grid import add_grid, grid_material
from .ops.uv import pack_objects, show_seams
from .part_overrides import OVERRIDE_ATTRIBUTES
from .progress_bar import IDLE_PROGRESS, progress_bar
from .similar import write_twin_output
from .utils.io import import_obj
from .utils.mesh import (
    add_shading_modifiers,
    check_collection,
    check_exists,
    edit_restore,
    face_vertices,
    mark_not_unwrapped,
    move_to_collection,
)
from .utils.paths import clear_io_dir, get_io_dir_paths, get_preferences
from .utils.ui import popup, set_status, switch_solid_color, tag_redraw

STATUS_SECONDS = 5
DISPATCH_SECONDS = 0.1
QUEUED_DISPATCH_SECONDS = 0.02
TRANSFER_PROGRESS_SHARE = 0.4
# sidebar rebuilds during a click drop the click
PANEL_REDRAW_SECONDS = 1.0


class SettingsCopy:
    def __init__(self, props):
        for prop in props.bl_rna.properties:
            if prop.identifier == "rna_type":
                continue
            value = getattr(props, prop.identifier)
            if prop.type == "POINTER":
                value = SettingsCopy(value)
            elif isinstance(value, set):
                value = set(value)
            setattr(self, prop.identifier, value)
        # bl_rna.properties excludes plain python properties like avoid_seams
        for cls in type(props).__mro__:
            for name, attribute in vars(cls).items():
                if isinstance(attribute, property) and not hasattr(self, name):
                    setattr(self, name, getattr(props, name))


PendingTransfer = namedtuple(
    "PendingTransfer", ["job", "output", "pack_index", "name", "started_at"]
)
Preparing = namedtuple("Preparing", ["name", "cancel"])
EngineFailure = namedtuple("EngineFailure", ["unwrap", "handle", "detail"])
STOP_SECONDS = 180


class UnwrapManager:
    def __init__(self):
        self._queue = deque()
        self._running = []
        self._shared_processes = []
        self._batch_processes = []
        self.engine = None
        self.engine_ctx = None
        self.is_active = False
        self.in_uv_editor = False
        self._dispatch_handle = None
        self.pieces_still_arriving = 0
        self.preparing = []
        self.summary = []
        self.summary_failed = False
        self._reset_session()

    def _reset_session(self):
        self.props = None
        self.moved_to_invalid = False
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
        self.progress = numpy.array(IDLE_PROGRESS)

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

    def claimed_paths(self):
        claimed = {unwrap.path for unwrap in self.active}
        claimed.update(unwrap.path for unwrap, _ in self.results)
        return claimed

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
        self.props = SettingsCopy(bpy.context.scene.uvgami)
        self._fill_slots()
        if get_preferences().show_progress_bar:
            progress_bar.start(uv_editor=uv_editor)
        self.in_uv_editor = uv_editor
        self.is_active = True
        self.clear_summary()
        self._dispatch_handle = self._dispatch
        bpy.app.timers.register(self._dispatch_handle)

    def _fill_slots(self):
        props = self.props
        engine = self.engine
        if engine.batches_queue(props):
            if self.pieces_still_arriving > 0:
                return
            if any(unwrap.batch_process is not None for unwrap in self._running):
                return
            if len(self._queue) > 1:
                self._start_batch_process()
                return
        max_concurrent = 1 if engine.batches_queue(props) else props.max_cores
        for unwrap in list(self._queue):
            if len(self._running) >= max_concurrent:
                break
            if not unwrap.is_exported:
                break
            if unwrap.copy_of is not None:
                continue
            self._queue.remove(unwrap)
            pieces = unwrap.join_job.expected if unwrap.join_job is not None else 1
            # optcuts uses every core with -t 0
            threads = 0
            if pieces >= props.max_cores:
                threads = max(1, multiprocessing.cpu_count() // props.max_cores)
            shared_args = engine.build_shared_args(
                self.engine_ctx, unwrap.path, props, threads
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

    def _send_to_shared_process(self, unwrap, args):
        # spawning a process costs 50ms inside Blender
        process = next(
            (
                process
                for process in self._shared_processes
                if process.is_idle() and process.args == args
            ),
            None,
        )
        if process is None:
            process = BatchProcess(args, self.engine.build_env(self.engine_ctx))
            self._shared_processes.append(process)
        process.send(unwrap.path, unwrap)
        unwrap.join_batch(process)

    def _close_processes(self):
        # shared processes exit when their stdin closes
        for process in self._shared_processes:
            process.close()
        self._shared_processes.clear()
        for process in self._batch_processes:
            # batch processes run their argv to the end
            if process.process.poll() is None:
                self.engine.stop(process.process, self.engine_ctx)
            process.close()
        self._batch_processes.clear()

    def _start_batch_process(self):
        unwraps = [unwrap for unwrap in self._queue if unwrap.copy_of is None]
        if not unwraps:
            return
        self._queue = deque(unwrap for unwrap in self._queue if unwrap not in unwraps)
        args = self.engine.build_batch_args(
            self.engine_ctx, [unwrap.path for unwrap in unwraps], self.props
        )
        batch_process = BatchProcess(
            args,
            self.engine.build_env(self.engine_ctx),
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
                            EngineFailure(
                                unwrap=unwrap,
                                handle=self._fail_without_exit_code,
                                detail="Stop timed out (force killed)",
                            )
                        )
                        continue

                timeout_minutes = self.props.unwrap_timeout
                if (
                    timeout_minutes > 0
                    and not unwrap.is_stopped
                    and unwrap.started_at is not None
                    and time.monotonic() - unwrap.started_at > timeout_minutes * 60
                ):
                    if self.engine.supports_early_stop:
                        unwrap.is_stopped = True
                        self.error_messages.append(
                            f"{unwrap.input_name}: timed out,"
                            " stopped with a partial result"
                        )
                    else:
                        unwrap.stop_process()
                        elapsed = (time.monotonic() - unwrap.started_at) / 60
                        failed.append(
                            EngineFailure(
                                unwrap=unwrap,
                                handle=self._fail_without_exit_code,
                                detail=f"Timed out after {elapsed:.1f} minutes",
                            )
                        )
                        continue

                exit_code = unwrap.poll_engine()
                if exit_code is not None:
                    if exit_code == 0 and unwrap.output_path.is_file():
                        completed.append(unwrap)
                    elif exit_code == 0:
                        failed.append(
                            EngineFailure(
                                unwrap=unwrap,
                                handle=self._fail_without_exit_code,
                                detail="Engine produced no output",
                            )
                        )
                    else:
                        stem = unwrap.path.stem
                        if (
                            unwrap.batch_process is not None
                            and unwrap.batch_process.should_retry(stem)
                            and unwrap.path.is_file()
                        ):
                            requeued.append(unwrap)
                        else:
                            failed.append(
                                EngineFailure(
                                    unwrap=unwrap,
                                    handle=self._handle_failure,
                                    detail=exit_code,
                                )
                            )

            if self._running or self._queue:
                logger.mark_unwrapping()
            logger.update_time()
            self._update_progress_bar()

            for unwrap in completed:
                self._settle_step(unwrap, "finishing", self._process_completion)

            for failure in failed:
                self._settle_step(
                    failure.unwrap,
                    "handling the failure of",
                    failure.handle,
                    failure.detail,
                )

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

        except Exception as error:
            from .handler import handle_dispatch_error

            handle_dispatch_error(error)
            return None

        return QUEUED_DISPATCH_SECONDS if self._queue else DISPATCH_SECONDS

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
        reported = {entry.job: entry.job.progress for entry in self.pending_transfers}
        transferred = numpy.array([reported.get(job, 0.0) for job in jobs])
        share = TRANSFER_PROGRESS_SHARE * has_transfer
        progress = engine * (1 - share)[:, None]
        progress[:, 0] += share * transferred
        progress[:, 2] += share * (1 - transferred)
        progress = progress.mean(axis=0)
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

    def _panel_state(self):
        return (
            self.elapsed_seconds,
            len(self.results),
            self.is_viewer_active,
            tuple(entry.name for entry in self.preparing),
            tuple(transfer.name for transfer in self.pending_transfers),
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
                    path, added_edges = unwrap.output_path, unwrap.export.added_edges
                self._import_and_finalize(unwrap, path, added_edges)
                # without this undo push, a ctrl z mid-session discards the result
                if self._running or self._queue or self.pieces_still_arriving:
                    bpy.ops.ed.undo_push(message="UVgami Unwrap")
            except Exception:
                logger.log_traceback("Error finishing unwrap:")
                self.error_messages.append(
                    f"Error finishing {unwrap.input_name}, see the console"
                )
        self._to_import.clear()

    def _settle_step(self, unwrap, doing, step, *args):
        try:
            step(unwrap, *args)
        except Exception:
            message = f"Error {doing} {unwrap.input_name}"
            logger.log_traceback(f"{message}:")
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
        export = unwrap.export
        pieces = unwrap.join_job.finished if unwrap.join_job is not None else [unwrap]

        matrix = output.matrix_world
        offset = mathutils.Matrix.Translation(-(matrix.inverted() @ export.origin))
        output.data.transform(offset)
        matrix.translation += export.origin - matrix.translation

        for material_name in export.materials:
            output.data.materials.append(
                None if material_name is None else bpy.data.materials.get(material_name)
            )

        # these restore before the preserve job so untriangulated faces carry them
        self._restore_face_data(pieces, output, "material_index", "material_indices")
        self._restore_face_data(pieces, output, "use_smooth", "face_smooth")
        self._restore_overrides(pieces, output)
        add_shading_modifiers(output, export.shading_modifiers)

        if unwrap.preserve_job is not None:
            unwrap.preserve_job.finish(output, added_edges)

        if unwrap.hide_job is not None:
            unwrap.hide_job.finish()

        symmetrize_job = unwrap.symmetrize_job
        if symmetrize_job is not None:
            symmetrize_job.finish(output, unwrap.matrix)

        if any(piece.preseeded for piece in pieces):
            face_count = len(output.data.polygons)
            ranges = [(0, face_count)]
            if len(pieces) > 1:
                ranges = []
                start = 0
                for piece in pieces:
                    stop = start + len(piece.export.material_indices)
                    if piece.preseeded:
                        ranges.append((start, stop))
                    start = stop
                if start != face_count:
                    ranges = [(0, face_count)]
            faces = face_vertices(output.data)
            finish_hard_surface(output, faces, ranges)
            self._pack_topology[output] = (faces, faces_by_edge(faces))

        if props.pack_after_unwrap:
            self._pack_output_objects.append(output)

        self._restore_vertex_groups(pieces, output)

        # vertex groups restore by index, and the weld renumbers vertices
        weld_job = unwrap.segment_weld_job
        if weld_job is not None and unwrap.transfer_uvs_job is None:
            weld_job.finish(output)
            self._pack_topology.pop(output, None)

        edit_restore([output], show_seams)

        if unwrap.transfer_uvs_job is not None:
            job = unwrap.transfer_uvs_job
            pack_index = None
            if props.pack_after_unwrap:
                pack_index = self._pack_output_objects.index(output)
            group = unwrap.join_job
            missing_pieces = group is not None and len(group.finished) < group.expected
            if missing_pieces and not job.allows_missing_pieces:
                failed = group.expected - len(group.finished)
                self.transfer_uv_reason_known = True
                report = TransferReport(
                    applied=False,
                    split_count=0,
                    detail=f"{failed} of {group.expected} parts were not unwrapped",
                )
                self._settle_transfer(job, output, pack_index, report)
                return
            report = job.start(output)
            if report is None:
                self.pending_transfers.append(
                    PendingTransfer(
                        job, output, pack_index, unwrap.input_name, time.monotonic()
                    )
                )
                for collection in output.users_collection:
                    collection.objects.unlink(output)
                return
            self._settle_transfer(job, output, pack_index, report)
            return

        self._show_output(output)

    def _settle_transfer(self, job, output, pack_index, report):
        job.settled = True
        input_mesh = job.input_mesh
        if report.applied:
            self.transfer_uv_split_count += report.split_count
            if pack_index is not None and job.repack_input:
                self._pack_output_objects[pack_index] = input_mesh
            replacement = job.replacement
            if replacement is None:
                self._add_auto_grid(input_mesh)
                return
            if pack_index is not None:
                self._pack_output_objects[pack_index] = replacement
            output = replacement
        else:
            if check_exists(input_mesh):
                input_mesh.hide_set(True)
            self.transfer_uv_failed = True
            self.transfer_uv_fail_detail = report.detail
            if report.reason == AMBIGUOUS_GEOMETRY:
                self.transfer_uv_reason_known = True
            logger.add_error(f"UV transfer failed ({report.detail}), keeping output")
            if not check_exists(output):
                return

        self._show_output(output)

    def _finish_transfers(self):
        timeout_minutes = self.props.unwrap_timeout
        for entry in list(self.pending_transfers):
            job, output, pack_index = entry.job, entry.output, entry.pack_index
            if (
                timeout_minutes > 0
                and time.monotonic() - entry.started_at > timeout_minutes * 60
            ):
                self.cancel_transfer(entry)
                self.transfer_uv_failed = True
                self.transfer_uv_reason_known = True
                self.transfer_uv_fail_detail = f"{entry.name} timed out"
                continue
            try:
                report = job.poll()
            except Exception:
                logger.log_traceback("Error finishing UV transfer:")
                report = TransferReport(
                    applied=False, split_count=0, detail="see the console"
                )
            if report is None:
                continue
            self.pending_transfers.remove(entry)
            try:
                self._settle_transfer(job, output, pack_index, report)
            except Exception:
                logger.log_traceback("Error finishing UV transfer:")
                self.error_messages.append(
                    f"Error finishing the UV transfer for {entry.name}, see the console"
                )

    def _add_auto_grid(self, obj):
        if self.props.auto_grid:
            add_grid(obj, grid_material())
            switch_solid_color("TEXTURE")

    def _show_output(self, output):
        self._add_auto_grid(output)
        move_to_collection(output, check_collection("UVgami Unwrapped"))

    def _restore_face_data(self, pieces, output, attribute, field):
        values = []
        for piece in pieces:
            values.extend(getattr(piece.export, field))
        if len(values) == len(output.data.polygons):
            output.data.polygons.foreach_set(attribute, values)

    def _restore_overrides(self, pieces, output):
        overrides = {
            name: [value for piece in pieces for value in piece.export.overrides[name]]
            for name in pieces[0].export.overrides
        }
        face_count = len(output.data.polygons)
        if not all(len(values) == face_count for values in overrides.values()):
            return
        attributes = output.data.attributes
        for name, values in overrides.items():
            attribute = attributes.get(name) or attributes.new(
                name, OVERRIDE_ATTRIBUTES[name], "FACE"
            )
            attribute.data.foreach_set("value", values)

    def _restore_vertex_groups(self, pieces, output):
        combined = {}
        offset = 0
        for piece in pieces:
            for name, (indices, weights) in piece.export.vertex_groups.items():
                combined.setdefault(name, []).append((indices + offset, weights))
            offset += piece.export.vertex_count

        vertex_count = len(output.data.vertices)
        for name, parts in combined.items():
            indices = numpy.concatenate([indices for indices, _ in parts])
            weights = numpy.concatenate([weights for _, weights in parts])
            inside = indices < vertex_count
            indices, weights = indices[inside], weights[inside]
            # group.add calls per vertex are slow on a dense mesh
            order = numpy.argsort(weights, kind="stable")
            values, starts = numpy.unique(weights[order], return_index=True)
            group = output.vertex_groups.new(name=name)
            for weight, run in zip(values, numpy.split(indices[order], starts[1:])):
                group.add(run.tolist(), float(weight), "REPLACE")

    def _move_to_invalid_log_line(self, unwrap, message):
        self.moved_to_invalid = True
        return mark_not_unwrapped(import_obj(unwrap.path), message)

    def _fail_without_exit_code(self, unwrap, message):
        logger.add_error(self._move_to_invalid_log_line(unwrap, message))
        self.record_result(unwrap, Result.INVALID)

    def _handle_failure(self, unwrap, exit_code):
        # windows reports exit codes unsigned
        if exit_code >= 2**31:
            exit_code -= 2**32

        described = self.engine.describe_failure(exit_code)
        if described is None:
            self.error_code = exit_code
            tail = unwrap.get_stderr_tail()
            last = last_meaningful_line(tail)
            if last:
                self.error_stderr = last
            if tail:
                print(f"UVgami engine stderr (exit {exit_code}):")
                for line in tail:
                    print(line)
            message = f"Unknown Engine Error ({exit_code})"
            logger.add_error(self._move_to_invalid_log_line(unwrap, message))
        else:
            message, move_to_invalid = described
            if move_to_invalid:
                logged = self._move_to_invalid_log_line(unwrap, message)
                logger.add_error(f"{logged} (exit {exit_code})")
            else:
                # one exit code can cover several causes
                last = last_meaningful_line(unwrap.get_stderr_tail())
                self.error_messages.append(f"{message} ({last})" if last else message)

        self.record_result(unwrap, Result.INVALID)

    def _reached_scene(self, unwrap):
        group = unwrap.join_job
        if group is not None and (group.discard or not group.is_settled()):
            return False
        return unwrap.transfer_uvs_job not in self.cancelled_transfers

    def _result_counts(self):
        counts = dict.fromkeys(Result, 0)
        for unwrap, result in self.results:
            if result is Result.FINISHED and not self._reached_scene(unwrap):
                result = Result.CANCELLED
            counts[result] += 1
        return counts

    def cancel_transfer(self, entry):
        entry.job.cancel()
        self.pending_transfers.remove(entry)
        self.cancelled_transfers.add(entry.job)

    def cancel_session(self):
        for entry in list(self.pending_transfers):
            self.cancel_transfer(entry)
        for unwrap in list(self.active):
            if unwrap.join_job is not None:
                unwrap.join_job.discard = True
            unwrap.stop_process()
            self.record_result(unwrap, Result.CANCELLED)
        self.finish_session()

    def finish_session(self):
        props = self.props
        if props.pack_after_unwrap and self._pack_output_objects:
            valid_objects = [
                obj for obj in self._pack_output_objects if check_exists(obj)
            ]
            if valid_objects:
                if props.combine_uvs:
                    pack_objects(valid_objects, props, self._pack_topology)
                else:
                    for obj in valid_objects:
                        pack_objects([obj], props, self._pack_topology)

        counts = self._result_counts()
        bpy.ops.ed.undo_push(message="UVgami Unwrap")
        self.end_session()
        all_cancelled = counts[Result.CANCELLED] == len(self.results)
        logger.update_time()
        logger.get_latest().status = "Cancelled" if all_cancelled else "Complete"
        logger.get_latest().piece_counts = counts
        logger.write_latest()

        if not all_cancelled:
            lines = []

            had_error = bool(
                self.transfer_uv_failed or self.error_code or self.error_messages
            )

            finished, invalid = counts[Result.FINISHED], counts[Result.INVALID]
            cancelled, stopped = counts[Result.CANCELLED], counts[Result.STOPPED]
            if finished and invalid:
                lines.append(f"{invalid} of {finished + invalid} parts failed")
            elif finished and stopped:
                lines.append(f"{stopped} of {finished + stopped} parts stopped")
            elif finished and had_error:
                lines.append("UV unwrap finished with errors")
            elif finished:
                lines.append("UV unwrap complete")
            elif stopped and not invalid:
                lines.append("UV unwrap stopped")
            else:
                lines.append("UV unwrap failed")

            if invalid:
                logger.add_error("Some meshes failed to unwrap")
            if finished and cancelled:
                objects = "object" if cancelled == 1 else "objects"
                lines.append(f"{cancelled} {objects} cancelled")
            if self.moved_to_invalid:
                lines.append("Check 'UVgami Not Unwrapped'.")

            if self.transfer_uv_failed:
                detail = self.transfer_uv_fail_detail or "unknown reason"
                line = f"UV transfer failed: {detail}."
                if not self.transfer_uv_reason_known:
                    line += " This can happen with cuts or symmetry enabled."
                lines.append(line)

            if self.transfer_uv_split_count > 0:
                count = self.transfer_uv_split_count
                lines.append(f"UV transfer split {count} face(s) crossed by a seam.")

            if self.error_code != 0:
                error_message = f"An unknown error occurred: {self.error_code}"
                if self.error_stderr:
                    error_message += f" ({self.error_stderr})"
                lines.append(error_message)
                logger.add_error(error_message)

            for error_message in dict.fromkeys(self.error_messages):
                lines.append(error_message)
                logger.add_error(error_message)

            self.summary = lines
            self.summary_failed = bool(invalid or had_error)
            self._show_status()

            if get_preferences().show_popup:
                popup(lines, "UVgami", "INFO")
        else:
            self.clear_summary()

    def _show_status(self):
        text = f"UVgami: {self.summary[0]}"
        if self.summary_failed:
            set_status(f"{text}", "ERROR")
        else:
            set_status(text)
            bpy.app.timers.register(
                functools.partial(set_status, None), first_interval=STATUS_SECONDS
            )

    def clear_summary(self):
        self.summary = []
        self.summary_failed = False
        set_status(None)

    def drop_preparing(self, entry):
        # end_session can clear the list before this call
        if entry in self.preparing:
            self.preparing.remove(entry)

    def finished_adding(self):
        # builder timers can call this after end_session sets the count to 0
        self.pieces_still_arriving = max(0, self.pieces_still_arriving - 1)

    def end_session(self):
        # imported here because ops.viewer imports the manager
        from .ops.viewer import stop_viewer_draw

        for entry in list(self.preparing):
            entry.cancel()
        # file loads remove the builder timers that clear these
        self.preparing.clear()
        self.pieces_still_arriving = 0
        for unwrap in list(self._running):
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
        if self._dispatch_handle is not None:
            if bpy.app.timers.is_registered(self._dispatch_handle):
                bpy.app.timers.unregister(self._dispatch_handle)
            self._dispatch_handle = None
        progress_bar.remove()
        # file loads end the viewer modal
        stop_viewer_draw()
        self.exit_viewer = True
        self.is_viewer_active = False
        self.is_active = False
        self._pack_output_objects.clear()
        self._pack_topology.clear()
        for path in get_io_dir_paths():
            clear_io_dir(path)
        tag_redraw()

    def shutdown(self):
        # file loads remove the timers that end a session
        self.end_session()
        self.clear_summary()
        logger.reset()


manager = UnwrapManager()
