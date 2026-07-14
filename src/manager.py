import functools
import time
import traceback
from collections import deque

import bmesh
import bpy
import numpy

from .batch import BatchProcess, last_meaningful_line
from .job import Join
from .logger import logger
from .ops.grid import add_grid, make_grid_img, make_grid_mat
from .ops.uv import pack, show_seams
from .progress_bar import progress_bar
from .reroute_seams import reroute_seams
from .utils.geometry import set_origin
from .utils.io import import_obj
from .utils.mesh import (
    check_collection,
    check_exists,
    edit_restore,
    move_to_collection,
    new_bmesh,
    set_active_any,
    set_bmesh,
)
from .utils.paths import get_extension_dir_path, get_preferences
from .utils.ui import popup, switch_shading


class UnwrapManager:
    def __init__(self):
        self._queue = deque()
        self._running = []
        self._pack_output_objects = []
        self.input = {}
        self.engine = None
        # run context returned by engine.validate, opaque to the manager
        self.engine_ctx = None
        self.is_active = False
        self.is_viewer_active = False
        self._dispatch_handle = None
        # pieces still being exported by drains that add unwraps incrementally;
        # blocks finalizing the session until every drain has drained
        self.hold_count = 0
        # unexported pieces already counted in starting_count, shown as
        # remaining so the finished ratio doesn't shrink as pieces get added
        self.pending_count = 0

    @property
    def active(self):
        """All unwraps (running and queued)"""
        return self._running + list(self._queue)

    def add(self, unwrap):
        """Add an unwrap to the queue."""
        self._queue.append(unwrap)

    def remove_unwrap(self, unwrap):
        """Remove an unwrap from running or queue."""
        if unwrap in self._running:
            self._running.remove(unwrap)
        elif unwrap in self._queue:
            self._queue.remove(unwrap)

    def start(self):
        self.starting_count = len(self._queue) + len(self._running)
        # fill initial slots from queue
        self._fill_slots()
        if get_preferences().show_progress_bar:
            progress_bar.start()
        self.is_active = True
        self.found_invalid_objects = False
        self.transfer_uv_failed = False
        self.transfer_uv_fail_detail = ""
        self.transfer_uv_topology_differed = False
        self.finished_count = 0
        self.cancelled_count = 0
        self.error_code = 0
        self.error_stderr = ""
        self.error_messages = []
        self.current_viewer = None
        self.is_viewer_active = False
        self.exit_viewer = False
        self._pack_output_objects = []
        # register central dispatch timer
        self._dispatch_handle = functools.partial(self._dispatch)
        bpy.app.timers.register(self._dispatch_handle)

    def _fill_slots(self):
        """Start queued unwraps up to the concurrency limit."""
        props = bpy.context.scene.uvgami
        engine = self.engine
        if engine.wants_batch(props):
            if self.hold_count > 0:
                # batch needs the whole queue at once; wait for every drain to
                # finish exporting (the dispatch timer retries each tick)
                return
            if any(u.batch_process is not None for u in self._running):
                # wait for the running batch process to finish
                return
            if len(self._queue) > 1:
                self._start_batch_process(engine, props)
                return
            # a single queued mesh runs the normal solo path
        if props.concurrent and engine.allows_concurrent(props):
            max_concurrent = props.max_cores
        else:
            max_concurrent = 1
        while len(self._running) < max_concurrent and self._queue:
            unwrap = self._queue.popleft()
            unwrap.start_unwrap()
            self._running.append(unwrap)

    def _start_batch_process(self, engine, props):
        """Unwrap every queued mesh in one engine process."""
        unwraps = list(self._queue)
        self._queue.clear()
        args = engine.build_batch_args(
            self.engine_ctx, [u.path for u in unwraps], props
        )
        batch_process = BatchProcess(
            args,
            engine.build_env(self.engine_ctx),
            {unwrap.path.stem: unwrap for unwrap in unwraps},
        )
        for unwrap in unwraps:
            unwrap.join_batch(batch_process)
            self._running.append(unwrap)

    def _dispatch(self):
        """Central dispatch timer that monitors all running unwraps."""
        # guard against running after finish
        if not self.is_active:
            return None

        try:
            prefs = get_preferences()
            completed = []
            failed = []
            requeued = []

            for unwrap in list(self._running):
                # update progress
                unwrap.update_progress()

                # check early stop
                early_stop = bpy.context.scene.uvgami.early_stop
                if (
                    self.engine.supports_early_stop
                    and early_stop != 100
                    and unwrap.progress[0] >= early_stop / 100
                ):
                    unwrap.is_stopped = True

                # update viewer
                if unwrap.viewing:
                    unwrap.update_viewer()

                # if part of batch unwrap, hasn't started and stop button pressed
                if unwrap.is_stopped:
                    self.engine.request_early_stop(unwrap.process)
                    # track when stop was first requested
                    if unwrap.stop_requested_at is None:
                        unwrap.stop_requested_at = time.monotonic()
                    # force kill if process doesn't respond within configured minutes
                    elif (
                        prefs.stop_timeout > 0
                        and time.monotonic() - unwrap.stop_requested_at
                        > prefs.stop_timeout * 60
                    ):
                        unwrap.stop_process()
                        failed.append((unwrap, -3))
                        # already failed this tick, don't let the poll below re-add
                        # it once the killed process reports an exit code
                        continue

                # check if unwrap has exceeded the timeout
                timeout_minutes = bpy.context.scene.uvgami.unwrap_timeout
                if (
                    timeout_minutes > 0
                    and hasattr(unwrap, "started_at")
                    and time.monotonic() - unwrap.started_at > timeout_minutes * 60
                ):
                    unwrap.stop_process()
                    failed.append((unwrap, -2))
                    # already failed this tick, don't let the poll below re-add
                    # it once the killed process reports an exit code
                    continue

                # check process status
                ret_code = unwrap.poll_engine()
                if ret_code is not None:
                    if ret_code == 0 and unwrap.output_path.is_file():
                        completed.append(unwrap)
                    elif ret_code != 0:
                        # a batched mesh that never started and still has its
                        # input goes back to the queue for a fresh batch instead
                        # of inheriting the dead process's exit code
                        stem = unwrap.path.stem
                        if (
                            unwrap.batch_process is not None
                            and unwrap.batch_process.should_retry(stem)
                            and unwrap.path.is_file()
                        ):
                            requeued.append(unwrap)
                        else:
                            failed.append((unwrap, ret_code))

            logger.update_time()

            # update progress bar
            self._update_progress_bar()

            # process completions (each isolated so one failure doesn't block others)
            for unwrap in completed:
                try:
                    self._process_completion(unwrap)
                except Exception:
                    error_list = traceback.format_exc().split("\n")[:-1]
                    logger.add_data("errors", "Error finishing unwrap:")
                    for line in error_list:
                        logger.add_data("errors", line)
                        print(line)
                    # ensure unwrap is removed even on error
                    if unwrap in self._running:
                        self._running.remove(unwrap)
                    unwrap.cleanup()

            # process failures (each isolated)
            for unwrap, ret_code in failed:
                try:
                    self._handle_failure(unwrap, ret_code)
                except Exception:
                    error_list = traceback.format_exc().split("\n")[:-1]
                    logger.add_data("errors", "Error handling unwrap failure:")
                    for line in error_list:
                        logger.add_data("errors", line)
                        print(line)
                    if unwrap in self._running:
                        self._running.remove(unwrap)
                    unwrap.cleanup()

            # requeue detached batch members so _fill_slots re-batches them
            for unwrap in requeued:
                if unwrap in self._running:
                    self._running.remove(unwrap)
                unwrap.leave_batch()
                self._queue.append(unwrap)
            if requeued:
                print(f"UVgami: requeued {len(requeued)} mesh(es) after batch ended")

            # fill empty slots from queue
            self._fill_slots()

            # check if everything is done; a drain still adding pieces holds the
            # session open even when nothing is running or queued yet
            if not self._running and not self._queue and self.hold_count == 0:
                self._finish_batch()
                return None

        except Exception as e:
            # catastrophic dispatch error
            from .handler import handle_error

            handle_error(e, "MIDDLE")
            return None

        return 0.1

    def _update_progress_bar(self):
        """Update the overall progress bar."""
        if not get_preferences().show_progress_bar:
            return

        all_unwraps = self.active
        progress = [numpy.array(unwrap.progress) for unwrap in all_unwraps]
        # pieces still exporting count as remaining, not finished
        for _ in range(self.pending_count):
            progress.append(numpy.array((0, 0, 1)))
        # fill up progress bar with finished unwraps
        for _ in range(self.starting_count - len(progress)):
            progress.append(numpy.array((1, 0, 0)))
        if self.starting_count > 0:
            new_progress = sum(progress) / self.starting_count
            progress_bar.update(new_progress)
            # force redraw of view3D
            bpy.context.view_layer.objects.active = (
                bpy.context.view_layer.objects.active
            )

    def _process_completion(self, unwrap, invalid_pass=False):
        """Process a successfully completed unwrap."""
        if not invalid_pass:
            self.finished_count += 1

        path, edge_path, added_edges, is_import_ready = self._resolve_join(
            unwrap, invalid_pass
        )

        if is_import_ready:
            self._import_and_finalize(unwrap, path, edge_path, added_edges)

        self.exit_viewer = True

        if not invalid_pass:
            # remove from running and clean up files
            if unwrap in self._running:
                self._running.remove(unwrap)
            unwrap.cleanup()

    def _resolve_join(self, unwrap, invalid_pass):
        """Resolve join job state and return final import paths."""
        path = unwrap.output_path
        edge_path = unwrap.edge_path
        added_edges = []
        is_import_ready = True

        if unwrap.join_job is not None:
            if not invalid_pass:
                unwrap.join_job.unwrapped.append(unwrap)
            # get all paths of finished unwraps before joining
            if unwrap.join_job.is_completed() and unwrap.join_job.count > 1:
                data = unwrap.join_job.finish(unwrap)
                path = data[0]
                edge_path = data[1]
                added_edges = data[2]
            # if the count is 1, that means all but one unwrap of group was cancelled
            elif not (unwrap.join_job.is_completed() and unwrap.join_job.count == 1):
                # in all other cases, wait until last unwrap finishes before importing
                is_import_ready = False

        return path, edge_path, added_edges, is_import_ready

    def _import_and_finalize(self, unwrap, path, edge_path, added_edges):
        """Import the unwrapped OBJ and apply all post-processing."""
        props = bpy.context.scene.uvgami

        # reroute seams before importing
        if unwrap.preserve_job is not None and props.maintain_mode == "FULL":
            reroute_seams(path, edge_path)

        old_active = bpy.context.view_layer.objects.active
        if old_active is None:
            old_active = set_active_any()
        output = import_obj(path, f"{unwrap.input_name}_unwrapped")
        # the new obj importer changes the active object
        if bpy.app.version >= (3, 2, 0):
            bpy.context.view_layer.objects.active = old_active

        set_origin(output, unwrap.origin)

        # set materials
        materials = [
            bpy.data.materials.get(m_name)
            for m_name in unwrap.materials
            if m_name is not None
        ]
        for m in materials:
            output.data.materials.append(m)

        # restore per-face material indices
        if unwrap.join_job is not None and len(unwrap.join_job.unwrapped) > 1:
            # concatenate material indices from all joined unwraps
            combined_indices = []
            for u in unwrap.join_job.unwrapped:
                combined_indices.extend(u.material_indices)
            if len(combined_indices) == len(output.data.polygons):
                output.data.polygons.foreach_set("material_index", combined_indices)
        elif len(unwrap.material_indices) == len(output.data.polygons):
            # single object (no join), restore directly
            output.data.polygons.foreach_set("material_index", unwrap.material_indices)

        if unwrap.preserve_job is not None:
            unwrap.preserve_job.finish(unwrap, output, added_edges)

        if unwrap.cleanup_job is not None:
            unwrap.cleanup_job.finish(self.input[unwrap.cleanup_job])

        if unwrap.symmetrize_job is not None:
            unwrap.symmetrize_job.finish(output)

        if unwrap.merge_cuts:
            bm = new_bmesh(output)
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
            set_bmesh(bm, output)

        # automatically add grid material to final object
        if props.auto_grid:
            grid_img = make_grid_img()
            add_grid(output, make_grid_mat(grid_img))

        if props.pack_after_unwrap:
            self._pack_output_objects.append(output)

        # show seams
        edit_restore([output], show_seams)

        # shade smooth
        if unwrap.shade_smooth:
            output.data.polygons.foreach_set(
                "use_smooth", [True] * len(output.data.polygons)
            )
            if unwrap.auto_smooth != -1:
                if bpy.app.version >= (4, 1, 0):
                    pass
                else:
                    output.data.use_auto_smooth = True
                    output.data.auto_smooth_angle = unwrap.auto_smooth

        self._restore_vertex_groups(unwrap, output)

        logger.add_data("objects", unwrap.input_name)

        # transfer UVs to original input mesh if enabled
        if unwrap.transfer_uvs_job is not None:
            input_mesh = self.input[unwrap.transfer_uvs_job]
            # replace output with input in pack list before finish deletes output
            pack_replaced = False
            if props.pack_after_unwrap:
                for i, obj in enumerate(self._pack_output_objects):
                    if obj == output:
                        self._pack_output_objects[i] = input_mesh
                        pack_replaced = True
                        break
            outcome = unwrap.transfer_uvs_job.finish(input_mesh, output)
            if outcome.applied:
                if not outcome.exact_topology:
                    self.transfer_uv_topology_differed = True
                return
            else:
                # transfer failed, restore pack list if we changed it
                if pack_replaced:
                    for i, obj in enumerate(self._pack_output_objects):
                        if obj == input_mesh:
                            self._pack_output_objects[i] = output
                            break
                self.transfer_uv_failed = True
                self.transfer_uv_fail_detail = outcome.detail
                logger.add_data(
                    "errors",
                    f"UV transfer failed ({outcome.detail}), keeping output",
                )

        collection = check_collection("UVgami Unwrapped", bpy.context.scene.collection)
        move_to_collection(output, collection)

    def _restore_vertex_groups(self, unwrap, output):
        """Restore pre-captured vertex groups to the output mesh."""
        if unwrap.join_job is not None and len(unwrap.join_job.unwrapped) > 1:
            # combine vertex groups from all joined unwraps with offset indices
            combined_groups = {}
            v_offset = 0
            for u in unwrap.join_job.unwrapped:
                for group_name, weights in u.vertex_groups.items():
                    if group_name not in combined_groups:
                        combined_groups[group_name] = {}
                    for v_idx, weight in weights.items():
                        combined_groups[group_name][v_idx + v_offset] = weight
                v_offset += u.vertex_count
            groups_data = combined_groups
        else:
            groups_data = unwrap.vertex_groups

        for group_name, weights in groups_data.items():
            new_group = output.vertex_groups.new(name=group_name)
            for v_idx, weight in weights.items():
                if v_idx < len(output.data.vertices):
                    new_group.add([v_idx], weight, "REPLACE")

    def _handle_failure(self, unwrap, ret_code):
        """Handle an unwrap process that exited with a non-zero code."""
        prefs = get_preferences()
        msg = ""

        # convert unsigned int
        THRESHOLD = 2147483648
        ADJUSTMENT = 4294967296
        if ret_code >= THRESHOLD:
            ret_code -= ADJUSTMENT

        move_to_invalid = False
        # manager-synthetic codes for timeout and force-kill
        if ret_code == -2:
            elapsed = (time.monotonic() - unwrap.started_at) / 60
            msg = f"Timed out after {elapsed:.1f} minutes"
            move_to_invalid = True
        elif ret_code == -3:
            msg = "Stop timed out (force killed)"
            move_to_invalid = True
        else:
            described = self.engine.describe_failure(ret_code)
            if described is not None:
                msg, move_to_invalid = described
                if not move_to_invalid:
                    self.error_messages.append(msg)
            else:
                self.error_code = ret_code
                tail = unwrap.get_stderr_tail()
                last = last_meaningful_line(tail)
                if last:
                    self.error_stderr = last
                if tail:
                    # full tail to the console so the whole traceback is findable
                    print(f"UVgami engine stderr (exit {ret_code}):")
                    for line in tail:
                        print(line)

        if move_to_invalid:
            if prefs.invalid_collection:
                # move to collection for invalid meshes
                invalid_obj = import_obj(unwrap.path)
                collection = check_collection(
                    "UVgami Not Unwrapped", bpy.context.scene.collection
                )
                move_to_collection(invalid_obj, collection)
                invalid_name = f"{invalid_obj.name}: {msg}"
                invalid_obj.name = invalid_name
                invalid_obj.hide_set(True)
                logger.add_data("errors", invalid_name)

            self.found_invalid_objects = True

        found_job = None
        # count has to be reduced because this object won't be unwrapped
        for job in unwrap.jobs:
            if job.count > 1:
                job.count = job.count - 1
                # found_job can't be a Cleanup job because the unwrapped list
                # will be empty
                if isinstance(job, Join):
                    found_job = job

        # remove from running
        if unwrap in self._running:
            self._running.remove(unwrap)
        unwrap.release_engine()
        unwrap.cleanup()

        # if the invalid obj has jobs that are complete with the now reduced count
        # that means that this unwrap was the last of the group
        if found_job is not None and found_job.is_completed():
            # use the last completed unwrap
            self._process_completion(found_job.unwrapped[-1], invalid_pass=True)

    def _finish_batch(self):
        """Called when all unwraps are done (completed, failed, or cancelled)."""
        props = bpy.context.scene.uvgami
        if props.pack_after_unwrap and self._pack_output_objects:
            valid_objects = [o for o in self._pack_output_objects if check_exists(o)]
            if valid_objects:
                if props.combine_uvs:
                    edit_restore(valid_objects, pack)
                else:
                    for obj in valid_objects:
                        edit_restore([obj], pack)

        self.finish()

        # don't show popup if all unwraps were cancelled
        if self.cancelled_count != self.starting_count:
            logger.change_status("Complete")
            if get_preferences().show_popup:
                msg = []

                if self.finished_count > 0:
                    msg.append("UV unwrap complete!")

                if self.found_invalid_objects:
                    msg.append("Some meshes were not unwrapped.")
                    msg.append("Check 'UVgami Not Unwrapped'.")
                    logger.add_data(
                        "errors", "Some meshes were not able to be unwrapped"
                    )

                if self.transfer_uv_failed:
                    detail = self.transfer_uv_fail_detail or "unknown reason"
                    msg.append(
                        f"UV transfer failed: {detail}."
                        " This can happen with cuts or symmetry enabled."
                    )

                if self.transfer_uv_topology_differed:
                    msg.append(
                        "UV transfer: input and output meshes have different topology."
                        " Enable 'Preserve Mesh' for best results."
                    )

                if self.error_code != 0:
                    err_msg = f"An unknown error occurred: {self.error_code}"
                    if self.error_stderr:
                        err_msg += f" ({self.error_stderr})"
                    msg.append(err_msg)
                    logger.add_data("errors", err_msg)

                for err in self.error_messages:
                    msg.append(err)
                    logger.add_data("errors", err)

                popup(msg, "UVgami", "INFO")
        else:
            logger.change_status("Cancelled")

    def _unregister_dispatch(self):
        """Unregister the dispatch timer if active."""
        if self._dispatch_handle is not None:
            if bpy.app.timers.is_registered(self._dispatch_handle):
                bpy.app.timers.unregister(self._dispatch_handle)
            self._dispatch_handle = None

    def finish(self):
        """Clean up everything."""
        self._unregister_dispatch()
        progress_bar.remove()
        self.is_active = False
        self._running.clear()
        self._queue.clear()
        self._pack_output_objects.clear()

        if (
            bpy.context.scene.uvgami.auto_grid
            and getattr(self, "finished_count", 0) > 0
        ):
            switch_shading("MATERIAL")

        # clean up io folders
        for file in (get_extension_dir_path() / "input").iterdir():
            file.unlink()
        for file in (get_extension_dir_path() / "output").iterdir():
            file.unlink()

    def release_jobs(self, jobs):
        """Decrement each job's count for a piece that won't be unwrapped and
        apply the Join cancelled/finished adjustment when its group empties.
        Shared by cancel_with_bookkeeping and the drain's group-stop skip."""
        for job in jobs:
            job.count = job.count - 1
            # if it was the last one
            if isinstance(job, Join) and job.count - len(job.unwrapped) == 0:
                # this makes it so the popup doesn't show if all cancelled
                self.finished_count -= len(job.unwrapped)
                self.cancelled_count += len(job.unwrapped)

    def cancel_unwrap(self, unwrap):
        """Cancel a specific unwrap."""
        self.cancelled_count += 1
        unwrap.release_engine()
        self.remove_unwrap(unwrap)
        unwrap.cleanup()
        self.exit_viewer = True
        # update 3d view to remove progress bar
        bpy.context.view_layer.objects.active = bpy.context.view_layer.objects.active

    def stop_all(self):
        """Stop all running processes and clean up."""
        for unwrap in list(self._running):
            unwrap.stop_process()
            unwrap.cleanup()
        for unwrap in list(self._queue):
            unwrap.cleanup()
        self._running.clear()
        self._queue.clear()
        self._unregister_dispatch()
        progress_bar.remove()
        self.is_active = False


manager = UnwrapManager()
