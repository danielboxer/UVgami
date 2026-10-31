import bpy

from ..job import Result
from ..manager import manager
from ..objfile import merge_obj_files
from ..utils.io import import_obj
from ..utils.mesh import mark_not_unwrapped
from ..utils.ui import count_label, tag_redraw


def group_targets(job_id):
    return [
        unwrap
        for unwrap in manager.active
        if unwrap.join_job is not None and unwrap.join_job.job_id == job_id
    ]


def piece_target(stem):
    return [unwrap for unwrap in manager.active if unwrap.path.stem == stem]


def confirm_discard(operator, context, event, *, finished_count, title):
    return context.window_manager.invoke_confirm(
        operator,
        event,
        title=title,
        message=f"{count_label(finished_count, 'finished piece')} will be discarded",
        icon="WARNING",
    )


def drop_unwrap(unwrap, invalid_label, result):
    unwrap.cancel_solve()
    if invalid_label is not None and unwrap.path.is_file():
        # record_result deletes unwrap.path
        mark_not_unwrapped(import_obj(unwrap.path), invalid_label)
        manager.moved_to_invalid = True
    manager.record_result(unwrap, result)


class UVGAMI_OT_stop(bpy.types.Operator):
    bl_idname = "uvgami.stop"
    bl_label = "Stop"
    bl_description = "Stop UV unwrap"

    stem: bpy.props.StringProperty(options={"SKIP_SAVE"})
    job_id: bpy.props.IntProperty(options={"SKIP_SAVE"})

    def execute(self, context):
        unwraps = group_targets(self.job_id) if self.job_id else piece_target(self.stem)

        to_cancel = []
        for unwrap in unwraps:
            if unwrap.is_solving and manager.engine.supports_early_stop:
                unwrap.is_stopped = True
                delivered = manager.engine.request_early_stop(unwrap.process)
                if not delivered and unwrap.process.poll() is None:
                    self.report({"ERROR"}, "Could not stop unwrap")
            elif not unwrap.is_running:
                to_cancel.append(unwrap)

        self._cancel_collected(to_cancel)
        if to_cancel:
            manager.exit_viewer = True
            tag_redraw()
        return {"FINISHED"}

    def _cancel_collected(self, to_cancel):
        groups = {}
        singles = []
        for unwrap in to_cancel:
            if unwrap.join_job is None:
                singles.append(unwrap)
            else:
                groups.setdefault(id(unwrap.join_job), []).append(unwrap)

        for group in groups.values():
            if len(group) < 2:
                singles.extend(group)
                continue
            self._import_merged_group(group)
            for unwrap in group:
                drop_unwrap(unwrap, None, Result.STOPPED)

        for unwrap in singles:
            drop_unwrap(unwrap, "Stopped", Result.STOPPED)

    def _import_merged_group(self, group):
        paths = [unwrap.path for unwrap in group if unwrap.path.is_file()]
        if not paths:
            return
        merged_obj = import_obj(merge_obj_files(paths))
        mark_not_unwrapped(merged_obj, "Stopped", group[0].input_name)
        manager.moved_to_invalid = True


class UVGAMI_OT_cancel(bpy.types.Operator):
    bl_idname = "uvgami.cancel"
    bl_label = "Cancel"
    bl_description = "Cancel UV unwrap"

    stem: bpy.props.StringProperty(options={"SKIP_SAVE"})
    job_id: bpy.props.IntProperty(options={"SKIP_SAVE"})

    def invoke(self, context, event):
        targets = group_targets(self.job_id) if self.job_id else []
        if targets and targets[0].join_job.finished:
            return confirm_discard(
                self,
                context,
                event,
                finished_count=len(targets[0].join_job.finished),
                title="Cancel Unwrap",
            )
        return self.execute(context)

    def execute(self, context):
        unwraps = group_targets(self.job_id) if self.job_id else piece_target(self.stem)
        if self.job_id:
            for unwrap in unwraps:
                if unwrap.join_job is not None:
                    unwrap.join_job.discard = True

        for unwrap in unwraps:
            is_individual_from_group = (
                not self.job_id
                and unwrap.join_job is not None
                and unwrap.join_job.expected > 1
            )
            label = "Cancelled (group)" if is_individual_from_group else None
            drop_unwrap(unwrap, label, Result.CANCELLED)

        if unwraps:
            manager.exit_viewer = True
            tag_redraw()
        self.report({"INFO"}, "UV unwrap cancelled")
        return {"FINISHED"}


class UVGAMI_OT_cancel_background(bpy.types.Operator):
    bl_idname = "uvgami.cancel_background"
    bl_label = "Cancel"
    bl_description = "Cancel this preseed or proxy finish"

    name: bpy.props.StringProperty()

    def execute(self, context):
        entry = next(
            (pending for pending in manager.preparing if pending.name == self.name),
            None,
        )
        if entry is not None:
            entry.cancel()
        else:
            transfer = next(
                (
                    pending
                    for pending in manager.pending_transfers
                    if pending.name == self.name
                ),
                None,
            )
            if transfer is None:
                return {"CANCELLED"}
            manager.cancel_transfer(transfer)
        tag_redraw()
        self.report({"INFO"}, "Cancelled")
        return {"FINISHED"}


class UVGAMI_OT_cancel_all(bpy.types.Operator):
    bl_idname = "uvgami.cancel_all"
    bl_label = "Cancel All"
    bl_description = "Cancel all active UV unwraps"

    def invoke(self, context, event):
        jobs = {
            unwrap.join_job for unwrap in manager.active if unwrap.join_job is not None
        }
        count = sum(len(job.finished) for job in jobs)
        if count:
            return confirm_discard(
                self, context, event, finished_count=count, title="Cancel All"
            )
        return self.execute(context)

    def execute(self, context):
        manager.cancel_session()
        self.report({"INFO"}, "UV unwrap cancelled")
        return {"FINISHED"}
