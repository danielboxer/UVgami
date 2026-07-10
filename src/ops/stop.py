# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import bpy

from ..job import Join
from ..manager import manager
from ..utils.io import import_obj
from ..utils.mesh import check_collection, move_to_collection
from ..utils.paths import get_preferences


def cancel_with_bookkeeping(context, unwrap, invalid_label=None):
    """Cancel one unwrap, optionally moving its input to the not unwrapped
    collection first. Import must happen before cancel_unwrap since
    unwrap.cleanup() deletes the input file."""
    if (
        invalid_label is not None
        and get_preferences().invalid_collection
        and unwrap.path.is_file()
    ):
        invalid_obj = import_obj(unwrap.path)
        collection = check_collection(
            "UVgami Not Unwrapped", context.scene.collection
        )
        move_to_collection(invalid_obj, collection)
        invalid_obj.name = f"{invalid_obj.name}: {invalid_label}"
        invalid_obj.hide_set(True)
        manager.found_invalid_objects = True

    for job in unwrap.jobs:
        job.count = job.count - 1
        # if it was the last one
        if isinstance(job, Join) and job.count - len(job.unwrapped) == 0:
            # this makes it so the popup doesn't show if all cancelled
            manager.finished_count -= len(job.unwrapped)
            manager.cancelled_count += len(job.unwrapped)

    manager.cancel_unwrap(unwrap)


class UVGAMI_OT_stop(bpy.types.Operator):
    bl_idname = "uvgami.stop"
    bl_label = "Stop"
    bl_description = "Stop UV unwrap"

    start_idx: bpy.props.IntProperty()
    end_idx: bpy.props.IntProperty()

    def execute(self, context):
        stopped_pending = False
        for unwrap in manager.active[self.start_idx : self.end_idx]:
            if unwrap.batch_process is not None:
                # pending batch members are cancelled by deleting their input
                # so the cli skips them; in-flight or done ones finish normally
                if unwrap.path.stem not in unwrap.batch_process.started:
                    cancel_with_bookkeeping(context, unwrap, invalid_label="Stopped")
                    stopped_pending = True
            elif unwrap.process is not None:
                if manager.engine.supports_early_stop:
                    # send stop command
                    if not manager.engine.request_early_stop(unwrap.process):
                        self.report({"ERROR"}, "Could not stop unwrap")
                # a running solo mesh on an engine without early stop just
                # finishes normally, like an in-flight batch member
            elif manager.engine.supports_early_stop:
                # queued: this triggers a graceful early stop when it starts
                unwrap.is_stopped = True
            else:
                # queued but the engine can't finish early with a result, so
                # cancel it now instead of letting it start and be force killed
                cancel_with_bookkeeping(context, unwrap, invalid_label="Stopped")
                stopped_pending = True

        if stopped_pending:
            self.report({"INFO"}, "Stop: running meshes will finish")
        else:
            self.report({"INFO"}, "UV unwrap stop in progress")
        return {"FINISHED"}


class UVGAMI_OT_cancel(bpy.types.Operator):
    bl_idname = "uvgami.cancel"
    bl_label = "Cancel"
    bl_description = "Cancel UV unwrap"

    start_idx: bpy.props.IntProperty()
    end_idx: bpy.props.IntProperty()

    def execute(self, context):
        unwraps = manager.active[self.start_idx : self.end_idx]
        cancel_count = len(unwraps)

        for unwrap in unwraps:
            # individual cancel from a group: move to not unwrapped collection
            is_individual_from_group = (
                cancel_count == 1
                and unwrap.join_job is not None
                and unwrap.join_job.count > 1
            )
            invalid_label = "Cancelled (group)" if is_individual_from_group else None
            cancel_with_bookkeeping(context, unwrap, invalid_label=invalid_label)

        self.report({"INFO"}, "UV unwrap cancelled")
        return {"FINISHED"}


class UVGAMI_OT_cancel_all(bpy.types.Operator):
    bl_idname = "uvgami.cancel_all"
    bl_label = "Cancel All"
    bl_description = "Cancel all active UV unwraps"

    def execute(self, context):
        manager.stop_all()
        manager.finish()
        self.report({"INFO"}, "UV unwrap cancelled")
        return {"FINISHED"}
