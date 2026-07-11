# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import bpy

from ..manager import manager
from ..objfile import merge_obj_files
from ..utils.io import import_obj
from ..utils.mesh import check_collection, move_to_collection
from ..utils.paths import get_preferences


def _expand_whole_group(unwraps):
    """The panel captures slice indices at draw time and the drain can add
    pieces between draw and click, so a stale slice can miss group members.
    Resolve membership from join jobs at execute time instead."""
    joins = {u.join_job for u in unwraps if u.join_job is not None}
    expanded = [u for u in manager.active if u in unwraps or u.join_job in joins]
    return expanded, joins


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
        collection = check_collection("UVgami Not Unwrapped", context.scene.collection)
        move_to_collection(invalid_obj, collection)
        invalid_obj.name = f"{invalid_obj.name}: {invalid_label}"
        invalid_obj.hide_set(True)
        manager.found_invalid_objects = True

    manager.release_jobs(unwrap.jobs)
    manager.cancel_unwrap(unwrap)


class UVGAMI_OT_stop(bpy.types.Operator):
    bl_idname = "uvgami.stop"
    bl_label = "Stop"
    bl_description = "Stop UV unwrap"

    start_idx: bpy.props.IntProperty()
    end_idx: bpy.props.IntProperty()
    whole_group: bpy.props.BoolProperty()

    def execute(self, context):
        unwraps = manager.active[self.start_idx : self.end_idx]
        if self.whole_group:
            unwraps, joins = _expand_whole_group(unwraps)
            # also stop pieces the drain hasn't exported into the session yet
            for join in joins:
                join.stop_requested = True

        stopped_pending = False
        # collect cancellations so group members can be merged into one import
        to_cancel = []
        for unwrap in unwraps:
            if unwrap.batch_process is not None:
                # pending batch members are cancelled by deleting their input
                # so the cli skips them; in-flight or done ones finish normally
                if unwrap.path.stem not in unwrap.batch_process.started:
                    to_cancel.append(unwrap)
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
                to_cancel.append(unwrap)
                stopped_pending = True

        self._cancel_collected(context, to_cancel)

        if stopped_pending:
            self.report({"INFO"}, "Stop: running meshes will finish")
        else:
            self.report({"INFO"}, "UV unwrap stop in progress")
        return {"FINISHED"}

    def _cancel_collected(self, context, to_cancel):
        # pieces of one separated mesh share a join_job; merge them into a
        # single import instead of littering the collection with N objects
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
            self._import_merged_group(context, group)
            # import already done above, so skip re-importing per member
            for unwrap in group:
                cancel_with_bookkeeping(context, unwrap, invalid_label=None)

        for unwrap in singles:
            cancel_with_bookkeeping(context, unwrap, invalid_label="Stopped")

    def _import_merged_group(self, context, group):
        if not get_preferences().invalid_collection:
            return
        # merge before any cancel: cleanup() deletes these input files
        paths = [unwrap.path for unwrap in group if unwrap.path.is_file()]
        if not paths:
            return
        merged_obj = import_obj(merge_obj_files(paths))
        collection = check_collection("UVgami Not Unwrapped", context.scene.collection)
        move_to_collection(merged_obj, collection)
        merged_obj.name = f"{group[0].input_name}: Stopped"
        merged_obj.hide_set(True)
        manager.found_invalid_objects = True


class UVGAMI_OT_cancel(bpy.types.Operator):
    bl_idname = "uvgami.cancel"
    bl_label = "Cancel"
    bl_description = "Cancel UV unwrap"

    start_idx: bpy.props.IntProperty()
    end_idx: bpy.props.IntProperty()
    whole_group: bpy.props.BoolProperty()

    def execute(self, context):
        unwraps = manager.active[self.start_idx : self.end_idx]
        if self.whole_group:
            unwraps, joins = _expand_whole_group(unwraps)
            # also cancel pieces the drain hasn't exported into the session yet
            for join in joins:
                join.cancel_requested = True
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
