import bpy

from ..job import Result
from ..manager import manager
from ..objfile import merge_obj_files
from ..utils.io import import_obj
from ..utils.mesh import check_collection, move_to_collection
from ..utils.paths import get_preferences
from ..utils.ui import tag_redraw


def resolve_targets(stem, whole_group):
    """The unwraps a panel button targets, resolved by input file stem at
    execute time so a stale click on an already settled piece is a no-op."""
    target = next((u for u in manager.active if u.path.stem == stem), None)
    if target is None:
        return []
    if whole_group and target.join_job is not None:
        return [u for u in manager.active if u.join_job is target.join_job]
    return [target]


def drop_unwrap(context, unwrap, invalid_label, result):
    if (
        invalid_label is not None
        and get_preferences().invalid_collection
        and unwrap.path.is_file()
    ):
        # the import must happen before record_result, which deletes the input file
        invalid_obj = import_obj(unwrap.path)
        collection = check_collection("UVgami Not Unwrapped", context.scene.collection)
        move_to_collection(invalid_obj, collection)
        invalid_obj.name = f"{invalid_obj.name}: {invalid_label}"
        invalid_obj.hide_set(True)
    manager.record_result(unwrap, result)


class UVGAMI_OT_stop(bpy.types.Operator):
    bl_idname = "uvgami.stop"
    bl_label = "Stop"
    bl_description = "Stop UV unwrap"

    stem: bpy.props.StringProperty()
    whole_group: bpy.props.BoolProperty()

    def execute(self, context):
        unwraps = resolve_targets(self.stem, self.whole_group)

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
                    # the flag re-sends the request each tick and arms the
                    # manager's STOP_SECONDS force kill
                    unwrap.is_stopped = True
                    if not manager.engine.request_early_stop(unwrap.process):
                        self.report({"ERROR"}, "Could not stop unwrap")
                # a running solo mesh on an engine without early stop just
                # finishes normally, like an in-flight batch member
            else:
                # queued: starting a mesh just to stop it gives a map with no
                # work in it, so drop it and let it show as not unwrapped
                to_cancel.append(unwrap)
                stopped_pending = True

        self._cancel_collected(context, to_cancel)
        if to_cancel:
            manager.exit_viewer = True
            tag_redraw()

        if stopped_pending:
            self.report({"INFO"}, "Stop: queued meshes dropped")
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
                drop_unwrap(context, unwrap, None, Result.INVALID)

        for unwrap in singles:
            drop_unwrap(context, unwrap, "Stopped", Result.INVALID)

    def _import_merged_group(self, context, group):
        if not get_preferences().invalid_collection:
            return
        # merge before any settle: record_result deletes these input files
        paths = [unwrap.path for unwrap in group if unwrap.path.is_file()]
        if not paths:
            return
        merged_obj = import_obj(merge_obj_files(paths))
        collection = check_collection("UVgami Not Unwrapped", context.scene.collection)
        move_to_collection(merged_obj, collection)
        merged_obj.name = f"{group[0].input_name}: Stopped"
        merged_obj.hide_set(True)


class UVGAMI_OT_cancel(bpy.types.Operator):
    bl_idname = "uvgami.cancel"
    bl_label = "Cancel"
    bl_description = "Cancel UV unwrap"

    stem: bpy.props.StringProperty()
    whole_group: bpy.props.BoolProperty()

    def execute(self, context):
        unwraps = resolve_targets(self.stem, self.whole_group)
        if self.whole_group:
            # the user dropped the whole mesh, so the already finished pieces
            # get discarded instead of joined when the group settles
            for unwrap in unwraps:
                if unwrap.join_job is not None:
                    unwrap.join_job.discard = True

        for unwrap in unwraps:
            # an individual cancel from a group goes to the collection, so the
            # joined result visibly misses a piece
            is_individual_from_group = (
                not self.whole_group
                and unwrap.join_job is not None
                and unwrap.join_job.expected > 1
            )
            if is_individual_from_group:
                drop_unwrap(context, unwrap, "Cancelled (group)", Result.INVALID)
            else:
                drop_unwrap(context, unwrap, None, Result.CANCELLED)

        if unwraps:
            manager.exit_viewer = True
            tag_redraw()
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
