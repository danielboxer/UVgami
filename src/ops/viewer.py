import bpy

from ..manager import manager
from ..utils.io import import_obj
from ..utils.mesh import check_exists, deselect_all, move_to_collection

old_ui = None


def enter_viewer():
    # switch to UV editor
    global old_ui
    if bpy.context.area.ui_type != "UV":
        old_ui = bpy.context.area.ui_type
        bpy.context.area.ui_type = "UV"
    bpy.ops.image.view_all(fit_view=True)


class UVGAMI_OT_view_unwrap(bpy.types.Operator):
    bl_idname = "uvgami.view_unwrap"
    bl_label = "View Unwrap"
    bl_description = (
        "View the selected unwrap's UV map. The UV editor will be opened automatically"
    )

    index: bpy.props.IntProperty()

    def execute(self, context):
        unwrap = manager.active[self.index]
        manager.is_viewer_active = True
        manager.exit_viewer = False

        # make viewer
        if unwrap.viewer_obj is None or not check_exists(unwrap.viewer_obj):
            viewer = import_obj(unwrap.path, f"{unwrap.name}_viewer")
            unwrap.viewer_obj = viewer

            # scale viewer down
            viewer.scale = (0, 0, 0)

        viewer = unwrap.viewer_obj
        if len(viewer.users_collection) == 0:
            # move to scene collection
            context.scene.collection.objects.link(viewer)
        elif viewer.users_collection[0] != context.scene.collection:
            move_to_collection(viewer, context.scene.collection)

        enter_viewer()

        # make uvs visible by going into edit mode
        deselect_all()
        viewer.select_set(True)
        bpy.context.view_layer.objects.active = viewer
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")

        manager.current_viewer = unwrap
        unwrap.viewing = True
        self.report({"INFO"}, "Click to exit viewer")
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if (
            event.type == "LEFTMOUSE"
            or event.type == "RIGHTMOUSE"
            or manager.exit_viewer
        ):
            manager.is_viewer_active = False
            if manager.current_viewer is not None:
                manager.current_viewer.viewing = False

            if old_ui is not None:
                context.area.ui_type = old_ui

            # unlink current viewer mesh
            current = manager.current_viewer.viewer_obj
            if current is not None and check_exists(current):
                for collection in current.users_collection:
                    collection.objects.unlink(current)

            return {"FINISHED"}

        return {"RUNNING_MODAL"}
