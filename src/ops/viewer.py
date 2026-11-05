import bpy

from ..unwrap.manager import manager
from ..unwrap.viewer import (
    clear_snapshot,
    load_input_mesh,
    start_viewer_draw,
    stop_viewer_draw,
)
from ..utils.mesh import check_exists
from ..utils.ui import tag_redraw
from .stop import group_targets, piece_target

VIEWER_WORKSPACE = "UVgami Viewer"

CYCLE_KEYS = {"RIGHT_ARROW": 1, "LEFT_ARROW": -1}


def _trim_to_uv(window):
    screen = window.screen
    while len(screen.areas) > 1:
        smallest = min(screen.areas, key=lambda area: area.width * area.height)
        with bpy.context.temp_override(window=window, area=smallest):
            bpy.ops.screen.area_close()
    area = screen.areas[0]
    area.ui_type = "UV"
    # with show_uv on, every selected object's uv map is drawn under the snapshot
    area.spaces.active.uv_editor.show_uv = False
    region = next(region for region in area.regions if region.type == "WINDOW")
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.image.view_all(fit_view=True)


def _schedule_workspace_trim(window):
    attempts = [0]

    def tick():
        workspace = bpy.data.workspaces.get(VIEWER_WORKSPACE)
        if workspace is None or attempts[0] > 20:
            return None
        if window.workspace != workspace:
            attempts[0] += 1
            return 0.01
        _trim_to_uv(window)
        return None

    # workspace switches take effect on the next main loop tick
    bpy.app.timers.register(tick, first_interval=0.0)


def _draw_viewer_status(header, context):
    header.layout.label(icon="EVENT_LEFT_ARROW")
    header.layout.label(icon="EVENT_RIGHT_ARROW")
    header.layout.separator()
    header.layout.label(text="Switch Unwrap")
    header.layout.label(icon="MOUSE_LMB")
    header.layout.label(icon="EVENT_ESC")
    header.layout.separator()
    header.layout.label(text="Exit")


class UVGAMI_OT_view_unwrap(bpy.types.Operator):
    bl_idname = "uvgami.view_unwrap"
    bl_label = "View Unwrap"
    bl_description = "Show the unwrap live in the UV editor"

    stem: bpy.props.StringProperty(options={"SKIP_SAVE"})
    job_id: bpy.props.IntProperty(options={"SKIP_SAVE"})

    def execute(self, context):
        if self.job_id:
            candidates = [
                unwrap for unwrap in group_targets(self.job_id) if unwrap.is_viewable
            ]
        else:
            candidates = piece_target(self.stem)
        if not candidates:
            return {"CANCELLED"}
        unwrap = candidates[0]
        manager.is_viewer_active = True
        manager.exit_viewer = False
        manager.viewer_done = False

        window = context.window
        self.old_workspace = window.workspace.name
        workspace = bpy.data.workspaces.get(VIEWER_WORKSPACE)
        if workspace is None:
            existing = set(bpy.data.workspaces.keys())
            bpy.ops.workspace.duplicate()
            new_name = next(
                name for name in bpy.data.workspaces.keys() if name not in existing
            )
            workspace = bpy.data.workspaces[new_name]
            workspace.name = VIEWER_WORKSPACE
            _schedule_workspace_trim(window)
        else:
            window.workspace = workspace

        # edit mode uv maps are drawn under the snapshot
        self.old_mode = None
        active = context.view_layer.objects.active
        if active is not None and active.mode != "OBJECT":
            self.old_mode = (active, active.mode)
            bpy.ops.object.mode_set(mode="OBJECT")

        start_viewer_draw()
        self._show(unwrap)
        workspace.status_text_set(_draw_viewer_status)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _show(self, unwrap):
        if manager.current_viewer is not None:
            manager.current_viewer.viewing = False
        manager.viewer_done = False
        clear_snapshot()
        load_input_mesh(unwrap.path)
        manager.current_viewer = unwrap
        unwrap.viewing = True
        tag_redraw(("WINDOW",))

    def _cycle(self, step):
        viewable = [unwrap for unwrap in manager.active if unwrap.is_viewable]
        if not viewable:
            return
        current = manager.current_viewer
        index = viewable.index(current) + step if current in viewable else 0
        target = viewable[index % len(viewable)]
        if target is not current:
            self._show(target)

    def modal(self, context, event):
        if event.type in CYCLE_KEYS and event.value == "PRESS":
            self._cycle(CYCLE_KEYS[event.type])
            return {"RUNNING_MODAL"}

        if event.type in {"LEFTMOUSE", "RIGHTMOUSE", "ESC"} or manager.exit_viewer:
            manager.is_viewer_active = False
            if manager.current_viewer is not None:
                manager.current_viewer.viewing = False
                manager.current_viewer = None
            stop_viewer_draw()

            # both ops are deferred a tick but process in order
            workspace = bpy.data.workspaces.get(VIEWER_WORKSPACE)
            if workspace is not None:
                workspace.status_text_set(None)
            if workspace is not None and context.window.workspace == workspace:
                bpy.ops.workspace.delete()
            restore = bpy.data.workspaces.get(self.old_workspace)
            if restore is not None:
                context.window.workspace = restore

            if self.old_mode is not None:
                obj, mode = self.old_mode
                if check_exists(obj) and obj.visible_get() and obj.mode == "OBJECT":
                    context.view_layer.objects.active = obj
                    bpy.ops.object.mode_set(mode=mode)

            return {"FINISHED"}

        return {"PASS_THROUGH"}
