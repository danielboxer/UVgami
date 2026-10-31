import bpy

from ..logger import HEADER_PREFIX, logger
from ..manager import manager


class UVGAMI_OT_clear_summary(bpy.types.Operator):
    bl_idname = "uvgami.clear_summary"
    bl_label = "Dismiss"
    bl_description = "Hide the result of the last unwrap"

    def execute(self, context):
        manager.clear_summary()
        return {"FINISHED"}


LOG_TEXT_NAME = "UVgami Log"


class UVGAMI_OT_open_logs(bpy.types.Operator):
    bl_idname = "uvgami.open_logs"
    bl_label = "Log"
    bl_description = "Show the info in a text editor, where it can be selected"

    def execute(self, context):
        body = logger.log_with_unfinished_runs() or "No previous unwraps"
        text = bpy.data.texts.get(LOG_TEXT_NAME) or bpy.data.texts.new(LOG_TEXT_NAME)
        text.clear()
        text.write(body + "\n")
        start = max(
            (
                i
                for i, line in enumerate(body.splitlines())
                if line.startswith(HEADER_PREFIX)
            ),
            default=0,
        )
        # Blender's text editor scrolls to the cursor
        text.cursor_set(start)

        area = next(
            (
                area
                for window in context.window_manager.windows
                for area in window.screen.areas
                if area.type == "TEXT_EDITOR"
            ),
            None,
        )
        if area is None:
            if not bpy.ops.wm.window_new.poll():
                self.report({"INFO"}, f"Info written to the text '{LOG_TEXT_NAME}'")
                return {"FINISHED"}
            bpy.ops.wm.window_new()
            # window_new copies the current layout
            areas = context.window_manager.windows[-1].screen.areas
            area = max(areas, key=lambda a: a.width * a.height)
            area.type = "TEXT_EDITOR"
        space = area.spaces.active
        space.text = text
        space.top = start
        return {"FINISHED"}
