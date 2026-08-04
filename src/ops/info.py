import bpy

from ..logger import logger
from ..manager import manager


class UVGAMI_OT_clear_summary(bpy.types.Operator):
    bl_idname = "uvgami.clear_summary"
    bl_label = "Dismiss"
    bl_description = "Hide the result of the last unwrap"

    def execute(self, context):
        manager.clear_summary()
        return {"FINISHED"}


class UVGAMI_OT_clear_logs(bpy.types.Operator):
    bl_idname = "uvgami.clear_logs"
    bl_label = "Clear"
    bl_description = "Delete all info"

    def execute(self, context):
        logger.unwrap_info.clear()
        self.report({"INFO"}, "Cleared info")
        return {"FINISHED"}


class UVGAMI_OT_copy_logs(bpy.types.Operator):
    bl_idname = "uvgami.copy_logs"
    bl_label = "Copy"
    bl_description = "Copy logs to clipboard"

    def execute(self, context):
        context.window_manager.clipboard = "\n".join(logger.get_all())
        self.report({"INFO"}, "Copied to clipboard")
        return {"FINISHED"}
