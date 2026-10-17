import bpy

from ..utils.paths import get_addon_id, get_addon_name
from ..utils.ui import tag_redraw
from .stop import group_targets


def reset_prop(group, prop):
    group.property_unset(prop)
    if prop in group:
        del group[prop]


def reset_group(group):
    # property_unset is a no-op on pointer props
    for prop in group.__annotations__:
        value = getattr(group, prop)
        if isinstance(value, bpy.types.PropertyGroup):
            reset_group(value)
        else:
            reset_prop(group, prop)


class UVGAMI_OT_expand(bpy.types.Operator):
    bl_idname = "uvgami.expand"
    bl_label = "Expand"
    bl_description = "Expand or collapse joined mesh details"

    job_id: bpy.props.IntProperty()

    def execute(self, context):
        targets = group_targets(self.job_id)
        if targets:
            job = targets[0].join_job
            job.is_expanded = not job.is_expanded
            # the arrow only flips on the next dispatch tick otherwise
            tag_redraw()
        return {"FINISHED"}


class UVGAMI_OT_reset_setting(bpy.types.Operator):
    bl_idname = "uvgami.reset_setting"
    bl_label = "Active Setting"
    bl_options = {"UNDO", "INTERNAL"}

    path: bpy.props.StringProperty()
    label: bpy.props.StringProperty()

    @classmethod
    def description(cls, context, properties):
        return f"{properties.label}. Click to reset to default"

    def execute(self, context):
        group = context.scene.uvgami
        path = self.path
        if "." in path:
            name, path = path.split(".", 1)
            group = getattr(group, name)
        reset_prop(group, path)
        # unset skips the notifier a normal click sends
        tag_redraw()
        return {"FINISHED"}


class UVGAMI_OT_reset_settings(bpy.types.Operator):
    bl_idname = "uvgami.reset_settings"
    bl_label = "Reset Settings"
    bl_description = "Reset all settings to their default values"
    bl_options = {"UNDO"}

    def execute(self, context):
        reset_group(context.scene.uvgami)
        return {"FINISHED"}


class UVGAMI_OT_open_preferences(bpy.types.Operator):
    bl_idname = "uvgami.open_preferences"
    bl_label = "Preferences"
    bl_description = "Open UVgami preferences"

    def execute(self, context):
        bpy.ops.screen.userpref_show()
        context.preferences.active_section = "ADDONS"
        context.window_manager.addon_search = get_addon_name()
        bpy.ops.preferences.addon_show(module=get_addon_id())
        return {"FINISHED"}
