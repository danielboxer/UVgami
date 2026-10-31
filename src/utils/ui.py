import bpy


def only_active(entries):
    return [(icon, label, path) for icon, label, path, on in entries if on]


def count_label(count, noun):
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def setting_group_and_name(props, path):
    if "." not in path:
        return props, path
    head, name = path.split(".", 1)
    return getattr(props, head), name


def is_non_default(props, path):
    group, name = setting_group_and_name(props, path)
    return getattr(group, name) != group.bl_rna.properties[name].default


ICON_WIDTH = 20
# disclosure triangle, panel name and drag grip
HEADER_FIXED_WIDTH = 140


def draw_active(layout, settings, context):
    scale = context.preferences.system.ui_scale
    free = context.region.width - HEADER_FIXED_WIDTH * scale
    # header rows can't wrap, extra icons are cut off at the sidebar edge
    limit = max(0, int(free / (ICON_WIDTH * scale)))
    if not settings or limit < 1:
        return
    row = layout.row(align=True)
    row.alignment = "RIGHT"
    hidden = len(settings) - limit
    if hidden > 0:
        settings = settings[: limit - 1]
    for icon, label, path in settings:
        op = row.operator("uvgami.reset_setting", text="", icon=icon, emboss=False)
        op.label = label
        op.path = path
    if hidden > 0:
        row.label(text=f"+{hidden + 1}")


def tag_redraw(region_types=("WINDOW", "UI"), area_types=("VIEW_3D", "IMAGE_EDITOR")):
    # bpy.context in timer callbacks has no window or area
    for window_manager in bpy.data.window_managers:
        for window in window_manager.windows:
            for area in window.screen.areas:
                if area.type in area_types:
                    for region in area.regions:
                        if region.type in region_types:
                            region.tag_redraw()


_status = None


def _draw_status(header, context):
    from bl_ui.space_statusbar import STATUSBAR_HT_header

    # labels added after STATUSBAR_HT_header's stats are drawn past them
    if _status is not None:
        header.layout.row().label(text=_status[0], icon=_status[1])
    STATUSBAR_HT_header._draw_orig(header, context)


def set_status(text, icon="CHECKMARK"):
    global _status
    _status = (text, icon) if text else None
    # status_text_set crashes blender when no window is active
    if bpy.context.screen is None:
        return
    for window_manager in bpy.data.window_managers:
        for window in window_manager.windows:
            window.workspace.status_text_set(_draw_status if text else None)


def popup(lines, title, icon):
    # popup_menu segfaults a background blender
    if bpy.app.background:
        return

    def draw(self, context):
        for line in lines:
            self.layout.label(text=line)

    bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)


def switch_solid_color(color_type):
    for area in bpy.context.screen.areas:
        for space in area.spaces:
            if space.type == "VIEW_3D":
                # in material preview objects are hidden until their shaders compile
                space.shading.type = "SOLID"
                space.shading.color_type = color_type
                break
