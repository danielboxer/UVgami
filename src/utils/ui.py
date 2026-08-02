import bpy


def newline_label(label, layout):
    for line in label:
        layout.label(text=line)


def toggle(layout, props, name, text, icon, active=True):
    """Checkbox row, returning a box for the options it reveals, or None."""
    split = layout.split(factor=0.7)
    split.active = active
    split.label(icon=icon, text=text)
    split.prop(props, name)
    if not getattr(props, name):
        return None
    box = layout.box()
    box.active = active
    return box


def draw_active(layout, names):
    """The settings that will change the run. A grid so a long list wraps
    instead of being cut off by the sidebar width."""
    if not names:
        return
    grid = layout.grid_flow(row_major=True, columns=0, align=True)
    for name in names:
        grid.label(text=name, icon="DOT")


def tag_redraw():
    """Repaint the editors that show unwrap progress. Goes through bpy.data
    because the manager calls this from a timer, where the context has no
    window or area. Whole areas, so the sidebar panel updates too."""
    for wm in bpy.data.window_managers:
        for window in wm.windows:
            for area in window.screen.areas:
                if area.type in {"VIEW_3D", "IMAGE_EDITOR"}:
                    area.tag_redraw()


_status = None


def _draw_status(header, context):
    from bl_ui.space_statusbar import STATUSBAR_HT_header

    # ours first, so it sits at the far left instead of past the right aligned
    # stats, where it's easy to miss
    if _status is not None:
        header.layout.row().label(text=_status[0], icon=_status[1])
    STATUSBAR_HT_header._draw_orig(header, context)


def set_status(text, icon="CHECKMARK"):
    """Add a message to the end of the status bar, None clears it."""
    global _status
    _status = (text, icon) if text else None
    for wm in bpy.data.window_managers:
        for window in wm.windows:
            window.workspace.status_text_set(_draw_status if text else None)


def popup(msg, title, icon):
    def draw(self, context):
        newline_label(msg, self.layout)

    bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)


def switch_shading(type):
    for area in bpy.context.screen.areas:
        for space in area.spaces:
            if space.type == "VIEW_3D":
                space.shading.type = type
                break
