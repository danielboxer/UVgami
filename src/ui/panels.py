import bpy

from ..engines import get_engine
from ..logger import logger
from ..manager import manager
from ..utils.ui import draw_active, newline_label, only_active, toggle


def unwrap_settings(props):
    """The settings a full unwrap will apply, as (icon, label, path) entries
    for draw_active. Only ones that change the result, so a sub-setting of
    something already listed is left out."""
    engine = get_engine(props.engine)
    return engine.active_settings(props) + only_active(
        (
            (
                "IMPORT",
                "Import UVs",
                "import_uvs",
                engine.supports_import_uvs and props.import_uvs,
            ),
            (
                "MOD_VERTEX_WEIGHT",
                "Weights",
                "use_weights",
                engine.supports_guided and props.use_weights,
            ),
            ("MOD_DECIM", "Proxy", "use_proxy", props.use_proxy),
            ("MOD_MIRROR", "Symmetry", "use_symmetry", props.use_symmetry),
            (
                "CON_ROTLIKE",
                "Concurrent",
                "concurrent",
                props.concurrent and not engine.batches_queue(props),
            ),
            ("TIME", "Timeout", "unwrap_timeout", props.unwrap_timeout > 0),
            (
                "MOD_TRIANGULATE",
                "Preserve Mesh",
                "untriangulate",
                engine.supports_preserve and props.untriangulate,
            ),
            ("UV_DATA", "Transfer UVs", "transfer_uvs", props.transfer_uvs),
            (
                "UGLYPACKAGE",
                "Pack After Unwrap",
                "pack_after_unwrap",
                props.pack_after_unwrap and not engine.requires_pack,
            ),
        )
    )


def fix_settings(props):
    """Same idea for the uv editor operators, which always run optcuts, so
    this is a different set from the main panel's."""
    return only_active(
        (
            (
                "SOLO_OFF",
                f"Quality {props.optcuts.quality.title()}",
                "optcuts.quality",
                props.optcuts.quality != "MEDIUM",
            ),
            ("CON_ROTLIKE", "Concurrent", "concurrent", props.concurrent),
            ("TIME", "Timeout", "unwrap_timeout", props.unwrap_timeout > 0),
        )
    )


def draw_result(layout):
    """Banner with the last session's summary, until dismissed or the next run."""
    if not manager.result:
        return
    box = layout.box()
    # a split, not a row: a label sizes to its text and leaves the x stranded
    # at the far end of an empty row, a full width one centers the message
    split = box.split(factor=0.9)
    split.alert = manager.result_failed
    split.operator(
        "uvgami.clear_result",
        text=manager.result[0],
        icon="ERROR" if manager.result_failed else "CHECKMARK",
        emboss=False,
    )
    split.operator("uvgami.clear_result", text="", icon="X", emboss=False)
    if len(manager.result) > 1:
        newline_label(manager.result[1:], box.column())


def draw_queue(box):
    """The running and queued unwraps, with their stop and cancel buttons."""
    active_unwraps = manager.active
    if not active_unwraps:
        return
    row = box.box().row()
    row.alignment = "CENTER"
    row.label(text="UV unwrap in progress")

    if manager.is_viewer_active:
        viewer_ui = box.box().row()
        viewer_ui.alignment = "CENTER"
        viewer_ui.label(text="Press ESC to exit viewer")

    groups, active_groups = _build_unwrap_groups(active_unwraps)
    _draw_unwrap_groups(box, groups, active_groups)
    box.separator()


def _build_unwrap_groups(active_unwraps):
    """Unwraps keyed by their join job, or by index when they have none. The
    key type is what tells the two apart when drawing."""
    groups = {}
    active_groups = []
    for unwrap_idx, unwrap in enumerate(active_unwraps):
        job = unwrap.join_job
        if job is None:
            groups[unwrap_idx] = [unwrap]
            continue
        groups.setdefault(job, []).append(unwrap)
        if unwrap.is_active and job not in active_groups:
            active_groups.append(job)
    return groups, active_groups


def _draw_unwrap_groups(box, groups, active_groups):
    """Draw all unwrap groups with their buttons."""
    cancel_index = 0
    for group_id, group in groups.items():
        display_box = box.box()
        row = display_box.row()
        # if the key isn't an int, it's a join job, so the group can be expanded
        expand_layout = not isinstance(group_id, int)

        if expand_layout:
            row.operator(
                "uvgami.expand",
                text="",
                icon=f"DISCLOSURE_TRI_{'DOWN' if group_id.is_expanded else 'RIGHT'}",
                emboss=False,
            ).index = manager.active.index(group[0])
            label_text = group[0].input_name
            is_active = group_id in active_groups
        else:
            label_text = group[0].name
            is_active = group[0].is_active
        row.label(
            text=label_text,
            icon=f"RADIOBUT_{'ON' if is_active else 'OFF'}",
        )

        # group stop and cancel button
        if expand_layout:
            # pieces the exporter hasn't written yet have no input file, so
            # stop couldn't put them in the not unwrapped collection
            is_exporting = group_id.count - len(group_id.unwrapped) - len(group) > 0
            if is_active and not is_exporting:
                stop_op = row.operator("uvgami.stop", text="", icon="SNAP_FACE")
                stop_op.start_idx = cancel_index
                stop_op.end_idx = cancel_index + len(group)
                stop_op.whole_group = True
            cancel_op = row.operator("uvgami.cancel", text="", icon="CANCEL")
            cancel_op.start_idx = cancel_index
            cancel_op.end_idx = cancel_index + len(group)
            cancel_op.whole_group = True

        if not expand_layout or group_id.is_expanded:
            for item in group:
                if expand_layout:
                    row = display_box.row()
                    row.label(
                        text=item.name,
                        icon=f"LAYER_{'ACTIVE' if item.is_active else 'USED'}",
                    )

                # viewer button, only once the unwrap has started producing
                if item.progress != (0, 0, 1) and manager.engine.supports_viewer:
                    view_op = row.operator(
                        "uvgami.view_unwrap", text="", icon="HIDE_OFF"
                    )
                    view_op.index = manager.active.index(item)
                # stop button, only a running mesh on an engine that can
                # finish early with a result
                if manager.engine.supports_early_stop and item.is_active:
                    stop_op = row.operator("uvgami.stop", text="", icon="SNAP_FACE")
                    stop_op.start_idx = cancel_index
                    stop_op.end_idx = cancel_index + 1
                cancel_op = row.operator("uvgami.cancel", text="", icon="CANCEL")
                cancel_op.start_idx = cancel_index
                cancel_op.end_idx = cancel_index + 1

                cancel_index += 1
        else:
            # collapsed, so no per-item rows drew, skip the whole group's indices
            cancel_index += len(group)

    if len(groups) > 1:
        row = box.row()
        row.operator("uvgami.cancel_all", icon="TRASH")


class UVGAMI_PT_main(bpy.types.Panel):
    bl_label = "UVgami"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"

    def draw(self, context):
        box = self.layout.box()
        props = context.scene.uvgami

        row = box.row()
        row.scale_y = 2
        row.operator("uvgami.start", icon="UV")

        if not manager.in_uv_editor:
            draw_result(box)

        draw_active(box, unwrap_settings(props))

        if not manager.in_uv_editor:
            draw_queue(box)

        row = box.row()
        row.label(icon="TOOL_SETTINGS", text="Engine")
        row.prop(props, "engine", text="")

        engine = get_engine(props.engine)
        engine.draw_settings(box, props)

        if engine.supports_import_uvs:
            split = box.split(factor=0.7)
            split.label(icon="IMPORT", text="Import UVs")
            split.prop(props, "import_uvs")

        if engine.supports_preserve:
            sub = toggle(
                box, props, "untriangulate", "Preserve Mesh", "MOD_TRIANGULATE"
            )
            if sub is not None:
                sub.row().prop(props, "maintain_mode", expand=True)

        split = box.split(factor=0.7)
        split.label(icon="UV_DATA", text="Transfer UVs")
        split.prop(props, "transfer_uvs")


def draw_concurrent(layout, props, engine):
    # hidden instead of grayed out: ai mode batches all meshes into one
    # process, so concurrency doesn't apply
    if engine.batches_queue(props):
        return
    sub = toggle(layout, props, "concurrent", "Concurrent", "CON_ROTLIKE")
    if sub is not None:
        split = sub.split()
        split.label(icon="SYSTEM", text="Cores")
        split.prop(props, "max_cores", slider=True)


def draw_timeout(layout, props):
    """Shared with the uv editor settings."""
    row = layout.row()
    row.label(text="Timeout", icon="TIME")
    row.prop(props, "unwrap_timeout")


class UVGAMI_PT_speed(bpy.types.Panel):
    bl_label = "Speed"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 3

    def draw(self, context):
        box = self.layout.box()
        props = context.scene.uvgami

        row = box.row()
        row.alignment = "CENTER"
        row.label(text="Speed", icon="SORTTIME")

        engine = get_engine(props.engine)
        draw_concurrent(box, props, engine)

        sub = toggle(box, props, "use_proxy", "Proxy", "MOD_DECIM")
        if sub is not None:
            row = sub.row()
            row.label(text="Proxy Faces", icon="MESH_DATA")
            row.prop(props, "proxy_faces")

        draw_timeout(box, props)


class UVGAMI_PT_weights(bpy.types.Panel):
    bl_label = "Weights"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 1

    @classmethod
    def poll(cls, context):
        return get_engine(context.scene.uvgami.engine).supports_guided

    def draw_header(self, context):
        self.layout.prop(context.scene.uvgami, "use_weights")

    def draw(self, context):
        props = context.scene.uvgami
        layout = self.layout
        # active, not enabled: painting turns the checkbox on itself, so the
        # buttons have to stay clickable while the panel reads as off
        layout.active = props.use_weights
        box = layout.box()

        row = box.row()
        row.alignment = "CENTER"
        row.label(text="Weights", icon="MOD_VERTEX_WEIGHT")

        row = box.row()
        row.scale_y = 1.5
        row.operator("uvgami.draw_guides", icon="GREASEPENCIL")

        generate = box.box()
        row = generate.row()
        row.label(text="Generate", icon="SHADERFX")
        row = generate.row()
        row.operator("uvgami.seed_restrictions", text="From View").mode = "VIEW"
        row.operator("uvgami.seed_restrictions", text="Crevices").mode = "CREVICES"
        row.operator("uvgami.seed_restrictions", text="Both").mode = "BOTH"

        row = box.row()
        row.scale_y = 1.5
        row.operator("uvgami.clear_draw", icon="FILE_REFRESH")
        row.operator("uvgami.exit_draw", icon="PANEL_CLOSE")

        row = box.row()
        # the engine only reads the strength for seam avoidance (-s), stretch
        # mode runs at a fixed face weight
        row.active = props.avoid_seams
        row.label(text="Strength", icon="MOD_VERTEX_WEIGHT")
        row.prop(props, "weight_value", slider=True)

        box.row().prop(props, "weight_mode", expand=True)


class UVGAMI_PT_symmetry(bpy.types.Panel):
    bl_label = "Symmetry"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 2

    def draw_header(self, context):
        self.layout.prop(context.scene.uvgami, "use_symmetry")

    def draw(self, context):
        props = context.scene.uvgami
        layout = self.layout
        layout.active = props.use_symmetry
        box = layout.box()

        row = box.row()
        row.alignment = "CENTER"
        row.label(text="Symmetry", icon="MOD_MIRROR")

        row = box.row()
        row.scale_y = 1.5
        row.prop(props, "sym_axes")

        row = box.row()
        row.prop(props, "sym_preview", icon="EMPTY_AXIS")
        row.prop(props, "sym_merge")


class UVGAMI_PT_grid(bpy.types.Panel):
    bl_label = "Grid"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 5

    def draw(self, context):
        props = context.scene.uvgami
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        box = layout.box()

        row = box.row()
        row.alignment = "CENTER"
        row.label(text="Grid", icon="TEXTURE")

        split = box.split(factor=0.8)
        split.scale_y = 1.5
        split.operator("uvgami.add_grid", icon="UV_DATA")
        split.operator("uvgami.remove_grid", icon="TRASH", text="")

        row = box.row()
        row.prop(props, "grid_type", expand=True)
        box.prop(props, "grid_res")
        box.prop(props, "auto_grid")


class UVGAMI_PT_island_uv(bpy.types.Panel):
    bl_label = "UVgami"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "UVgami"

    def draw(self, context):
        props = context.scene.uvgami
        box = self.layout.box()

        draw_active(box, fix_settings(props))

        island = box.box()
        row = island.row()
        row.alignment = "CENTER"
        row.label(text="Island Operators", icon="GROUP_UVS")
        col = island.column()
        col.scale_y = 1.5
        col.operator("uvgami.unwrap_island", icon="UV")
        col.operator("uvgami.combine_islands", icon="UV_ISLANDSEL")

        # expand only feeds the area operators, so it's grouped with them
        area = box.box()
        row = area.row()
        row.alignment = "CENTER"
        row.label(text="Area Operators", icon="FACESEL")
        col = area.column()
        col.scale_y = 1.5
        col.operator("uvgami.recut_area", icon="UV_FACESEL")
        col.operator("uvgami.relax_area", icon="UV_VERTEXSEL")
        row = area.row()
        row.label(icon="PROP_ON", text="Expand Area")
        row.prop(props, "area_expand", text="")

        if manager.in_uv_editor:
            draw_result(box)
            draw_queue(box)


class UVGAMI_PT_island_settings(bpy.types.Panel):
    bl_label = "Settings"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_island_uv"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        props = context.scene.uvgami
        box = self.layout.box()

        row = box.row()
        row.label(icon="SOLO_OFF", text="Quality")
        row.prop(props.optcuts, "quality", text="")

        # these operators always run optcuts, whatever the main panel is set to
        engine = get_engine("OPTCUTS")
        draw_concurrent(box, props, engine)
        draw_timeout(box, props)


class UVGAMI_PT_pack(bpy.types.Panel):
    bl_label = "Pack"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 4

    def draw(self, context):
        props = context.scene.uvgami
        box = self.layout.box()

        row = box.row()
        row.alignment = "CENTER"
        row.label(text="Pack", icon="PACKAGE")

        row = box.row()
        row.scale_y = 1.5
        row.operator("uvgami.pack", icon="UGLYPACKAGE")

        row = box.split(factor=0.425)
        row.label(text="Margin", icon="IMGDISPLAY")
        row.prop(props, "margin", slider=True)

        box.prop(props, "combine_uvs")
        box.prop(props, "fix_scale")

        if get_engine(props.engine).requires_pack:
            # drawn as a checked label, not the prop, so a stored False can't
            # show an unchecked box next to packing that always runs
            row = box.row()
            row.enabled = False
            row.label(text="Pack After Unwrap (required)", icon="CHECKBOX_HLT")
        else:
            box.prop(props, "pack_after_unwrap")


class UVGAMI_PT_misc(bpy.types.Panel):
    bl_label = "Misc"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 6

    def draw(self, context):
        box = self.layout.box()

        row = box.row()
        row.scale_y = 1.5
        row.operator("uvgami.open_preferences", text="Preferences", icon="PREFERENCES")

        box.separator()

        row = box.row()
        row.alignment = "CENTER"
        row.label(text="Info", icon="INFO")

        if logger.unwrap_info:
            row = box.row()
            row.operator("uvgami.copy_logs", icon="COPYDOWN")
            row.operator("uvgami.clear_logs", icon="TRASH")
            col = box.column()
            newline_label(logger.get_all(), col)
        else:
            row = box.row()
            row.alignment = "CENTER"
            row.label(text="No previous unwraps")
