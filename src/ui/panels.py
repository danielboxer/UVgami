import bpy

from ..engines import ENGINES, active_engine, installed_engines
from ..job import Result
from ..logger import logger
from ..manager import manager
from ..ops.auto_preview import draw_auto_preview
from ..ops.part_override_editing import is_editing, selected_part_count

# from ..ops.segment_edit import draw_segment_edit
from ..utils.paths import read_manifest
from ..utils.ui import count_label, draw_active
from .props import SETTING_ICONS_AND_LABELS, fix_settings, unwrap_settings


class EnginePanel:
    @classmethod
    def poll(cls, context):
        return bool(installed_engines())


ICON_BUTTON_SPLIT = 0.85


SUCCESS_ICON = "COLORSET_03_VEC"
FAILED_ICON = "COLORSET_01_VEC"


def draw_summary(layout):
    if not manager.summary:
        return
    box = layout.box()
    split = box.split(factor=0.9)
    split.operator(
        "uvgami.clear_summary",
        text=manager.summary[0],
        icon=FAILED_ICON if manager.summary_failed else SUCCESS_ICON,
        emboss=False,
    )
    split.operator("uvgami.clear_summary", text="", icon="X", emboss=False)
    notes = manager.summary[1:] or [f"Finished in {logger.get_latest().time:.1f}s"]
    column = box.column(align=True)
    column.active = False
    for note in notes:
        row = column.row()
        row.alignment = "CENTER"
        row.label(text=note)


def draw_queue(box):
    active_unwraps = manager.active
    if not active_unwraps and not manager.preparing and not manager.pending_transfers:
        return
    row = box.box().row()
    row.alignment = "CENTER"
    minutes, seconds = divmod(manager.elapsed_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    clock = f"{hours}:{minutes:02}:{seconds:02}" if hours else f"{minutes}:{seconds:02}"
    row.label(text=f"UV unwrap in progress {clock}")

    if manager.is_viewer_active:
        viewer_ui = box.box().row()
        viewer_ui.alignment = "CENTER"
        viewer_ui.label(text="Press ESC to exit viewer")

    jobs, singles = _jobs_and_singles(active_unwraps)
    rows = (
        len(jobs)
        + len(singles)
        + len(manager.preparing)
        + len(manager.pending_transfers)
    )
    if rows > 1:
        box.row().operator("uvgami.cancel_all", icon="CANCEL")

    for job, members in jobs.items():
        _draw_job_row(box, job, members)
    for unwrap in singles:
        _draw_single_row(box, unwrap)

    for entry in manager.preparing:
        _draw_background_row(box, entry.name, entry.name)

    for transfer in manager.pending_transfers:
        _draw_background_row(box, f"{transfer.name} (finishing)", transfer.name)

    box.separator()


def _draw_background_row(box, label, name):
    row = box.box().row()
    row.label(text=label, icon="SORTTIME")
    row.operator("uvgami.cancel_background", text="", icon="CANCEL").name = name


def _jobs_and_singles(active_unwraps):
    jobs = {}
    singles = []
    for unwrap in active_unwraps:
        if unwrap.join_job is None:
            singles.append(unwrap)
        else:
            jobs.setdefault(unwrap.join_job, []).append(unwrap)
    return jobs, singles


def _running_icon(is_running):
    return f"RADIOBUT_{'ON' if is_running else 'OFF'}"


def _draw_job_row(box, job, members):
    job_box = box.box()
    row = job_box.row()
    expand_row = row.row(align=True)
    expand_row.operator(
        "uvgami.expand",
        text="",
        icon=f"DISCLOSURE_TRI_{'DOWN' if job.is_expanded else 'RIGHT'}",
        emboss=False,
    ).job_id = job.job_id
    label = members[0].input_name
    if any(unwrap.is_stalled for unwrap in members):
        label += " (stalled)"
    expand_row.operator(
        "uvgami.expand",
        text=label,
        icon=_running_icon(any(unwrap.is_running for unwrap in members)),
        emboss=False,
    ).job_id = job.job_id
    _draw_target_buttons(
        row,
        viewable=any(_is_viewable(unwrap) for unwrap in members),
        stoppable=any(unwrap.is_stoppable for unwrap in members),
        job_id=job.job_id,
    )
    if job.is_expanded:
        _draw_group_pieces(job_box, job)


def _draw_single_row(box, unwrap):
    row = box.box().row()
    label = unwrap.name + (" (stalled)" if unwrap.is_stalled else "")
    row.label(text=label, icon=_running_icon(unwrap.is_running))
    _draw_piece_buttons(row, unwrap)


PIECE_ROW_LIMIT = 10

_RESULT_ICONS = {
    Result.FINISHED: "CHECKMARK",
    Result.INVALID: "ERROR",
    Result.CANCELLED: "X",
    Result.STOPPED: "SNAP_FACE",
}


def _draw_group_pieces(box, job):
    small = job.expected <= PIECE_ROW_LIMIT
    for item in job.members:
        if item.result is not None:
            if small:
                box.row().label(text=item.name, icon=_RESULT_ICONS[item.result])
            continue
        if not small and not item.is_running:
            continue
        row = box.row()
        row.label(
            text=item.name + (" (stalled)" if item.is_stalled else ""),
            icon=f"LAYER_{'ACTIVE' if item.is_running else 'USED'}",
        )
        _draw_piece_buttons(row, item)
    if not small:
        box.row().label(text=f"{job.reported} of {job.expected} done", icon="INFO")


def _is_viewable(item):
    return item.is_viewable and not manager.is_viewer_active


def _icon_button(row, enabled, operator, icon):
    button_row = row.row()
    button_row.enabled = enabled
    return button_row.operator(operator, text="", icon=icon)


def _draw_target_buttons(row, *, viewable, stoppable, **target):
    buttons = []
    if manager.engine.supports_viewer:
        buttons.append(_icon_button(row, viewable, "uvgami.view_unwrap", "HIDE_OFF"))
    if manager.engine.supports_early_stop:
        buttons.append(_icon_button(row, stoppable, "uvgami.stop", "SNAP_FACE"))
    buttons.append(row.operator("uvgami.cancel", text="", icon="CANCEL"))
    for button in buttons:
        for field, value in target.items():
            setattr(button, field, value)


def _draw_piece_buttons(row, item):
    _draw_target_buttons(
        row,
        viewable=_is_viewable(item),
        stoppable=item.is_stoppable,
        stem=item.path.stem,
    )


def _setting_label(layout, path):
    icon, label = SETTING_ICONS_AND_LABELS[path]
    layout.label(icon=icon, text=label)


class UVGAMI_PT_main(bpy.types.Panel):
    # blender draws bl_label after the header content
    bl_label = ""
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"

    def draw_header(self, context):
        props = context.scene.uvgami
        self.layout.label(text=read_manifest()["name"])
        if active_engine(props.engine) is None:
            return
        draw_active(self.layout, unwrap_settings(props), context)

    def draw(self, context):
        props = context.scene.uvgami
        engine = active_engine(props.engine)
        if engine is None:
            ENGINES["OPTCUTS"].draw_not_installed(self.layout)
            return

        box = self.layout.box()

        for candidate in ENGINES.values():
            if candidate.is_available():
                candidate.draw_update_notice(box)

        row = box.row()
        row.scale_y = 2
        row.enabled = not manager.is_active
        row.operator("uvgami.start", icon="UV")

        if not manager.in_uv_editor:
            draw_summary(box)
            draw_queue(box)

        row = box.row()
        row.label(icon="TOOL_SETTINGS", text="Engine")
        row.prop(props, "engine", text="")

        engine.draw_settings(box, props)
        row = box.row()
        _setting_label(row, "priority")
        row.prop(props, "priority", text="")

        if engine.supports_keep_quads:
            split = box.split(factor=0.7)
            _setting_label(split, "keep_quads")
            split.prop(props, "keep_quads")


def draw_concurrent(layout, props, engine):
    row = layout.row()
    row.active = not engine.batches_queue(props)
    row.label(icon="MOD_ARRAY", text="Concurrent")
    row.prop(props, "max_cores", slider=True)


def draw_proxy(layout, props):
    container = layout.box() if props.use_proxy else layout
    split = container.split(factor=0.7)
    _setting_label(split, "use_proxy")
    split.prop(props, "use_proxy")
    if not props.use_proxy:
        return
    row = container.row()
    row.label(text="Faces", icon="MESH_DATA")
    row.prop(props, "proxy_faces")


def draw_timeout(layout, props):
    row = layout.row()
    _setting_label(row, "unwrap_timeout")
    row.prop(props, "unwrap_timeout")


class UVGAMI_PT_speed(EnginePanel, bpy.types.Panel):
    bl_label = "Speed"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 2

    def draw(self, context):
        box = self.layout.box()
        props = context.scene.uvgami

        engine = active_engine(props.engine)
        draw_concurrent(box, props, engine)

        if engine.supports_proxy:
            draw_proxy(box, props)

        split = box.split(factor=0.7)
        split.active = not engine.uses_proxy(props)
        _setting_label(split, "stack_similar")
        split.prop(props, "stack_similar")

        draw_timeout(box, props)


def draw_part_overrides(layout, context, engine):
    # props = context.scene.uvgami
    overrides = context.scene.uvgami_part_overrides

    count = selected_part_count()
    selection = (
        f"{count_label(count, 'part')} selected" if count else "Click a part to edit it"
    )
    row = layout.row()
    row.scale_y = 1.5
    row.operator("uvgami.exit_part_overrides", icon="PANEL_CLOSE")
    layout.label(text=selection, icon="MOD_EXPLODE")

    column = layout.column()
    column.enabled = count > 0

    split = column.split(factor=0.7)
    split.label(icon="PINNED", text="Keep UVs")
    split.prop(overrides, "keeps_uvs", text="")

    settings = column.column()
    settings.active = not overrides.keeps_uvs
    if engine.supports_hard_surface:
        row = settings.row()
        row.label(icon="MOD_BEVEL", text="Mode")
        row.prop(overrides, "surface_mode", expand=True)
    # priority = settings.column(align=True)
    # priority.label(icon="SOLO_OFF", text="Priority")
    # priority.row(align=True).prop(overrides, "priority", expand=True)
    # row = settings.row()
    # # the scale is applied after Average Islands Scale
    # row.active = props.fix_scale
    # row.label(icon="FULLSCREEN_ENTER", text="UV Scale")
    # row.prop(overrides, "uv_scale", text="")


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
        engine = active_engine(context.scene.uvgami.engine)
        return engine is not None and engine.supports_guided

    def draw_header(self, context):
        self.layout.prop(context.scene.uvgami, "use_weights")

    def draw(self, context):
        props = context.scene.uvgami
        layout = self.layout
        # # proxy mode never unwraps the original
        # engine = active_engine(props.engine)
        # stretch_ignored = engine.uses_proxy(props) and props.reduce_stretching
        layout.active = props.use_weights
        # layout.active = props.use_weights and not stretch_ignored
        box = layout.box()

        row = box.row()
        row.scale_y = 1.5
        row.operator("uvgami.draw_guides", icon="GREASEPENCIL")

        row = box.row()
        row.scale_y = 1.5
        row.operator("uvgami.clear_draw", icon="TRASH")
        row.operator("uvgami.exit_draw", icon="PANEL_CLOSE")

        row = box.row()
        # weight_value only sets optcuts' -s seam avoidance flag
        row.active = props.avoid_seams
        row.label(text="Strength", icon="MOD_VERTEX_WEIGHT")
        row.prop(props, "weight_value", slider=True)

        # box.row().prop(props, "weight_mode", expand=True)

        # generate = box.box()
        # row = generate.row()
        # row.label(text="Generate", icon="SHADERFX")
        # row = generate.row()
        # row.operator("uvgami.seed_restrictions", text="From View").mode = "VIEW"
        # row.operator("uvgami.seed_restrictions", text="Crevices").mode = "CREVICES"
        # row.operator("uvgami.seed_restrictions", text="Both").mode = "BOTH"


class UVGAMI_PT_advanced(EnginePanel, bpy.types.Panel):
    bl_label = "Advanced"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 3

    def draw(self, context):
        props = context.scene.uvgami
        engine = active_engine(props.engine)
        box = self.layout.box()

        if engine.supports_import_uvs:
            split = box.split(factor=0.7)
            split.active = not engine.import_uvs_ignored(props)
            _setting_label(split, "import_uvs")
            split.prop(props, "import_uvs")

        split = box.split(factor=0.7)
        _setting_label(split, "transfer_uvs")
        split.prop(props, "transfer_uvs")

        if engine.supports_regions:
            split = box.split(factor=0.7)
            split.active = not engine.uses_import_uvs(props)
            _setting_label(split, "split_regions")
            split.prop(props, "split_regions")
        # if engine.splits_regions(props):
        #     draw_segment_edit(box, context)

        if engine.supports_symmetry:
            split = box.split(factor=0.7)
            split.active = not engine.uses_import_uvs(props)
            _setting_label(split, "symmetry")
            split.prop(props, "symmetry", text="")
            # if props.symmetry == "AXES":
            #     box.row().prop(props, "sym_axes")
            # overlap = row.row()
            # overlap.active = engine.uses_symmetry(props)
            # overlap.prop(props, "sym_overlap")
            # weld = box.row()
            # weld.active = engine.uses_symmetry(props) and engine.supports_plane_glue
            # weld.prop(props, "sym_weld")
        draw_auto_preview(box)

        if is_editing() and context.mode == "EDIT_MESH":
            draw_part_overrides(box.box(), context, engine)
        else:
            row = box.row()
            row.scale_y = 1.5
            row.operator("uvgami.edit_part_overrides", icon="EDITMODE_HLT")


class UVGAMI_PT_grid(EnginePanel, bpy.types.Panel):
    bl_label = "Grid"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_misc"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 2

    def draw(self, context):
        props = context.scene.uvgami
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        box = layout.box()

        split = box.split(factor=ICON_BUTTON_SPLIT)
        split.scale_y = 1.5
        split.operator("uvgami.add_grid", icon="UV_DATA")
        split.operator("uvgami.remove_grid", icon="TRASH", text="")

        row = box.row()
        row.prop(props, "grid_type", expand=True)
        box.prop(props, "grid_res")
        box.prop(props, "auto_grid")


def optcuts_installed():
    return ENGINES["OPTCUTS"] in installed_engines()


class UVGAMI_PT_island_uv(bpy.types.Panel):
    bl_label = ""
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "UVgami"

    def draw_header(self, context):
        self.layout.label(text=read_manifest()["name"])
        if not optcuts_installed():
            return
        draw_active(self.layout, fix_settings(context.scene.uvgami), context)

    def draw(self, context):
        props = context.scene.uvgami
        if not optcuts_installed():
            ENGINES["OPTCUTS"].draw_not_installed(self.layout, waiting_for="optcuts")
            return

        box = self.layout.box()
        ENGINES["OPTCUTS"].draw_update_notice(box)

        island = box.box()
        row = island.row()
        row.alignment = "CENTER"
        row.label(text="Island Operators", icon="GROUP_UVS")
        column = island.column()
        column.scale_y = 1.5
        column.operator("uvgami.unwrap_island", icon="UV")
        column.operator("uvgami.relax_island", icon="UV_VERTEXSEL")
        column.operator("uvgami.combine_islands", icon="UV_ISLANDSEL")

        area = box.box()
        row = area.row()
        row.alignment = "CENTER"
        row.label(text="Area Operators", icon="FACESEL")
        column = area.column()
        column.scale_y = 1.5
        column.operator("uvgami.unwrap_area", icon="UV_FACESEL")
        column.operator("uvgami.relax_area", icon="UV_VERTEXSEL")
        row = area.row()
        row.label(icon="PROP_ON", text="Expand Area")
        row.prop(props, "area_expand", text="")

        if manager.in_uv_editor:
            draw_summary(box)
            draw_queue(box)


class UVGAMI_PT_island_settings(bpy.types.Panel):
    bl_label = "Settings"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_island_uv"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return optcuts_installed()

    def draw(self, context):
        props = context.scene.uvgami
        box = self.layout.box()

        row = box.row()
        _setting_label(row, "priority")
        row.prop(props, "priority", text="")

        engine = ENGINES["OPTCUTS"]
        draw_concurrent(box, props, engine)
        draw_proxy(box, props)
        draw_timeout(box, props)


class UVGAMI_PT_pack(EnginePanel, bpy.types.Panel):
    bl_label = "Pack"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_misc"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 1

    def draw(self, context):
        props = context.scene.uvgami
        box = self.layout.box()

        row = box.row()
        row.scale_y = 1.5
        row.operator("uvgami.pack", icon="UGLYPACKAGE")

        row = box.split(factor=0.425)
        row.label(text="Margin", icon="IMGDISPLAY")
        row.prop(props, "margin")

        box.prop(props, "combine_uvs")
        box.prop(props, "fix_scale")

        box.prop(props, "pack_after_unwrap")


class UVGAMI_PT_misc(EnginePanel, bpy.types.Panel):
    bl_label = "Misc"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 4

    def draw(self, context):
        box = self.layout.box()

        row = box.row()
        row.scale_y = 1.5
        row.operator("uvgami.open_preferences", text="Preferences", icon="PREFERENCES")

        row = box.row()
        row.scale_y = 1.5
        row.operator("uvgami.open_logs", icon="TEXT")
