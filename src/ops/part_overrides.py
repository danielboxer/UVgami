from collections import namedtuple

import bmesh
import bpy
import gpu
import numpy
from gpu_extras.batch import batch_for_shader

from ..engines.optcuts import SURFACE_MODES

# from ..ui.props import PRIORITY_LABELS
from ..no_bpy.mesh import face_starts, lowest_index_part_labels, next_corners_in_face
from ..part_overrides import (
    KEEP_ATTRIBUTE,
    OVERRIDE_ATTRIBUTES,
    # PRIORITY_ATTRIBUTE,
    # PRIORITY_ORDER,
    SURFACE_MODE_ATTRIBUTE,
    SURFACE_MODE_CODES,
    # UV_SCALE_ATTRIBUTE,
)
from ..utils.mesh import (
    lifted_triangles,
    loop_vertices_and_face_sizes,
    validate_obj,
    vertex_position_array,
)
from ..utils.ui import tag_redraw

OVERRIDE_COLOR = (0.45, 0.6, 0.95, 0.4)


def _layer(bm, name, create=False):
    # if OVERRIDE_ATTRIBUTES[name] == "FLOAT":
    #     layers = bm.faces.layers.float
    # else:
    layers = bm.faces.layers.int
    layer = layers.get(name)
    if layer is None and create:
        layer = layers.new(name)
    return layer


def _face_selection(bm):
    return numpy.fromiter((face.select for face in bm.faces), bool, len(bm.faces))


def _topology(bm):
    return (len(bm.verts), len(bm.edges), len(bm.faces))


_expects_update = False
_geometry_updated = False
_undone = False


def _update_edit_mesh(mesh):
    global _expects_update
    bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
    _expects_update = True


def _written_back(obj):
    global _expects_update
    # foreach_get reads stale values until the edit mesh is written back
    obj.update_from_editmode()
    _expects_update = True
    return obj.data


class MeshParts:
    def __init__(self, obj):
        self.topology = _topology(bmesh.from_edit_mesh(obj.data))
        mesh = _written_back(obj)
        corners, totals = loop_vertices_and_face_sizes(mesh)
        edges = numpy.stack((corners, corners[next_corners_in_face(totals)]), axis=1)
        labels = lowest_index_part_labels(len(mesh.vertices), edges)
        vertex_part = numpy.unique(labels, return_inverse=True)[1]
        self.ids = vertex_part[corners[face_starts(totals)]]
        self.count = int(self.ids.max()) + 1 if len(self.ids) else 0
        self.selection_state = None
        self.selected_count = 0
        self._read_geometry(mesh)

    def matches(self, obj):
        return self.topology == _topology(bmesh.from_edit_mesh(obj.data))

    def refresh(self, obj):
        self._read_geometry(_written_back(obj))

    def _read_geometry(self, mesh):
        # attribute reads through the datablock are empty in edit mode
        bm = bmesh.from_edit_mesh(mesh)
        overridden = numpy.zeros(len(bm.faces), dtype=bool)
        for name in OVERRIDE_ATTRIBUTES:
            layer = _layer(bm, name)
            if layer is not None:
                values = (face[layer] != 0 for face in bm.faces)
                overridden |= numpy.fromiter(values, bool, len(bm.faces))
        positions = vertex_position_array(mesh)
        extent = numpy.ptp(positions, axis=0) if len(positions) else numpy.zeros(3)
        # depth testing hides an overlay that sits flat on the surface
        lift = 1e-4 * float(numpy.linalg.norm(extent))
        corners, triangle_faces = lifted_triangles(mesh, positions, lift)
        visible = overridden[triangle_faces]
        self.overlay_positions = corners[visible].reshape(-1, 3).astype(numpy.float32)
        self.overlay_batch = None


_parts = {}


def mesh_parts(obj):
    name = obj.data.name
    parts = _parts.get(name)
    if parts is None or not parts.matches(obj):
        parts = _parts[name] = MeshParts(obj)
        tag_redraw(("WINDOW",))
    return parts


def _edited_objects():
    return [
        obj for obj in bpy.context.objects_in_mode_unique_data if obj.type == "MESH"
    ]


def selected_part_count():
    count = 0
    for obj in _edited_objects():
        mesh = obj.data
        parts = mesh_parts(obj)
        state = _selection_state(mesh)
        # reading the selection of a big mesh slows every panel redraw
        if parts.selection_state != state:
            selected = _face_selection(bmesh.from_edit_mesh(mesh))
            parts.selected_count = len(numpy.unique(parts.ids[selected]))
            parts.selection_state = state
        count += parts.selected_count
    return count


def _active_value(name):
    obj = bpy.context.edit_object
    if obj is None or obj.type != "MESH":
        return 0
    bm = bmesh.from_edit_mesh(obj.data)
    face = bm.faces.active
    layer = _layer(bm, name)
    if face is None or layer is None:
        return 0
    return face[layer]


def _write_selected_parts(name, value):
    for obj in _edited_objects():
        mesh = obj.data
        bm = bmesh.from_edit_mesh(mesh)
        parts = mesh_parts(obj)
        selected = _face_selection(bm)
        if not selected.any():
            continue
        layer = _layer(bm, name, create=value != 0)
        if layer is None:
            continue
        bm.faces.ensure_lookup_table()
        targets = numpy.isin(parts.ids, parts.ids[selected])
        for index in numpy.flatnonzero(targets).tolist():
            bm.faces[index][layer] = value
        _update_edit_mesh(mesh)
        parts.refresh(obj)


def _int_getter(name):
    return lambda self: int(_active_value(name))


def _setter(name):
    return lambda self, value: _write_selected_parts(name, value)


SURFACE_MODE_ITEMS = [
    ("GLOBAL", "Global", "Use the Mode setting above", 0),
    *(
        (mode, label, description, SURFACE_MODE_CODES[mode])
        for mode, label, description in SURFACE_MODES
        if mode != "AUTO"
    ),
]
# PRIORITY_ITEMS = [
#     ("GLOBAL", "Global", "Use the Priority setting of the whole object", 0),
#     *(
#         (priority, PRIORITY_LABELS[priority], "", code)
#         for code, priority in enumerate(PRIORITY_ORDER, 1)
#     ),
# ]


class UVGAMI_PG_part_overrides(bpy.types.PropertyGroup):
    surface_mode: bpy.props.EnumProperty(
        name="Mode",
        description="Whether seams on these parts follow sharp features",
        items=SURFACE_MODE_ITEMS,
        get=_int_getter(SURFACE_MODE_ATTRIBUTE),
        set=_setter(SURFACE_MODE_ATTRIBUTE),
    )
    keeps_uvs: bpy.props.BoolProperty(
        name="Keep UVs",
        description="Leave these parts out of the unwrap and keep their UV map",
        get=lambda self: bool(_active_value(KEEP_ATTRIBUTE)),
        set=_setter(KEEP_ATTRIBUTE),
    )
    # priority: bpy.props.EnumProperty(
    #     name="Priority",
    #     description="Whether the unwrap of these parts favors less stretching or fewer seams",
    #     items=PRIORITY_ITEMS,
    #     get=_int_getter(PRIORITY_ATTRIBUTE),
    #     set=_setter(PRIORITY_ATTRIBUTE),
    # )
    # uv_scale: bpy.props.FloatProperty(
    #     name="UV Scale",
    #     description=(
    #         "Size of these parts' islands relative to the rest when packing."
    #         " Needs Average Islands Scale"
    #     ),
    #     min=0.01,
    #     soft_max=10,
    #     get=lambda self: _active_value(UV_SCALE_ATTRIBUTE) or 1.0,
    #     set=_setter(UV_SCALE_ATTRIBUTE),
    # )


def _select_whole_parts(mesh, parts, previous):
    bm = bmesh.from_edit_mesh(mesh)
    bm.faces.index_update()
    selected = _face_selection(bm)
    if parts.count == 0:
        return selected
    if previous is None or len(previous) != len(selected):
        previous = numpy.zeros(len(selected), dtype=bool)
    added = selected & ~previous
    active = bm.faces.active
    if active is not None and active.select:
        added[active.index] = True

    size = numpy.bincount(parts.ids, minlength=parts.count)
    chosen = numpy.bincount(parts.ids, weights=selected, minlength=parts.count)
    grows = numpy.bincount(parts.ids, weights=added, minlength=parts.count) > 0
    partial = (chosen > 0) & (chosen < size)
    target = selected.copy()
    target[(partial & grows)[parts.ids]] = True
    target[(partial & ~grows)[parts.ids]] = False

    changed = numpy.flatnonzero(target != selected).tolist()
    if changed:
        bm.faces.ensure_lookup_table()
        for index in changed:
            bm.faces[index].select_set(bool(target[index]))
        bm.select_flush_mode()
        _update_edit_mesh(mesh)
    return target


_old_mode = None
_is_editing = False


def is_editing():
    return _is_editing


def _selection_state(mesh):
    active = bmesh.from_edit_mesh(mesh).faces.active
    return (mesh.total_face_sel, -1 if active is None else active.index)


SeenSelection = namedtuple("SeenSelection", ["state", "selection"])


class UVGAMI_OT_edit_part_overrides(bpy.types.Operator):
    bl_idname = "uvgami.edit_part_overrides"
    bl_label = "Edit Parts"
    bl_description = (
        "Select loose parts in Edit Mode and give them their own unwrap settings."
        " Clicking a face selects its whole part"
    )

    def invoke(self, context, event):
        global _old_mode, _is_editing
        obj = context.active_object
        if obj is None:
            self.report({"ERROR"}, "No object selected")
            return {"CANCELLED"}
        if not validate_obj(self, obj, report=True):
            return {"CANCELLED"}
        if context.mode != "EDIT_MESH":
            _old_mode = obj.mode
            bpy.ops.object.mode_set(mode="EDIT")
        context.tool_settings.mesh_select_mode = (False, False, True)
        if _is_editing:
            return {"FINISHED"}

        _is_editing = True
        start_overlay()
        self.seen = {}
        self._sync(context)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if not _is_editing or context.mode != "EDIT_MESH":
            stop_part_overrides()
            return {"CANCELLED"}
        self._sync(context)
        return {"PASS_THROUGH"}

    def _sync(self, context):
        global _geometry_updated, _undone
        selection_changed = False
        objects = _edited_objects()
        for obj in objects:
            mesh = obj.data
            parts = mesh_parts(obj)
            state = _selection_state(mesh)
            seen = self.seen.get(mesh.name)
            if seen is not None and seen.state == state:
                continue
            selection_changed = True
            previous = None if seen is None else seen.selection
            selection = _select_whole_parts(mesh, parts, previous)
            self.seen[mesh.name] = SeenSelection(_selection_state(mesh), selection)
        if not _geometry_updated and not _undone:
            return
        stale = _undone or not selection_changed
        _geometry_updated = _undone = False
        if not stale:
            return
        for obj in objects:
            mesh_parts(obj).refresh(obj)
        tag_redraw(("WINDOW",))


class UVGAMI_OT_exit_part_overrides(bpy.types.Operator):
    bl_idname = "uvgami.exit_part_overrides"
    bl_label = "Exit Edit Parts"
    bl_description = "Go back to the previous mode"

    def execute(self, context):
        mode = _old_mode if _old_mode not in (None, "EDIT") else "OBJECT"
        bpy.ops.object.mode_set(mode=mode)
        return {"FINISHED"}


_overlay_handler = None
_overlay_shader = None


def _draw_overlay():
    global _overlay_shader
    context = bpy.context
    if context.mode != "EDIT_MESH":
        return
    if _overlay_shader is None:
        _overlay_shader = gpu.shader.from_builtin("UNIFORM_COLOR")

    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("LESS_EQUAL")
    _overlay_shader.bind()
    _overlay_shader.uniform_float("color", OVERRIDE_COLOR)
    for obj in context.objects_in_mode_unique_data:
        if obj.type != "MESH":
            continue
        parts = _parts.get(obj.data.name)
        if parts is None or not parts.matches(obj) or not len(parts.overlay_positions):
            continue
        if parts.overlay_batch is None:
            parts.overlay_batch = batch_for_shader(
                _overlay_shader, "TRIS", {"pos": parts.overlay_positions}
            )
        with gpu.matrix.push_pop():
            gpu.matrix.multiply_matrix(obj.matrix_world)
            parts.overlay_batch.draw(_overlay_shader)
    gpu.state.depth_test_set("NONE")
    gpu.state.blend_set("NONE")


def _on_depsgraph_update(scene, depsgraph):
    global _expects_update, _geometry_updated
    if not any(update.is_updated_geometry for update in depsgraph.updates):
        return
    if _expects_update:
        _expects_update = False
        return
    _geometry_updated = True


def _on_undo(scene):
    global _undone
    _undone = True


_HANDLERS = (
    (bpy.app.handlers.depsgraph_update_post, _on_depsgraph_update),
    (bpy.app.handlers.undo_post, _on_undo),
    (bpy.app.handlers.redo_post, _on_undo),
)


def start_overlay():
    global _overlay_handler
    if _overlay_handler is None:
        _overlay_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_overlay, (), "WINDOW", "POST_VIEW"
        )
        for handlers, handler in _HANDLERS:
            handlers.append(handler)
    tag_redraw(("WINDOW",))


def stop_part_overrides():
    global _overlay_handler, _is_editing
    global _expects_update, _geometry_updated, _undone
    # file loads end the modal operator without calling modal again
    _is_editing = False
    _expects_update = _geometry_updated = _undone = False
    if _overlay_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_overlay_handler, "WINDOW")
        _overlay_handler = None
    # file loads remove handlers not marked persistent
    for handlers, handler in _HANDLERS:
        if handler in handlers:
            handlers.remove(handler)
    _parts.clear()
    tag_redraw(("WINDOW",))
