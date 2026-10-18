import bmesh
import bpy
import gpu
import numpy
from gpu_extras.batch import batch_for_shader

from ..engines.optcuts import SURFACE_MODES
from ..part_overrides import (
    KEEP_ATTRIBUTE,
    OVERRIDE_ATTRIBUTES,
    # PRIORITY_ATTRIBUTE,
    # PRIORITY_ORDER,
    SURFACE_MODE_ATTRIBUTE,
    SURFACE_MODE_CODES,
    # UV_SCALE_ATTRIBUTE,
)

# from ..ui.props import PRIORITY_LABELS
from ..seams.proxy_transfer import connected_labels
from ..utils.mesh import (
    face_vertex_arrays,
    lifted_triangles,
    loop_starts,
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


def _layer_values(bm, name):
    layer = _layer(bm, name)
    if layer is None:
        return numpy.zeros(len(bm.faces))
    return numpy.fromiter((f[layer] for f in bm.faces), float, len(bm.faces))


def _face_selection(bm):
    return numpy.fromiter((f.select for f in bm.faces), bool, len(bm.faces))


def _topology(bm):
    return (len(bm.verts), len(bm.edges), len(bm.faces))


# marks this module's mesh writes for the depsgraph handler
_expects_update = False
_geometry_updated = False
_undone = False


def _update_edit_mesh(mesh):
    global _expects_update
    bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
    _expects_update = True


# foreach_get reads stale values until the edit mesh is written back
def _written_back(obj):
    global _expects_update
    obj.update_from_editmode()
    _expects_update = True
    return obj.data


class MeshParts:
    def __init__(self, obj):
        self.topology = _topology(bmesh.from_edit_mesh(obj.data))
        mesh = _written_back(obj)
        edges = numpy.empty(len(mesh.edges) * 2, dtype=numpy.int64)
        mesh.edges.foreach_get("vertices", edges)
        corners, _ = face_vertex_arrays(mesh)
        labels = connected_labels(len(mesh.vertices), edges.reshape(-1, 2))
        vertex_part = numpy.unique(labels, return_inverse=True)[1]
        self.ids = vertex_part[corners[loop_starts(mesh)]]
        self.count = int(self.ids.max()) + 1 if len(self.ids) else 0
        self._read_geometry(mesh)

    def refresh(self, obj):
        self._read_geometry(_written_back(obj))

    def _read_geometry(self, mesh):
        # attribute reads through the datablock are empty in edit mode
        bm = bmesh.from_edit_mesh(mesh)
        values = [_layer_values(bm, name) for name in OVERRIDE_ATTRIBUTES]
        self.overridden = numpy.any([v != 0 for v in values], axis=0)
        positions = vertex_position_array(mesh)
        extent = numpy.ptp(positions, axis=0) if len(positions) else numpy.zeros(3)
        # the depth test hides an overlay flat on the surface
        lift = 1e-4 * float(numpy.linalg.norm(extent))
        corners, self.triangle_faces = lifted_triangles(mesh, positions, lift)
        self.corner_positions = corners.reshape(-1, 3).astype(numpy.float32)


# keyed by mesh name
_parts = {}


def mesh_parts(obj):
    name = obj.data.name
    parts = _parts.get(name)
    if parts is None or parts.topology != _topology(bmesh.from_edit_mesh(obj.data)):
        parts = _parts[name] = MeshParts(obj)
        _batches.pop(name, None)
        tag_redraw(("WINDOW",))
    return parts


def _edited_objects():
    return [
        obj for obj in bpy.context.objects_in_mode_unique_data if obj.type == "MESH"
    ]


# reading the selection of a big mesh slows every panel redraw
_selected_part_counts = {}


def selected_part_count():
    count = 0
    for obj in _edited_objects():
        mesh = obj.data
        parts = mesh_parts(obj)
        key = (_selection_state(mesh), parts)
        cached = _selected_part_counts.get(mesh.name)
        if cached is None or cached[0] != key:
            selected = _face_selection(bmesh.from_edit_mesh(mesh))
            cached = (key, len(numpy.unique(parts.ids[selected])))
            _selected_part_counts[mesh.name] = cached
        count += cached[1]
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


# a part with one selected face gets the value on all of its faces
def _write_selected_parts(name, value):
    for obj in _edited_objects():
        mesh = obj.data
        bm = bmesh.from_edit_mesh(mesh)
        parts = mesh_parts(obj)
        selected = _face_selection(bm)
        if not selected.any():
            continue
        # a reset has nothing to clear on a mesh without the layer
        layer = _layer(bm, name, create=value != 0)
        if layer is None:
            continue
        bm.faces.ensure_lookup_table()
        targets = numpy.isin(parts.ids, parts.ids[selected])
        for index in numpy.flatnonzero(targets).tolist():
            bm.faces[index][layer] = value
        _update_edit_mesh(mesh)
        parts.refresh(obj)
    _batches.clear()


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


# read from the active face and written to every selected part
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


# a click on one face selects its whole part, a click that removes faces drops the part
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
        # (face selection count, active face index, selection) per mesh name
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
            if seen is not None and seen[:2] == state:
                continue
            selection_changed = True
            previous = None if seen is None else seen[2]
            selection = _select_whole_parts(mesh, parts, previous)
            self.seen[mesh.name] = (*_selection_state(mesh), selection)
        if not _geometry_updated and not _undone:
            return
        # a mesh edit is a geometry update with no selection change, or an undo
        stale = _undone or not selection_changed
        _geometry_updated = _undone = False
        if not stale:
            return
        for obj in objects:
            mesh_parts(obj).refresh(obj)
        _batches.clear()
        tag_redraw(("WINDOW",))


class UVGAMI_OT_exit_part_overrides(bpy.types.Operator):
    bl_idname = "uvgami.exit_part_overrides"
    bl_label = "Exit Edit Parts"
    bl_description = "Go back to the previous mode"

    def execute(self, context):
        # leaving Edit Mode ends the modal
        mode = _old_mode if _old_mode not in (None, "EDIT") else "OBJECT"
        bpy.ops.object.mode_set(mode=mode)
        return {"FINISHED"}


# batch per mesh name
_batches = {}
_overlay_handler = None
_overlay_shader = None


def _build_batch(parts):
    visible = parts.overridden[parts.triangle_faces]
    if not visible.any():
        return None
    positions = parts.corner_positions.reshape(-1, 3, 3)[visible].reshape(-1, 3)
    return batch_for_shader(_overlay_shader, "TRIS", {"pos": positions})


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
        # the modal rebuilds a stale table on its next event
        if parts is None or parts.topology != _topology(bmesh.from_edit_mesh(obj.data)):
            continue
        if obj.data.name not in _batches:
            _batches[obj.data.name] = _build_batch(parts)
        batch = _batches[obj.data.name]
        if batch is None:
            continue
        with gpu.matrix.push_pop():
            gpu.matrix.multiply_matrix(obj.matrix_world)
            batch.draw(_overlay_shader)
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


# a file load drops the modal without calling it again
def stop_part_overrides():
    global _overlay_handler, _is_editing
    global _expects_update, _geometry_updated, _undone
    _is_editing = False
    _expects_update = _geometry_updated = _undone = False
    if _overlay_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_overlay_handler, "WINDOW")
        _overlay_handler = None
    # a file load clears these lists itself
    for handlers, handler in _HANDLERS:
        if handler in handlers:
            handlers.remove(handler)
    _batches.clear()
    _parts.clear()
    _selected_part_counts.clear()
    tag_redraw(("WINDOW",))
