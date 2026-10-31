import contextlib
from collections import namedtuple

import bmesh
import bmesh.utils
import bpy
import mathutils
import numpy

from ..no_bpy.mesh import split_per_face, transform_positions, undirected_edge_keys
from ..no_bpy.mesh_reads import (
    loop_totals,
    loop_vertices_and_face_sizes,
    vertex_position_array,
)


def new_bmesh(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    return bm


def face_vertices(mesh):
    corners, totals = loop_vertices_and_face_sizes(mesh)
    return split_per_face(corners.tolist(), totals.tolist())


MeshArrays = namedtuple("MeshArrays", "positions corners totals")


def mesh_arrays(mesh):
    return MeshArrays(vertex_position_array(mesh), *loop_vertices_and_face_sizes(mesh))


def edge_vertex_pairs(mesh):
    pairs = numpy.empty(len(mesh.edges) * 2, dtype=numpy.int64)
    mesh.edges.foreach_get("vertices", pairs)
    return pairs.reshape(-1, 2)


def vertex_positions(mesh):
    return vertex_position_array(mesh).tolist()


def world_positions(mesh, matrix):
    return transform_positions(vertex_position_array(mesh), matrix)


def lifted_triangles(mesh, positions, lift):
    triangles = numpy.empty(len(mesh.loop_triangles) * 3, dtype=numpy.int64)
    mesh.loop_triangles.foreach_get("vertices", triangles)
    triangle_faces = numpy.empty(len(mesh.loop_triangles), dtype=numpy.int64)
    mesh.loop_triangle_polygons.foreach_get("value", triangle_faces)
    normals = numpy.empty(len(mesh.polygons) * 3)
    mesh.polygons.foreach_get("normal", normals)
    offsets = normals.reshape(-1, 3)[triangle_faces] * lift
    corners = positions[triangles.reshape(-1, 3)] + offsets[:, None, :]
    return corners, triangle_faces


def loop_starts(mesh):
    starts = numpy.empty(len(mesh.polygons), dtype=numpy.int64)
    mesh.polygons.foreach_get("loop_start", starts)
    return starts


def loop_uvs(mesh):
    flat = numpy.empty(len(mesh.loops) * 2)
    mesh.uv_layers.active.data.foreach_get("uv", flat)
    return flat.reshape(-1, 2)


def set_loop_uvs(mesh, coords):
    mesh.uv_layers.active.data.foreach_set("uv", coords.ravel())


def corner_uvs(mesh):
    corners = [tuple(uv) for uv in loop_uvs(mesh).tolist()]
    return split_per_face(corners, loop_totals(mesh).tolist())


# uvs of loops on one vertex differ by float noise
UV_ROUNDING_DIGITS = 6


def face_uvs(mesh):
    return [
        [(round(u, UV_ROUNDING_DIGITS), round(v, UV_ROUNDING_DIGITS)) for u, v in face]
        for face in corner_uvs(mesh)
    ]


def triangulate(bm, mesh):
    # BEAUTY alone can give two quads the same diagonal, leaving an edge with four faces
    _split_conflicting_quads(bm, mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces, quad_method="BEAUTY")


def triangulate_mesh(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    if any(len(face.verts) > 3 for face in bm.faces):
        triangulate(bm, mesh)
        bm.to_mesh(mesh)
    bm.free()


ConflictingQuad = namedtuple(
    "ConflictingQuad", ["face", "diagonals", "diagonal_is_edge"]
)


def _quads_with_diagonal_already_used(mesh):
    corners, totals = loop_vertices_and_face_sizes(mesh)
    quads = numpy.flatnonzero(totals == 4)
    if len(quads) == 0:
        return []
    edge_vertices = edge_vertex_pairs(mesh)

    quad_corners = corners[loop_starts(mesh)[quads][:, None] + numpy.arange(4)]
    diagonals = numpy.stack(
        [
            undirected_edge_keys(quad_corners[:, 0], quad_corners[:, 2]),
            undirected_edge_keys(quad_corners[:, 1], quad_corners[:, 3]),
        ],
        axis=1,
    )
    mesh_edges = numpy.sort(
        undirected_edge_keys(edge_vertices[:, 0], edge_vertices[:, 1])
    )
    # searchsorted is seven times faster than isin here
    slots = numpy.searchsorted(mesh_edges, diagonals).clip(max=len(mesh_edges) - 1)
    is_edge = mesh_edges[slots] == diagonals
    _, inverse, counts = numpy.unique(
        diagonals.ravel(), return_inverse=True, return_counts=True
    )
    is_shared = counts[inverse].reshape(diagonals.shape) > 1
    flagged = numpy.flatnonzero((is_edge | is_shared).any(axis=1))
    return [
        ConflictingQuad(face, keys, edge_flags)
        for face, keys, edge_flags in zip(
            quads[flagged].tolist(),
            diagonals[flagged].tolist(),
            is_edge[flagged].tolist(),
        )
    ]


def _split_conflicting_quads(bm, mesh):
    conflicting = _quads_with_diagonal_already_used(mesh)
    if not conflicting:
        return
    bm.faces.ensure_lookup_table()
    conflicting = [quad._replace(face=bm.faces[quad.face]) for quad in conflicting]

    claims = {}
    for quad in conflicting:
        for key in quad.diagonals:
            claims.setdefault(key, []).append(quad.face)

    split_keys = set()
    for quad in conflicting:
        face = quad.face
        keys = quad.diagonals
        verts = face.verts
        ends = {keys[0]: (verts[0], verts[2]), keys[1]: (verts[1], verts[3])}
        edge_of = dict(zip(keys, quad.diagonal_is_edge))

        def conflicts(key):
            return (
                edge_of[key]
                or key in split_keys
                or any(
                    other is not face and len(other.verts) == 4 for other in claims[key]
                )
            )

        if not any(conflicts(key) for key in keys):
            continue
        safest = min(
            keys,
            key=lambda diagonal: (
                conflicts(diagonal),
                (ends[diagonal][0].co - ends[diagonal][1].co).length,
            ),
        )
        bmesh.utils.face_split(face, *ends[safest])
        split_keys.add(safest)


def set_bmesh(bm, obj):
    if obj.mode == "EDIT":
        bmesh.update_edit_mesh(obj.data)
    else:
        bm.to_mesh(obj.data)
    bm.free()


def move_to_collection(obj, target):
    for collection in obj.users_collection:
        collection.objects.unlink(obj)
    target.objects.link(obj)


def check_collection(name):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
    if not bpy.context.scene.user_of_id(collection):
        bpy.context.scene.collection.children.link(collection)
    return collection


NOT_UNWRAPPED_COLLECTION = "UVgami Not Unwrapped"


def mark_not_unwrapped(obj, reason, name=None):
    move_to_collection(obj, check_collection(NOT_UNWRAPPED_COLLECTION))
    # long names are cut short in the outliner
    obj.name = f"{reason}: {name or obj.name}"
    obj.hide_set(True)
    return obj.name


def check_exists(reference):
    try:
        reference.name
        return True
    except ReferenceError:
        return False


def validate_obj(op, obj, report=False):
    if obj.type != "MESH":
        if report:
            op.report({"ERROR"}, "Selected object is not a mesh")
        return False
    if len(obj.data.polygons) == 0:
        if report:
            op.report({"ERROR"}, "Selected object has zero polygons")
        return False
    return True


def deselect_all():
    for obj in bpy.context.selected_objects:
        obj.select_set(False)


@contextlib.contextmanager
def _shown(objects):
    view_layer = bpy.context.view_layer
    targets = set(objects)
    restore = []

    def clear(holder, attr):
        if getattr(holder, attr):
            restore.append((holder, attr, True))
            setattr(holder, attr, False)

    def walk(layer_collection):
        if targets & set(layer_collection.collection.objects):
            clear(layer_collection, "exclude")
            clear(layer_collection, "hide_viewport")
            clear(layer_collection.collection, "hide_viewport")
        for child in layer_collection.children:
            walk(child)

    walk(view_layer.layer_collection)
    for obj in targets:
        clear(obj, "hide_viewport")
        if obj.hide_get(view_layer=view_layer):
            restore.append((obj, None, True))
            obj.hide_set(False, view_layer=view_layer)

    try:
        yield
    finally:
        for holder, attr, value in reversed(restore):
            if attr is None:
                holder.hide_set(value, view_layer=view_layer)
            else:
                setattr(holder, attr, value)


def edit_restore(objects, func, *args):
    old_selection = bpy.context.selected_objects
    old_active = bpy.context.view_layer.objects.active

    # hidden active objects fail the mode_set poll the same as no active object
    if old_active is None or not old_active.visible_get():
        old_active = next(
            (
                obj
                for obj in bpy.data.objects
                if obj.type == "MESH" and obj.users_collection and obj.visible_get()
            ),
            None,
        )
        if old_active is not None:
            bpy.context.view_layer.objects.active = old_active

    old_mode = old_active.mode if old_active is not None else "OBJECT"

    if old_active is not None:
        bpy.ops.object.mode_set(mode="OBJECT")

    # hidden objects are left out of objects_in_mode even after mode_set succeeds
    with _shown(objects):
        deselect_all()
        for obj in objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = objects[0]
        bpy.ops.object.mode_set(mode="EDIT")

        func(*args)

        bpy.ops.object.mode_set(mode="OBJECT")

    deselect_all()
    for obj in old_selection:
        obj.select_set(True)
    if old_active is not None:
        bpy.context.view_layer.objects.active = old_active
        bpy.ops.object.mode_set(mode=old_mode)


def in_object_mode(target, func, *args):
    if target.mode == "OBJECT":
        return func(*args)
    old_active = bpy.context.view_layer.objects.active
    old_mode = target.mode
    bpy.ops.object.mode_set(mode="OBJECT")
    try:
        return func(*args)
    finally:
        if check_exists(old_active):
            bpy.context.view_layer.objects.active = old_active
            bpy.ops.object.mode_set(mode=old_mode)


AUTO_SMOOTH_MODIFIER_NAME = "Smooth by Angle"
WEIGHTED_NORMAL_PROPERTIES = (
    "weight",
    "mode",
    "thresh",
    "keep_sharp",
    "vertex_group",
    "invert_vertex_group",
    "use_face_influence",
)


def _is_auto_smooth(modifier):
    return modifier.type == "NODES" and AUTO_SMOOTH_MODIFIER_NAME in modifier.name


def is_shading_modifier(modifier):
    return _is_auto_smooth(modifier) or modifier.type == "WEIGHTED_NORMAL"


def _node_input_identifiers(modifier):
    return [
        item.identifier
        for item in modifier.node_group.interface.items_tree
        if item.item_type == "SOCKET"
        and item.in_out == "INPUT"
        and item.socket_type != "NodeSocketGeometry"
    ]


def _has_property_inputs(modifier):
    # from blender 5.2 geometry nodes inputs are on modifier.properties, not id properties
    return hasattr(modifier, "properties")


def _set_node_input_values(modifier, values):
    if _has_property_inputs(modifier):
        inputs = modifier.properties.inputs
        for key, value in values.items():
            getattr(inputs, key).value = value
        return
    for key, value in values.items():
        modifier[key] = value


def _node_input_values(modifier):
    identifiers = _node_input_identifiers(modifier)
    if _has_property_inputs(modifier):
        inputs = modifier.properties.inputs
        return {key: getattr(inputs, key).value for key in identifiers}
    return {key: modifier[key] for key in identifiers}


ShadingModifier = namedtuple(
    "ShadingModifier", ["kind", "name", "node_group_name", "settings"]
)


def get_shading_modifiers(obj):
    records = []
    for modifier in obj.modifiers:
        if _is_auto_smooth(modifier) and modifier.node_group is not None:
            records.append(
                ShadingModifier(
                    modifier.type,
                    modifier.name,
                    modifier.node_group.name,
                    _node_input_values(modifier),
                )
            )
        elif modifier.type == "WEIGHTED_NORMAL":
            settings = {
                key: getattr(modifier, key) for key in WEIGHTED_NORMAL_PROPERTIES
            }
            records.append(
                ShadingModifier(modifier.type, modifier.name, None, settings)
            )
    return records


def add_shading_modifiers(obj, records):
    for record in records:
        if record.kind == "NODES":
            node_group = bpy.data.node_groups.get(record.node_group_name)
            if node_group is None:
                continue
            modifier = obj.modifiers.new(record.name, record.kind)
            modifier.node_group = node_group
            modifier.use_pin_to_last = True
            _set_node_input_values(modifier, record.settings)
        else:
            modifier = obj.modifiers.new(record.name, record.kind)
            for key, value in record.settings.items():
                setattr(modifier, key, value)


def applied_matrix(obj):
    location, _, scale = obj.matrix_basis.decompose()
    return (
        mathutils.Matrix.Translation(location)
        @ obj.matrix_basis.to_3x3().normalized().to_4x4()
        @ mathutils.Matrix.Diagonal(scale).to_4x4()
    )
