import bmesh
import bpy
import numpy
from mathutils import Matrix, Vector

# from mathutils.kdtree import KDTree
from .seams.mesh import connected_labels
from .utils.geometry import cut_on_axis
from .utils.mesh import (
    corner_uvs,
    face_vertex_arrays,
    face_vertices,
    new_bmesh,
    set_bmesh,
    vertex_position_array,
)


# every face fans into loop_total - 2 triangles
def triangle_count(obj):
    return len(obj.data.loops) - 2 * len(obj.data.polygons)


def part_triangles(data):
    labels = connected_labels(len(data.vertices), _edge_array(data))
    corners, totals = face_vertex_arrays(data)
    starts = numpy.concatenate(([0], numpy.cumsum(totals)[:-1]))
    part_labels, face_part = numpy.unique(labels[corners[starts]], return_inverse=True)
    counts = numpy.bincount(face_part, weights=totals - 2).astype(numpy.int64)
    return labels, part_labels, counts


# the budget is per loose part, parts is the part_triangles a caller already has
def needs_proxy(obj, target_faces, parts=None):
    if triangle_count(obj) <= target_faces:
        return False
    _, _, counts = part_triangles(obj.data) if parts is None else parts
    return int(counts.max()) > target_faces


# the bake also applies the object's other modifiers
def make_proxy(obj, target_faces, parts=None):
    if parts is None:
        parts = part_triangles(obj.data)
    if not needs_proxy(obj, target_faces, parts):
        return False
    labels, part_labels, counts = parts
    over = numpy.flatnonzero(counts > target_faces).tolist()
    triangles = triangle_count(obj)

    modifiers = []
    groups = []
    # a ratio is a fraction of the previous modifier's output
    remaining = triangles
    for part in over:
        target = remaining - (int(counts[part]) - target_faces)
        modifier = obj.modifiers.new("UVgami Proxy", "DECIMATE")
        modifier.ratio = target / remaining
        remaining = target
        modifiers.append(modifier)
        if len(part_labels) > 1:
            group = obj.vertex_groups.new(name="UVgami Proxy")
            part_verts = numpy.flatnonzero(labels == part_labels[part])
            group.add(part_verts.tolist(), 1.0, "REPLACE")
            modifier.vertex_group = group.name
            groups.append(group)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    # preserve_all_data_layers would run the decimate a second time
    baked = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph))

    for modifier in modifiers:
        obj.modifiers.remove(modifier)
    # swapping in the baked mesh drops the object's group list
    for group in groups:
        obj.vertex_groups.remove(group)
    stale = obj.data
    obj.data = baked
    bpy.data.meshes.remove(stale)

    if triangle_count(obj) == triangles:
        # a hidden object is left out of the depsgraph
        raise RuntimeError(f"{obj.name} was not decimated, it has to be visible")

    clean_proxy(obj)
    return True


SOURCE_FACE_ATTRIBUTE = "uvgami_proxy_source_face"


def mirror_proxy(obj, axis, center, tolerance, kept_side):
    obj.data.attributes.new(SOURCE_FACE_ATTRIBUTE, "INT", "FACE").data.foreach_set(
        "value", numpy.arange(len(obj.data.polygons), dtype=numpy.int32)
    )
    cut_on_axis(obj, center, axis, tolerance, kept_side)
    bm = new_bmesh(obj)
    bmesh.ops.mirror(
        bm,
        geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
        matrix=Matrix.Translation(-Vector(center)),
        merge_dist=tolerance,
        axis=axis,
    )
    set_bmesh(bm, obj)
    attribute = obj.data.attributes[SOURCE_FACE_ATTRIBUTE]
    source_faces = numpy.empty(len(obj.data.polygons), dtype=numpy.int32)
    attribute.data.foreach_get("value", source_faces)
    obj.data.attributes.remove(attribute)
    return source_faces


# neighbour normals this far apart are a triangle folded over the other
FOLD_DOT = -0.8
# a flip can fold another triangle
UNFOLD_ROUNDS = 5


# collapsing leaves loose vertices and folded triangles behind
def clean_proxy(obj):
    bm = new_bmesh(obj)
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    unfold(bm)
    set_bmesh(bm, obj)


# a collapse drags a vertex past its opposite edge and inverts the triangle
def unfold(bm):
    for _ in range(UNFOLD_ROUNDS):
        bm.normal_update()
        inverted = [face for face in bm.faces if _inverted(face)]
        if not inverted:
            return
        for face in inverted:
            # a flip next to it may have removed the face already
            if face.is_valid:
                _flip(max(face.edges, key=lambda edge: edge.calc_length()))


# bmesh.ops.rotate_edges leaves a folded edge as it is
def _flip(edge):
    if len(edge.link_faces) != 2:
        return
    apexes = [
        next(vert for vert in face.verts if vert not in edge.verts)
        for face in edge.link_faces
    ]
    # a doubled triangle has one apex
    if apexes[0] is apexes[1]:
        return
    if any(other.other_vert(apexes[0]) is apexes[1] for other in apexes[0].link_edges):
        return
    quad = bmesh.utils.face_join(edge.link_faces)
    bmesh.utils.face_split(quad, apexes[0], apexes[1])


# the vertex normals come from the properly oriented neighbours around the fold
def _inverted(face):
    folded = any(
        len(edge.link_faces) == 2
        and edge.link_faces[0].normal.dot(edge.link_faces[1].normal) < FOLD_DOT
        for edge in face.edges
    )
    if not folded:
        return False
    around = sum((vert.normal for vert in face.verts), Vector())
    return face.normal.dot(around) < 0


# # two copies of a model line up at any placement or size
# def bounds_frame(obj):
#     corners = [Vector(corner) for corner in obj.bound_box]
#     low = Vector([min(corner[axis] for corner in corners) for axis in range(3)])
#     high = Vector([max(corner[axis] for corner in corners) for axis in range(3)])
#     return Matrix.Scale(1 / max(high - low), 4) @ Matrix.Translation(-(low + high) / 2)


def _vertex_normals(data):
    flat = numpy.empty(len(data.vertices) * 3)
    data.vertices.foreach_get("normal", flat)
    return flat.reshape(-1, 3)


def _edge_array(data):
    pairs = numpy.empty(len(data.edges) * 2, dtype=numpy.int64)
    data.edges.foreach_get("vertices", pairs)
    return pairs.reshape(-1, 2)


# # candidates to pick a facing match from when snapping a cut vertex
# NEAREST_VERTS = 8


# def _rotation_array(matrix):
#     return numpy.array(Matrix(matrix.tolist()).to_3x3().inverted_safe().transposed())


# # on a thin wall the far side of the wall is the nearest
# def facing_matcher(
#     input_positions,
#     input_normals,
#     input_matrix,
#     output_positions,
#     output_normals,
#     output_matrix,
# ):
#     matrix = numpy.asarray(input_matrix, dtype=numpy.float64)
#     positions = numpy.asarray(input_positions).reshape(-1, 3)
#     positions = positions @ matrix[:3, :3].T + matrix[:3, 3]

#     normals = numpy.asarray(input_normals).reshape(-1, 3) @ _rotation_array(matrix).T
#     lengths = numpy.linalg.norm(normals, axis=1)
#     lengths[lengths == 0] = 1.0
#     normals /= lengths[:, None]

#     out_matrix = numpy.asarray(output_matrix, dtype=numpy.float64)
#     queries = numpy.asarray(output_positions).reshape(-1, 3)
#     queries = queries @ out_matrix[:3, :3].T + out_matrix[:3, 3]
#     # only the sign of the dot is read
#     facings = numpy.asarray(output_normals).reshape(-1, 3)
#     facings = facings @ _rotation_array(out_matrix).T

#     def nearest(input_indices, output_indices):
#         kd = KDTree(len(input_indices))
#         for i in input_indices:
#             kd.insert(positions[i], int(i))
#         kd.balance()
#         mapped = []
#         for i in output_indices:
#             found = kd.find_n(queries[i], NEAREST_VERTS)
#             best = found[0][1]
#             for _, index, _ in found:
#                 if normals[index] @ facings[i] > 0:
#                     best = index
#                     break
#             mapped.append(best)
#         return mapped

#     return nearest


# # in world space unless a frame is given for each
# def vertex_map(input_mesh, output, matrix=None, out_matrix=None):
#     if matrix is None:
#         matrix = input_mesh.matrix_world
#     if out_matrix is None:
#         out_matrix = output.matrix_world
#     nearest = facing_matcher(
#         vertex_position_array(input_mesh.data),
#         _vertex_normals(input_mesh.data),
#         numpy.array(matrix, dtype=numpy.float64),
#         vertex_position_array(output.data),
#         _vertex_normals(output.data),
#         numpy.array(out_matrix, dtype=numpy.float64),
#     )
#     return nearest(
#         range(len(input_mesh.data.vertices)), range(len(output.data.vertices))
#     )


# # another mesh's cut network redrawn along input_mesh's own edges
# def snap_cuts(input_mesh, mapped, cuts):
#     data = input_mesh.data
#     verts = vertex_position_array(data).tolist()
#     return proxy_transfer.snap_cuts(verts, _edge_array(data), mapped, cuts)


# the (dense, proxy) arrays the transfer pipeline reads
def transfer_inputs(input_mesh, output):
    data = input_mesh.data
    corners, totals = face_vertex_arrays(data)
    dense = {
        "positions": vertex_position_array(data),
        "normals": _vertex_normals(data),
        "matrix": numpy.array(input_mesh.matrix_world, dtype=numpy.float64),
        "corners": corners,
        "face_sizes": totals,
    }
    out_data = output.data
    proxy = {
        "positions": vertex_position_array(out_data),
        "matrix": numpy.array(output.matrix_world, dtype=numpy.float64),
        "faces": face_vertices(out_data),
        "corner_uvs": corner_uvs(out_data),
    }
    return dense, proxy
