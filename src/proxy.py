import collections

import bmesh
import bpy
import numpy
from mathutils import Matrix, Vector

# from mathutils.kdtree import KDTree
from .hard_surface import apply_seams
from .job import Transfer, apply_island_uvs_and_seams, island_edges_on_mesh
from .no_bpy import fit_uvs_keeping_area
from .no_bpy.mesh import face_starts, lowest_index_part_labels
from .no_bpy.proxy_transfer import dense_subset, start_proxy_transfer
from .symmetry import cut_on_axis
from .utils.mesh import (
    check_exists,
    corner_uvs,
    edge_vertex_pairs,
    face_vertices,
    loop_vertices_and_face_sizes,
    new_bmesh,
    set_bmesh,
    set_loop_uvs,
    triangulate_mesh,
    vertex_position_array,
)
from .utils.paths import get_extension_dir_path
from .utils.uvgami_tools import tools_command


def triangle_count(obj):
    return len(obj.data.loops) - 2 * len(obj.data.polygons)


PartTriangles = collections.namedtuple(
    "PartTriangles", "vertex_labels part_labels triangle_counts"
)


def part_triangles(data):
    labels = lowest_index_part_labels(len(data.vertices), edge_vertex_pairs(data))
    corners, totals = loop_vertices_and_face_sizes(data)
    part_labels, face_part = numpy.unique(
        labels[corners[face_starts(totals)]], return_inverse=True
    )
    counts = numpy.bincount(face_part, weights=totals - 2).astype(numpy.int64)
    return PartTriangles(labels, part_labels, counts)


def needs_proxy(obj, target_faces, parts=None):
    if triangle_count(obj) <= target_faces:
        return False
    if parts is None:
        parts = part_triangles(obj.data)
    return int(parts.triangle_counts.max()) > target_faces


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
    # each decimate ratio is a fraction of the previous modifier's output
    remaining = triangles
    for part in over:
        target = remaining - (int(counts[part]) - target_faces)
        modifier = obj.modifiers.new("UVgami Proxy", "DECIMATE")
        modifier.ratio = target / remaining
        remaining = target
        modifiers.append(modifier)
        if len(part_labels) > 1:
            group = obj.vertex_groups.new(name="UVgami Proxy")
            part_vertices = numpy.flatnonzero(labels == part_labels[part])
            group.add(part_vertices.tolist(), 1.0, "REPLACE")
            modifier.vertex_group = group.name
            groups.append(group)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    # preserve_all_data_layers runs the decimate a second time
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
        # hidden objects are left out of the depsgraph
        raise RuntimeError(f"{obj.name} was not decimated, it has to be visible")

    clean_proxy(obj)
    return True


def values_carried_through(obj, attribute_name, values, operation):
    obj.data.attributes.new(attribute_name, "INT", "FACE").data.foreach_set(
        "value", values
    )
    operation()
    attribute = obj.data.attributes[attribute_name]
    carried = numpy.empty(len(obj.data.polygons), dtype=numpy.int32)
    attribute.data.foreach_get("value", carried)
    obj.data.attributes.remove(attribute)
    return carried


SOURCE_FACE_ATTRIBUTE = "uvgami_proxy_source_face"


def mirror_proxy(obj, symmetry):
    center = Vector(symmetry.center)

    def mirror():
        cut_on_axis(obj, center, symmetry.axis, symmetry.tolerance, symmetry.kept_side)
        bm = new_bmesh(obj)
        bmesh.ops.mirror(
            bm,
            geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
            matrix=Matrix.Translation(-center),
            merge_dist=symmetry.tolerance,
            axis=symmetry.axis,
        )
        set_bmesh(bm, obj)

    source_faces = numpy.arange(len(obj.data.polygons), dtype=numpy.int32)
    return values_carried_through(obj, SOURCE_FACE_ATTRIBUTE, source_faces, mirror)


FOLD_DOT = -0.8
# flips can fold another triangle
UNFOLD_ROUNDS = 5


def clean_proxy(obj):
    bm = new_bmesh(obj)
    # collapse decimation leaves loose vertices and folded triangles
    loose = [vertex for vertex in bm.verts if not vertex.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    unfold(bm)
    set_bmesh(bm, obj)


def unfold(bm):
    for _ in range(UNFOLD_ROUNDS):
        bm.normal_update()
        inverted = [face for face in bm.faces if _inverted(face)]
        if not inverted:
            return
        for face in inverted:
            # earlier flips next to the face can remove it
            if face.is_valid:
                _flip(max(face.edges, key=lambda edge: edge.calc_length()))


def _flip(edge):
    if len(edge.link_faces) != 2:
        return
    apexes = [
        next(vert for vert in face.verts if vert not in edge.verts)
        for face in edge.link_faces
    ]
    if apexes[0] is apexes[1]:
        return
    if any(other.other_vert(apexes[0]) is apexes[1] for other in apexes[0].link_edges):
        return
    # bmesh.ops.rotate_edges leaves a folded edge as it is
    quad = bmesh.utils.face_join(edge.link_faces)
    bmesh.utils.face_split(quad, apexes[0], apexes[1])


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


def transfer_inputs(input_mesh, output):
    data = input_mesh.data
    corners, totals = loop_vertices_and_face_sizes(data)
    dense = {
        "positions": vertex_position_array(data),
        "normals": _vertex_normals(data),
        "matrix": numpy.array(input_mesh.matrix_world, dtype=numpy.float64),
        "corners": corners,
        "face_sizes": totals,
    }
    output_data = output.data
    proxy = {
        "positions": vertex_position_array(output_data),
        "matrix": numpy.array(output.matrix_world, dtype=numpy.float64),
        "faces": face_vertices(output_data),
        "corner_uvs": corner_uvs(output_data),
    }
    return dense, proxy


class ProxyUVs(Transfer):
    allows_missing_pieces = False

    def _extract(self, target, output):
        return transfer_inputs(target, output)

    def _begin(self, inputs):
        self.task = start_proxy_transfer(
            tools_command(), get_extension_dir_path() / "proxy", *inputs
        )

    def _apply(self, target, result):
        seams, uvs = result
        data = target.data
        apply_seams(data, seams)
        if not data.uv_layers:
            data.uv_layers.new()
        set_loop_uvs(data, uvs)
        return 0


class ProxyCopyUVs(ProxyUVs):
    repack_input = False

    def _target(self, input_mesh):
        target = input_mesh.copy()
        target.data = input_mesh.data.copy()
        triangulate_mesh(target.data)
        return target

    def _show(self):
        self.input_mesh.hide_set(True)
        self.target.name = f"{self.input_mesh.name}_unwrapped"
        self.replacement = self.target

    def _discard_start_extras(self):
        if check_exists(self.target):
            bpy.data.objects.remove(self.target, do_unlink=True)


class ProxyIslandUVs(ProxyUVs):
    repack_input = False

    def __init__(self, input_mesh, faces, bbox, area):
        super().__init__(input_mesh)
        self.faces = faces
        self.bbox = bbox
        self.area = area
        self.original_vertices = []

    def _extract(self, target, output):
        dense, proxy = transfer_inputs(target, output)
        dense, self.original_vertices = dense_subset(dense, self.faces)
        return dense, proxy

    def _apply(self, target, result):
        seams, uvs = result
        data = target.data
        sizes = [data.polygons[face].loop_total for face in self.faces]
        polygons = [
            part.tolist() for part in numpy.split(uvs, numpy.cumsum(sizes)[:-1])
        ]
        move = fit_uvs_keeping_area(polygons, self.area, self.bbox)
        apply_island_uvs_and_seams(
            data,
            self.faces,
            numpy.array([move(uv) for points in polygons for uv in points]),
            island_edges_on_mesh(self.original_vertices.tolist(), seams),
        )
        return 0
