import bmesh
import numpy
from mathutils.kdtree import KDTree

from .part_overrides import KEEP_ATTRIBUTE, face_values, surface_modes
from .proxy import make_proxy, part_triangles
from .seams.segmentation import segment_job

# from .seams.segmentation import split_loop_job, split_region_job
from .seams.worker import WorkerProcess
from .utils.mesh import (
    face_vertex_arrays,
    new_bmesh,
    set_bmesh,
    vertex_position_array,
)
from .utils.paths import get_extension_dir_path
from .utils.uvgami_tools import tools_command

# # hand edited region numbers per face from one up, zero outside the regions
# REGION_ATTRIBUTE = "uvgami_region"


# # object mode only, an edit mode mesh reads stale values
# def region_labels(mesh):
#     labels = numpy.zeros(len(mesh.polygons), dtype=numpy.int32)
#     attribute = mesh.attributes.get(REGION_ATTRIBUTE)
#     if attribute is not None and attribute.domain == "FACE":
#         attribute.data.foreach_get("value", labels)
#     return labels.astype(numpy.int64)


# def write_region_labels(mesh, labels):
#     attribute = mesh.attributes.get(REGION_ATTRIBUTE) or mesh.attributes.new(
#         REGION_ATTRIBUTE, "INT", "FACE"
#     )
#     attribute.data.foreach_set("value", labels.astype(numpy.int32))


# def clear_region_labels(mesh):
#     attribute = mesh.attributes.get(REGION_ATTRIBUTE)
#     if attribute is not None:
#         mesh.attributes.remove(attribute)


def segment_work(mesh, positions, segment_command, unset_mode):
    return WorkerProcess(
        segment_job,
        segment_command,
        tools_command(),
        get_extension_dir_path() / "segment",
        positions,
        *face_vertex_arrays(mesh),
        surface_modes(mesh, unset_mode),
        face_values(mesh, KEEP_ATTRIBUTE) != 0,
        # region_labels(mesh),
    )


# def split_region_work(mesh, segment_command, labels, seed, piece, step):
#     return WorkerProcess(
#         split_region_job,
#         segment_command,
#         get_extension_dir_path() / "segment",
#         vertex_position_array(mesh),
#         *face_vertex_arrays(mesh),
#         labels,
#         seed,
#         piece,
#         step,
#     )


# def split_loop_work(mesh, segment_command, labels, seed, corner):
#     return WorkerProcess(
#         split_loop_job,
#         segment_command,
#         get_extension_dir_path() / "segment",
#         vertex_position_array(mesh),
#         *face_vertex_arrays(mesh),
#         labels,
#         seed,
#         corner,
#     )


# returns the border vertex positions and part_triangles of the mesh before the split
def split_on_borders(obj, border_edges):
    mesh = obj.data
    labels, part_labels, counts = part_triangles(mesh)
    corner_parts = labels[face_vertex_arrays(mesh)[0]]

    bm = new_bmesh(obj)
    bm.verts.ensure_lookup_table()
    edges = [bm.edges.get((bm.verts[a], bm.verts[b])) for a, b in border_edges]
    border_vertices = {vertex for edge in edges for vertex in edge.verts}
    border_positions = numpy.array([tuple(vertex.co) for vertex in border_vertices])
    bmesh.ops.split_edges(bm, edges=edges)
    set_bmesh(bm, obj)

    # the split keeps face and corner order, so each corner keeps its part
    vertex_labels = numpy.zeros(len(mesh.vertices), dtype=labels.dtype)
    vertex_labels[: len(labels)] = labels
    vertex_labels[face_vertex_arrays(mesh)[0]] = corner_parts
    return border_positions, (vertex_labels, part_labels, counts)


RESOLVED_MODE_ATTRIBUTE = "uvgami_resolved_surface_mode"


def make_segmented_proxy(obj, target_faces, parts, surface_modes):
    obj.data.attributes.new(RESOLVED_MODE_ATTRIBUTE, "INT", "FACE").data.foreach_set(
        "value", surface_modes
    )
    make_proxy(obj, target_faces, parts)
    # make_proxy swaps in a new mesh
    attribute = obj.data.attributes[RESOLVED_MODE_ATTRIBUTE]
    decimated_modes = numpy.empty(len(obj.data.polygons), dtype=numpy.int32)
    attribute.data.foreach_get("value", decimated_modes)
    obj.data.attributes.remove(attribute)
    return decimated_modes


# share of the output's bounding box diagonal
WELD_TOLERANCE = 1e-5


# merges the split borders back on the output, an artist's own doubles elsewhere stay
class SegmentWeld:
    def __init__(self, border_positions, matrix):
        # the output is the input moved by matrix, then set back to its origin
        rotation_scale = numpy.array(matrix.to_3x3())
        self.border_positions = border_positions @ rotation_scale.T

    def finish(self, output):
        positions = vertex_position_array(output.data)
        diagonal = numpy.linalg.norm(positions.max(axis=0) - positions.min(axis=0))
        tolerance = WELD_TOLERANCE * float(diagonal)
        tree = KDTree(len(self.border_positions))
        for index, position in enumerate(self.border_positions.tolist()):
            tree.insert(position, index)
        tree.balance()

        bm = new_bmesh(output)
        near = [vertex for vertex in bm.verts if tree.find(vertex.co)[2] <= tolerance]
        bmesh.ops.remove_doubles(bm, verts=near, dist=tolerance)
        set_bmesh(bm, output)
