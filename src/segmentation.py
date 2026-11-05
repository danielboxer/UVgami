from collections import namedtuple

import bmesh
import numpy
from mathutils.kdtree import KDTree

from .no_bpy.mesh import bounding_box_diagonal
from .no_bpy.segmentation import segment_job, stored_segment_job

# from .no_bpy.segmentation import split_loop_job, split_region_job
from .no_bpy.worker import WorkerProcess
from .part_overrides import KEEP_ATTRIBUTE, face_values, surface_modes
from .proxy import part_triangles
from .utils.mesh import (
    edge_vertex_pairs,
    loop_vertices_and_face_sizes,
    new_bmesh,
    set_bmesh,
    vertex_position_array,
)
from .utils.paths import get_extension_dir_path

# fine region numbers per face from one up, zero outside the regions, written by Edit Regions
REGION_ATTRIBUTE = "uvgami_region"
# the edges between fine regions that stay seams
BORDER_ATTRIBUTE = "uvgami_region_border"
# the edges of loops Edit Regions split a fine region along
LOOP_ATTRIBUTE = "uvgami_region_loop"

StoredRegions = namedtuple("StoredRegions", "fine_labels border_edges")


def stored_regions(mesh):
    regions = mesh.attributes.get(REGION_ATTRIBUTE)
    borders = mesh.attributes.get(BORDER_ATTRIBUTE)
    if regions is None or borders is None:
        return None
    fine_labels = numpy.zeros(len(mesh.polygons), dtype=numpy.int32)
    # object mode only, an edit mode mesh reads stale values
    regions.data.foreach_get("value", fine_labels)
    flags = numpy.zeros(len(mesh.edges), dtype=bool)
    borders.data.foreach_get("value", flags)
    return StoredRegions(
        fine_labels.astype(numpy.int64),
        numpy.sort(edge_vertex_pairs(mesh)[flags], axis=1),
    )


def stored_loop_edges(mesh):
    loops = mesh.attributes.get(LOOP_ATTRIBUTE)
    if loops is None:
        return numpy.zeros((0, 2), dtype=numpy.int64)
    flags = numpy.zeros(len(mesh.edges), dtype=bool)
    loops.data.foreach_get("value", flags)
    return numpy.sort(edge_vertex_pairs(mesh)[flags], axis=1)


def write_regions(mesh, fine_labels, border_flags, loop_flags):
    regions = mesh.attributes.get(REGION_ATTRIBUTE) or mesh.attributes.new(
        REGION_ATTRIBUTE, "INT", "FACE"
    )
    regions.data.foreach_set("value", fine_labels.astype(numpy.int32))
    borders = mesh.attributes.get(BORDER_ATTRIBUTE) or mesh.attributes.new(
        BORDER_ATTRIBUTE, "BOOLEAN", "EDGE"
    )
    borders.data.foreach_set("value", border_flags)
    loops = mesh.attributes.get(LOOP_ATTRIBUTE) or mesh.attributes.new(
        LOOP_ATTRIBUTE, "BOOLEAN", "EDGE"
    )
    loops.data.foreach_set("value", loop_flags)


def clear_regions(mesh):
    for name in (REGION_ATTRIBUTE, BORDER_ATTRIBUTE, LOOP_ATTRIBUTE):
        attribute = mesh.attributes.get(name)
        if attribute is not None:
            mesh.attributes.remove(attribute)


def segment_work(mesh, positions, segment_command, unset_mode, *, fine_cut):
    stored = stored_regions(mesh)
    if stored is not None:
        return WorkerProcess(
            stored_segment_job,
            segment_command,
            get_extension_dir_path() / "segment",
            positions,
            *loop_vertices_and_face_sizes(mesh),
            surface_modes(mesh, unset_mode),
            *stored,
        )
    return WorkerProcess(
        segment_job,
        segment_command,
        get_extension_dir_path() / "segment",
        positions,
        *loop_vertices_and_face_sizes(mesh),
        surface_modes(mesh, unset_mode),
        face_values(mesh, KEEP_ATTRIBUTE) != 0,
        fine_cut=fine_cut,
    )


# def split_region_work(mesh, segment_command, labels, seed, piece, step):
#     return WorkerProcess(
#         split_region_job,
#         segment_command,
#         get_extension_dir_path() / "segment",
#         vertex_position_array(mesh),
#         *loop_vertices_and_face_sizes(mesh),
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
#         *loop_vertices_and_face_sizes(mesh),
#         labels,
#         seed,
#         corner,
#     )


def split_on_borders(obj, border_edges):
    mesh = obj.data
    parts = part_triangles(mesh)
    labels = parts.vertex_labels
    corner_parts = labels[loop_vertices_and_face_sizes(mesh)[0]]

    bm = new_bmesh(obj)
    bm.verts.ensure_lookup_table()
    edges = [bm.edges.get((bm.verts[a], bm.verts[b])) for a, b in border_edges]
    border_vertices = {vertex for edge in edges for vertex in edge.verts}
    border_positions = numpy.array([tuple(vertex.co) for vertex in border_vertices])
    bmesh.ops.split_edges(bm, edges=edges)
    set_bmesh(bm, obj)

    vertex_labels = numpy.zeros(len(mesh.vertices), dtype=labels.dtype)
    vertex_labels[: len(labels)] = labels
    vertex_labels[loop_vertices_and_face_sizes(mesh)[0]] = corner_parts
    return border_positions, parts._replace(vertex_labels=vertex_labels)


WELD_TOLERANCE = 1e-5


class SegmentWeld:
    def __init__(self, border_positions, matrix):
        rotation_scale = numpy.array(matrix.to_3x3())
        self.border_positions = border_positions @ rotation_scale.T

    def finish(self, output):
        positions = vertex_position_array(output.data)
        tolerance = WELD_TOLERANCE * bounding_box_diagonal(positions)
        tree = KDTree(len(self.border_positions))
        for index, position in enumerate(self.border_positions.tolist()):
            tree.insert(position, index)
        tree.balance()

        bm = new_bmesh(output)
        near = [vertex for vertex in bm.verts if tree.find(vertex.co)[2] <= tolerance]
        bmesh.ops.remove_doubles(bm, verts=near, dist=tolerance)
        set_bmesh(bm, output)
