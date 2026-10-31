import collections

import bmesh
import bpy
import mathutils
import numpy

from .no_bpy.mesh import bounding_box_diagonal, transform_positions
from .no_bpy.mirror import (
    mirrored_uvs,
    mutual_mirror_partners,
    partner_map,
    symmetric_axis,
    symmetric_regions,
)
from .ops.guides import painted_weight_by_vertex
from .part_overrides import read_overrides
from .utils.mesh import (
    loop_uvs,
    loop_vertices_and_face_sizes,
    mesh_arrays,
    new_bmesh,
    set_bmesh,
    set_loop_uvs,
    vertex_position_array,
)
from .utils.paths import get_extension_dir_path
from .utils.uvgami_tools import tools_command

# bisecting just past a vertex near the plane leaves needle faces
PLANE_SNAP = 0.0001


def cut_on_axis(obj, center, axis, snap, kept_side):
    bm = new_bmesh(obj)
    index = "XYZ".index(axis)
    for vert in bm.verts:
        if abs(vert.co[index] - center[index]) < snap:
            vert.co[index] = center[index]
    direction = [0, 0, 0]
    direction[index] = kept_side
    # snapped float32 vertices miss the plane by rounding and get a duplicate at dist 0
    bmesh.ops.bisect_plane(
        bm,
        geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
        plane_co=center,
        plane_no=direction,
        dist=snap,
        clear_inner=True,
    )
    set_bmesh(bm, obj)


def bounding_box_center_and_diagonal(mesh):
    positions = vertex_position_array(mesh)
    center = (positions.min(axis=0) + positions.max(axis=0)) / 2
    return center, bounding_box_diagonal(positions)


def mirror_workdir():
    return get_extension_dir_path() / "mirror"


def mesh_symmetric_axis(mesh, center, tolerance):
    return symmetric_axis(
        tools_command(),
        mirror_workdir(),
        *mesh_arrays(mesh),
        center,
        tolerance,
    )


def mesh_mirror_partners(mesh, center, axis):
    return mutual_mirror_partners(
        tools_command(), mirror_workdir(), vertex_position_array(mesh), center, axis
    )


MirrorPlan = collections.namedtuple("MirrorPlan", "axis whole partners")


def mirror_plan(mesh, center, tolerance):
    # if props.symmetry == "AXES":
    #     return sorted(props.sym_axes)
    if any(any(values) for values in read_overrides(mesh).values()):
        return None
    # optcuts welds only one plane
    found = mesh_symmetric_axis(mesh, center, tolerance)
    if found is None:
        return None
    whole = not found.exact or found.crosses
    return MirrorPlan(found.axis, whole, found.partners)


def paints_seams(obj, engine, props):
    return (
        engine.supports_guided
        and props.avoid_seams
        and any(weight > 0 for weight in painted_weight_by_vertex(obj).values())
    )


SymmetryPlan = collections.namedtuple(
    "SymmetryPlan", "kind axis center tolerance partners kept_side"
)


def symmetry_plan(engine, mesh, painted):
    center, diagonal = bounding_box_center_and_diagonal(mesh)
    tolerance = PLANE_SNAP * diagonal
    found = mirror_plan(mesh, center, tolerance)
    if found is None:
        return None
    if not engine.keeps_plane_vertices_on_u_zero:
        if found.whole:
            return None
        return SymmetryPlan("MIRROR_UVS", found.axis, center, tolerance, None, 1)
    if not found.whole and not painted:
        return SymmetryPlan("BISECT", found.axis, center, tolerance, None, 1)
    partners = partner_map(found.partners)
    return SymmetryPlan("WHOLE", found.axis, center, tolerance, partners, 1)


def mirror_segmentation(positions, corners, totals, segmentation, symmetry):
    mirrored = symmetry is not None and symmetry.kind in ("BISECT", "WHOLE")
    if not mirrored or not segmentation.border_edges:
        return symmetry, segmentation
    regions = symmetric_regions(
        tools_command(),
        mirror_workdir(),
        positions,
        corners,
        totals,
        segmentation.labels,
        symmetry.axis,
        symmetry.center,
        symmetry.tolerance,
    )
    segmentation = segmentation._replace(
        border_edges=regions.border_edges,
        region_count=regions.region_count,
        labels=regions.labels,
    )
    return symmetry._replace(kept_side=regions.kept_side), segmentation


class Symmetrise:
    def __init__(
        self,
        symmetry,
        # overlap,
    ):
        self.axis = symmetry.axis
        self.center = mathutils.Vector(symmetry.center)
        # self.overlap = overlap
        self.tolerance = symmetry.tolerance
        self.kept_side = symmetry.kept_side

    def cut(self, obj):
        cut_on_axis(obj, self.center, self.axis, self.tolerance, self.kept_side)

    def with_mirrored(self, positions):
        index = "XYZ".index(self.axis)
        flipped = positions.copy()
        flipped[:, index] = 2 * self.center[index] - flipped[:, index]
        return numpy.concatenate((positions, flipped))

    def write_plane_sidecar(self, path, mesh, *, keep_charts):
        index = "XYZ".index(self.axis)
        distance = numpy.abs(vertex_position_array(mesh)[:, index] - self.center[index])
        on_plane = numpy.flatnonzero(distance <= self.tolerance).tolist()
        if not on_plane:
            return
        sidecar = path.parent / f"{path.stem}_plane"
        # _plane files list the vertices optcuts glues on the first line
        lines = ",".join(map(str, on_plane)) + "\n"
        # "keep" marks a preseed whose charts are the seams
        if keep_charts:
            lines += "keep\n"
        sidecar.write_text(lines)

    def finish(self, output, matrix):
        mesh = output.data
        has_uvs = mesh.uv_layers.active is not None
        if has_uvs:
            # Mirror modifiers flip u about 0.5
            coords = loop_uvs(mesh)
            coords[:, 0] += 0.5
            set_loop_uvs(mesh, coords)

        # optcuts output has the plane vertices on one uv line
        mirror = output.modifiers.new("Mirror", "MIRROR")
        mirror.use_axis = tuple(axis == self.axis for axis in "XYZ")
        # obj exports bake the matrix's rotation and scale into the vertices
        mirror.merge_threshold = self.tolerance * max(map(abs, matrix.to_scale()))
        empty = bpy.data.objects.new("Empty", None)
        empty.matrix_world = matrix @ mathutils.Matrix.Translation(self.center)
        mirror.mirror_object = empty

        # if not self.overlap:
        mirror.use_mirror_u = True
        mirror.use_mirror_v = False

        old_active = bpy.context.view_layer.objects.active
        bpy.context.view_layer.objects.active = output
        bpy.ops.object.modifier_apply(modifier=mirror.name)
        bpy.context.view_layer.objects.active = old_active
        bpy.data.objects.remove(empty, do_unlink=True)

        # optcuts plane output is unnormalized
        if has_uvs:
            coords = loop_uvs(mesh)
            low = coords.min(axis=0)
            size = (coords.max(axis=0) - low).max()
            if size > 0:
                set_loop_uvs(mesh, (coords - low) / size)


class MirrorUVs:
    def __init__(self, symmetry):
        self.axis = symmetry.axis
        self.center = symmetry.center
        self.tolerance = symmetry.tolerance

    def cut(self, obj):
        pass

    def with_mirrored(self, positions):
        return positions

    def write_plane_sidecar(self, path, mesh, *, keep_charts):
        pass

    def finish(self, output, matrix):
        mesh = output.data
        corners, totals = loop_vertices_and_face_sizes(mesh)
        # obj exports bake the matrix's rotation and scale into the vertices
        local = matrix.to_3x3().inverted().to_4x4()
        coords = mirrored_uvs(
            tools_command(),
            mirror_workdir(),
            transform_positions(vertex_position_array(mesh), local),
            corners,
            totals,
            loop_uvs(mesh),
            self.center,
            self.axis,
            self.tolerance,
        )
        set_loop_uvs(mesh, coords)
