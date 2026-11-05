import collections

import bmesh
import mathutils
import numpy

from .guides import painted_weight_by_vertex
from .no_bpy.mesh import bounding_box_diagonal, face_starts, transform_positions
from .no_bpy.proxy import TransferMirror
from .no_bpy.symmetry import (
    FaceHasNoMirror,
    FaceSpansCharts,
    faces_mirror,
    mirrored_uvs,
    mutual_mirror_partners,
    partner_map,
    start_symmetric_axis,
    symmetric_regions,
)
from .part_overrides import read_overrides
from .unwrap.logger import logger
from .utils.mesh import (
    loop_uvs,
    loop_vertices_and_face_sizes,
    mesh_arrays,
    new_bmesh,
    set_bmesh,
    set_loop_uvs,
    symmetric_faces,
    vertex_position_array,
)
from .utils.paths import get_extension_dir_path
from .utils.uvgami_tools import tools_command

# bisecting just past a vertex near the plane leaves needle faces
PLANE_SNAP = 0.0001


def part_geometry(bm, faces):
    bm.faces.ensure_lookup_table()
    chosen = [bm.faces[index] for index in numpy.flatnonzero(faces).tolist()]
    verts = {vert for face in chosen for vert in face.verts}
    edges = {edge for face in chosen for edge in face.edges}
    return [*verts, *edges, *chosen]


def geometry_verts(geom):
    return [element for element in geom if isinstance(element, bmesh.types.BMVert)]


def mirror_geometry(bm, geom, axis, to_plane, merge_distance, *, mirror_u=False):
    verts = geometry_verts(geom)
    # bmesh.ops.mirror welds only across the axis plane through the origin
    bmesh.ops.transform(bm, matrix=to_plane, verts=verts)
    old_verts = set(bm.verts)
    old_faces = set(bm.faces)
    bmesh.ops.mirror(
        bm,
        geom=geom,
        matrix=mathutils.Matrix.Identity(4),
        merge_dist=merge_distance,
        axis=axis,
        mirror_u=mirror_u,
    )
    new_verts = [vert for vert in bm.verts if vert not in old_verts]
    # bmesh.ops.mirror copies keep their winding, so their normals point inward
    bmesh.ops.reverse_faces(
        bm, faces=[face for face in bm.faces if face not in old_faces]
    )
    bmesh.ops.transform(bm, matrix=to_plane.inverted(), verts=verts + new_verts)


def cut_on_axis(obj, center, axis, snap, kept_side, faces):
    bm = new_bmesh(obj)
    index = "XYZ".index(axis)
    geom = part_geometry(bm, faces)
    for vert in geometry_verts(geom):
        if abs(vert.co[index] - center[index]) < snap:
            vert.co[index] = center[index]
    direction = [0, 0, 0]
    direction[index] = kept_side
    # snapped float32 vertices miss the plane by rounding and get a duplicate at dist 0
    bmesh.ops.bisect_plane(
        bm,
        geom=geom,
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


def start_mesh_symmetric_axis(mesh, center, tolerance):
    return start_symmetric_axis(
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


MirrorPlan = collections.namedtuple("MirrorPlan", "axis whole partners symmetric")


def paints_seams(obj, engine, props):
    return (
        engine.supports_guided
        and props.avoid_seams
        and any(weight > 0 for weight in painted_weight_by_vertex(obj).values())
    )


SymmetryPlan = collections.namedtuple(
    "SymmetryPlan", "kind axis center tolerance partners kept_side symmetric_faces"
)


class SymmetryCheck:
    def __init__(self, engine, mesh, painted):
        self.engine = engine
        self.painted = painted
        self.center, diagonal = bounding_box_center_and_diagonal(mesh)
        self.tolerance = PLANE_SNAP * diagonal
        self.corners, self.totals = loop_vertices_and_face_sizes(mesh)
        # optcuts welds only one plane
        self.run = start_mesh_symmetric_axis(mesh, self.center, self.tolerance)

    def done(self):
        return self.run.done()

    def cancel(self):
        self.run.close()

    def close(self):
        self.run.close()

    def mirror_plan(self):
        found = self.run.result()
        if found is None:
            return None
        whole = not found.exact or found.crosses
        return MirrorPlan(found.axis, whole, found.partners, found.symmetric)

    def result(self):
        found = self.mirror_plan()
        if found is None:
            return None
        center, tolerance = self.center, self.tolerance
        faces = found.symmetric[self.corners[face_starts(self.totals)]]
        if not self.engine.keeps_plane_vertices_on_u_zero:
            if found.whole:
                return None
            return SymmetryPlan(
                "MIRROR_UVS", found.axis, center, tolerance, None, 1, faces
            )
        bisects = (
            not found.whole
            and not self.painted
            and faces_mirror(self.corners, self.totals, found.partners, faces)
        )
        if bisects:
            return SymmetryPlan("BISECT", found.axis, center, tolerance, None, 1, faces)
        partners = partner_map(found.partners)
        return SymmetryPlan("WHOLE", found.axis, center, tolerance, partners, 1, faces)


def start_symmetry_check(engine, mesh, painted):
    # if props.symmetry == "AXES":
    #     return sorted(props.sym_axes)
    if any(any(values) for values in read_overrides(mesh).values()):
        return None
    return SymmetryCheck(engine, mesh, painted)


def mirror_segmentation(positions, corners, totals, segmentation, symmetry):
    mirrored = symmetry is not None and symmetry.kind in ("BISECT", "WHOLE")
    if not mirrored or not segmentation.border_edges or segmentation.hand_edited:
        return symmetry, segmentation

    def mirrored_regions(labels):
        return symmetric_regions(
            tools_command(),
            mirror_workdir(),
            positions,
            corners,
            totals,
            labels,
            symmetry.axis,
            symmetry.center,
            symmetry.tolerance,
            symmetry.symmetric_faces,
        )

    regions = mirrored_regions(segmentation.labels)
    segmentation = segmentation._replace(
        border_edges=regions.border_edges,
        region_count=regions.region_count,
        labels=regions.labels,
    )
    if segmentation.fine_labels is not None:
        fine = mirrored_regions(segmentation.fine_labels)
        segmentation = segmentation._replace(fine_labels=fine.labels)
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
        cut_on_axis(
            obj,
            self.center,
            self.axis,
            self.tolerance,
            self.kept_side,
            symmetric_faces(obj.data),
        )

    def transfer_mirror(self, mesh):
        index = "XYZ".index(self.axis)
        found = mesh_mirror_partners(mesh, self.center, self.axis)
        partners = numpy.full(len(mesh.vertices), -1, dtype=numpy.int64)
        partners[list(found)] = list(found.values())
        offsets = vertex_position_array(mesh)[:, index] - self.center[index]
        return TransferMirror(partners, offsets * self.kept_side < -self.tolerance)

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

    def finish(self, output, matrix, symmetric):
        mesh = output.data
        has_uvs = mesh.uv_layers.active is not None
        if has_uvs:
            # bmesh.ops.mirror flips u about 0.5
            coords = loop_uvs(mesh)
            coords[:, 0] += 0.5
            set_loop_uvs(mesh, coords)

        corners, totals = loop_vertices_and_face_sizes(mesh)
        bm = new_bmesh(output)
        # obj exports bake the matrix's rotation and scale into the vertices
        to_plane = (
            mathutils.Matrix.Translation(-self.center)
            @ matrix.to_3x3().inverted().to_4x4()
        )
        # optcuts output has the plane vertices on one uv line
        # if not self.overlap:
        mirror_geometry(
            bm,
            part_geometry(bm, symmetric[corners[face_starts(totals)]]),
            self.axis,
            to_plane,
            self.tolerance,
            mirror_u=True,
        )
        set_bmesh(bm, output)

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

    def transfer_mirror(self, mesh):
        return None

    def with_mirrored(self, positions):
        return positions

    def write_plane_sidecar(self, path, mesh, *, keep_charts):
        pass

    def finish(self, output, matrix, symmetric):
        mesh = output.data
        corners, totals = loop_vertices_and_face_sizes(mesh)
        # obj exports bake the matrix's rotation and scale into the vertices
        local = matrix.to_3x3().inverted().to_4x4()
        try:
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
                symmetric[corners[face_starts(totals)]],
            )
        except (FaceSpansCharts, FaceHasNoMirror):
            logger.add_error(
                f"{output.name}: symmetry skipped, its faces don't mirror across"
                f" the {self.axis} axis"
            )
            return
        set_loop_uvs(mesh, coords)
