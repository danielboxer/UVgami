import collections
from collections import namedtuple
from enum import Enum

import bmesh
import bpy
import mathutils
import numpy

from .hard_surface import (
    apply_interior_seams,
    apply_loop_uvs,
    apply_seams,
    apply_seams_except_faces,
)
from .logger import logger
from .objfile import merge_obj_files
from .ops.guides import seam_restriction_weights
from .part_overrides import read_overrides
from .proxy import transfer_inputs
from .seams import uv_area_fit
from .seams.mesh import transformed
from .seams.mirror import (
    mirrored_corner_sources,
    partner_map,
    symmetric_axis,
    symmetric_regions,
)
from .seams.proxy_transfer import TransferRun, dense_subset
from .seams.segmentation import border_edges, edge_face_pairs
from .seams.uv_transfer import transfer_exact_job
from .seams.worker import WorkerProcess
from .utils.geometry import PLANE_SNAP, cut_on_axis
from .utils.mesh import (
    check_exists,
    face_vertex_arrays,
    in_object_mode,
    loop_uvs,
    new_bmesh,
    set_bmesh,
    set_loop_uvs,
    triangulate_mesh,
    vertex_position_array,
    world_positions,
)
from .utils.paths import get_extension_dir_path
from .utils.uvgami_tools import tools_command

REMOVE_EDGE_ERROR = "Error removing added edge"

TransferReport = namedtuple(
    "TransferReport", ["applied", "split_count", "detail", "reason"], defaults=("",)
)


class Result(Enum):
    FINISHED = "finished"
    INVALID = "invalid"
    CANCELLED = "cancelled"
    STOPPED = "stopped"


# in the flat form transfer_exact_job takes
def output_mesh_data(output):
    output_data = output.data
    coords = numpy.empty(len(output_data.loops) * 2)
    output_data.uv_layers.active.uv.foreach_get("vector", coords)
    return (
        world_positions(output_data, output.matrix_world),
        *face_vertex_arrays(output_data),
        coords.reshape(-1, 2),
    )


# a corner uv that differs between the edge's two faces means a seam runs along it
def _carries_seam(edge, uv_layer):
    faces = edge.link_faces
    if len(faces) != 2:
        return True
    for vert in edge.verts:
        uvs = {
            tuple(loop[uv_layer].uv)
            for face in faces
            for loop in face.loops
            if loop.vert is vert
        }
        if len(uvs) > 1:
            return True
    return False


class Preserve:
    def finish(self, output, added_edges):
        bm = new_bmesh(output)
        uv_layer = bm.loops.layers.uv.active
        edges = {(edge.verts[0].index, edge.verts[1].index): edge for edge in bm.edges}

        dissolve_edges = []
        for a, b in added_edges:
            edge = edges.get((a, b)) or edges.get((b, a))
            if edge is None:
                errors = logger.get_latest().errors
                if not errors or errors[-1] != REMOVE_EDGE_ERROR:
                    logger.add_data("errors", REMOVE_EDGE_ERROR)
                continue
            # the engine locks these edges, so a seam here comes from an input map
            if not _carries_seam(edge, uv_layer):
                dissolve_edges.append(edge)

        bmesh.ops.dissolve_edges(bm, edges=dissolve_edges)
        set_bmesh(bm, output)


class Join:
    _last_id = 0

    def __init__(self, expected):
        # a member's stem goes stale as soon as that piece settles
        Join._last_id += 1
        self.job_id = Join._last_id
        self.expected = expected
        # every piece in creation order, settled or not
        self.members = []
        # the merge concatenates in creation order, not completion order
        self.finished = []
        self.reported = 0
        self.discard = False
        self.is_expanded = False

    # finished is read only once the group is settled
    def record(self, unwrap, result):
        self.reported += 1
        if result is Result.FINISHED:
            self.finished.append(unwrap)
        if self.is_settled():
            position = {member: index for index, member in enumerate(self.members)}
            self.finished.sort(key=position.__getitem__)

    def is_settled(self):
        return self.reported == self.expected

    def finish(self):
        unwraps = self.finished

        # the merge writes into the first obj, and that is what gets imported
        path = merge_obj_files([u.output_path for u in unwraps])

        added_edges = []
        offset = 0
        for unwrap in unwraps:
            added_edges.extend((a + offset, b + offset) for a, b in unwrap.added_edges)
            offset += unwrap.vertex_count

        return path, added_edges


class HideInput:
    def __init__(self, input_mesh):
        self.input_mesh = input_mesh

    def finish(self):
        if check_exists(self.input_mesh):
            self.input_mesh.hide_set(True)


# the read is bpy-free and runs in another process
class Transfer:
    # the manager repacks the input in place of the deleted output
    repack_input = True
    allows_missing_pieces = True
    # a copy of the input the result went onto, in place of the output
    replacement = None

    def __init__(self, input_mesh):
        self.input_mesh = input_mesh
        self.output = None
        self.target = None
        self.task = None
        self.progress = 0.0
        self.loop_count = 0
        self.settled = False

    # None means poll() finishes it, a report means it failed before starting
    def start(self, output):
        if not check_exists(self.input_mesh) or not check_exists(output):
            return TransferReport(False, 0, "input or output object missing")
        if output.data.uv_layers.active is None:
            return TransferReport(False, 0, "output mesh has no uv layer")
        self.output = output
        self.target = self._target(self.input_mesh)
        inputs = in_object_mode(self.target, self._extract, self.target, output)
        self.loop_count = len(self.target.data.loops)
        self._begin(inputs)
        return None

    # None while the worker runs, the final report once it is done
    def poll(self):
        if not self._work_done():
            return None
        result = self.task.result()
        if not check_exists(self.target) or not check_exists(self.output):
            return self._fail("input or output object missing")
        if len(self.target.data.loops) != self.loop_count:
            # an undo during the run replaced the mesh
            return self._fail("mesh changed during the unwrap")
        failure = self._failure(result)
        if failure is not None:
            return self._fail(*failure)
        split_count = in_object_mode(self.target, self._apply, self.target, result)
        # delete the output only once the whole result applied
        bpy.data.objects.remove(self.output, do_unlink=True)
        self._show()
        return TransferReport(True, split_count, "")

    # the output only makes sense with its uvs applied, so it goes too
    def cancel(self):
        self.settled = True
        if self.task is not None:
            self.task.close()
        self._discard()
        if check_exists(self.output):
            bpy.data.objects.remove(self.output, do_unlink=True)

    def _work_done(self):
        return self.task.done()

    def _fail(self, detail, reason=""):
        self._discard()
        return TransferReport(False, 0, detail, reason)

    def _target(self, input_mesh):
        return input_mesh

    # (detail, reason) when the read came back with nothing to apply
    def _failure(self, result):
        return None

    # drops what start() made besides the worker
    def _discard(self):
        pass

    def _show(self):
        self.input_mesh.hide_set(False)


# the output has the input's own vertices, its unwrap triangulated
class TransferUVs(Transfer):
    def _extract(self, target, output):
        return (
            world_positions(target.data, target.matrix_world),
            *face_vertex_arrays(target.data),
            *output_mesh_data(output),
        )

    def _begin(self, inputs):
        self.task = WorkerProcess(
            transfer_exact_job,
            *inputs,
            repack=self.repack_input,
            partial=self.allows_missing_pieces,
        )

    def _failure(self, plan):
        if plan.ok:
            return None
        return f"{plan.reason}: {plan.detail}", plan.reason

    def _apply(self, target, plan):
        if plan.split_faces:
            self._apply_with_splits(target, plan)
            return len(plan.split_faces)

        data = target.data
        if not data.uv_layers:
            data.uv_layers.new(name="UVMap")

        coords = numpy.empty((len(data.loops), 2))
        data.uv_layers.active.data.foreach_get("uv", coords.ravel())
        coords[list(plan.loop_uvs)] = list(plan.loop_uvs.values())
        set_loop_uvs(data, coords)

        apply_seams_except_faces(data, plan.untouched_faces, plan.seam_edges)
        data.update()
        return 0

    # like _apply, but rebuilds the faces a uv cut runs through
    def _apply_with_splits(self, input_mesh, plan):
        bm = new_bmesh(input_mesh)
        uv_layer = bm.loops.layers.uv.verify()

        loop_idx = 0
        to_split = []
        kept_edges = set()
        for face_idx, face in enumerate(bm.faces):
            parts = plan.split_faces.get(face_idx)
            if face_idx in plan.untouched_faces:
                kept_edges.update(face.edges)
                loop_idx += len(face.loops)
            elif parts is None:
                for loop in face.loops:
                    loop[uv_layer].uv = plan.loop_uvs[loop_idx]
                    loop_idx += 1
            else:
                loop_idx += len(face.loops)
                to_split.append((face, parts))

        _rebuild_split_faces(bm, uv_layer, to_split)
        for edge in bm.edges:
            if edge in kept_edges:
                continue
            a, b = edge.verts[0].index, edge.verts[1].index
            edge.seam = ((a, b) if a < b else (b, a)) in plan.seam_edges

        set_bmesh(bm, input_mesh)


# parts are (mesh vertex indices, uvs) pairs, each one a new face
def _rebuild_split_faces(bm, uv_layer, to_split):
    pieces = [(parts, face.material_index, face.smooth) for face, parts in to_split]
    # delete first so a new piece can never collide with the face it replaces
    bmesh.ops.delete(bm, geom=[face for face, _ in to_split], context="FACES_ONLY")
    bm.verts.ensure_lookup_table()
    for parts, material_index, smooth in pieces:
        for verts, uvs in parts:
            new_face = bm.faces.new([bm.verts[v] for v in verts])
            new_face.material_index = material_index
            new_face.smooth = smooth
            for loop, uv in zip(new_face.loops, uvs):
                loop[uv_layer].uv = uv


def _move_plan_uvs(plan, move):
    for k in plan.loop_uvs:
        plan.loop_uvs[k] = move(plan.loop_uvs[k])
    for fi, parts in plan.split_faces.items():
        plan.split_faces[fi] = [
            (verts, [move(uv) for uv in part_uvs]) for verts, part_uvs in parts
        ]


# the edges with one of these faces on each side, as (low, high) pairs
def _interior_edges(data, faces):
    owner_count = {}
    for fi in faces:
        poly = data.polygons[fi].vertices
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            key = (a, b) if a < b else (b, a)
            owner_count[key] = owner_count.get(key, 0) + 1
    return {key for key, count in owner_count.items() if count == 2}


# TransferUVs' position matching on the island's faces only
class IslandUVs(TransferUVs):
    repack_input = False
    allows_missing_pieces = False

    def __init__(self, input_mesh, faces, bbox, area, mirrored=False):
        super().__init__(input_mesh)
        self.faces = faces
        self.bbox = bbox
        self.area = area
        self.mirrored = mirrored
        self.orig_vert = []
        self.loop_base = {}
        self.loop_counts = []

    # a mirrored island was exported with u negated
    def _unmirror(self, plan):
        if self.mirrored:
            _move_plan_uvs(plan, lambda uv: (-uv[0], uv[1]))

    def _extract(self, target, output):
        data = target.data
        matrix = target.matrix_world

        used = sorted({v for fi in self.faces for v in data.polygons[fi].vertices})
        self.orig_vert = used
        local = {v: i for i, v in enumerate(used)}

        positions = [tuple(matrix @ data.vertices[v].co) for v in used]
        corners = []
        self.loop_counts = []
        base = 0
        for fi in self.faces:
            poly = data.polygons[fi]
            self.loop_base[fi] = base
            self.loop_counts.append(poly.loop_total)
            base += poly.loop_total
            corners.extend(local[v] for v in poly.vertices)

        return (
            positions,
            numpy.array(corners, dtype=numpy.int64),
            numpy.array(self.loop_counts, dtype=numpy.int64),
            *output_mesh_data(output),
        )

    # a split face's uvs come from its parts, not its old loops
    def _fit(self, plan):
        self._unmirror(plan)
        polygons = []
        for i, count in enumerate(self.loop_counts):
            parts = plan.split_faces.get(i)
            if parts is None:
                base = self.loop_base[self.faces[i]]
                polygons.append([plan.loop_uvs[base + c] for c in range(count)])
            else:
                polygons.extend(part_uvs for _, part_uvs in parts)
        _move_plan_uvs(plan, uv_area_fit(polygons, self.area, self.bbox))

    def _apply(self, target, plan):
        self._fit(plan)
        data = target.data
        ov = self.orig_vert
        seams = {
            ((ov[a], ov[b]) if ov[a] < ov[b] else (ov[b], ov[a]))
            for a, b in plan.seam_edges
        }

        # the island boundary and the rest of the mesh keep their marks
        interior = _interior_edges(data, self.faces)

        split_faces = {self.faces[fi]: parts for fi, parts in plan.split_faces.items()}
        if split_faces:
            self._apply_island_splits(target, plan, split_faces, seams, interior)
            return len(plan.split_faces)

        uvs = [
            plan.loop_uvs[self.loop_base[fi] + c]
            for i, fi in enumerate(self.faces)
            for c in range(self.loop_counts[i])
        ]
        apply_loop_uvs(data, numpy.array(uvs), self.faces)

        apply_interior_seams(data, interior, seams)
        data.update()
        return 0

    # like _apply, but rebuilds the island faces a uv cut runs through
    def _apply_island_splits(self, input_mesh, plan, split_faces, seams, interior):
        bm = new_bmesh(input_mesh)
        uv_layer = bm.loops.layers.uv.verify()
        bm.faces.ensure_lookup_table()

        ov = self.orig_vert
        to_split = []
        for fi in self.faces:
            face = bm.faces[fi]
            parts = split_faces.get(fi)
            if parts is None:
                base = self.loop_base[fi]
                for c, loop in enumerate(face.loops):
                    loop[uv_layer].uv = plan.loop_uvs[base + c]
            else:
                parts = [([ov[v] for v in verts], uvs) for verts, uvs in parts]
                to_split.append((face, parts))

        _rebuild_split_faces(bm, uv_layer, to_split)
        for edge in bm.edges:
            a, b = edge.verts[0].index, edge.verts[1].index
            key = (a, b) if a < b else (b, a)
            if key in seams:
                edge.seam = True
            elif key in interior:
                edge.seam = False

        set_bmesh(bm, input_mesh)


# the engine output is scaled into the unit box
class AreaUVs(IslandUVs):
    def __init__(self, input_mesh, faces, pins, mirrored):
        super().__init__(input_mesh, faces, None, None, mirrored)
        self.pins = pins  # (face index, corner, old uv)

    def _fit(self, plan):
        self._unmirror(plan)
        pairs = []
        for fi, corner, old in self.pins:
            new = plan.loop_uvs.get(self.loop_base[fi] + corner)
            if new is not None:
                pairs.append((old, new))
        if len(pairs) < 2:
            return

        lo = min(pairs, key=lambda p: p[0][0])
        hi = max(pairs, key=lambda p: p[0][0])
        if hi[1][0] == lo[1][0]:
            lo = min(pairs, key=lambda p: p[0][1])
            hi = max(pairs, key=lambda p: p[0][1])
            scale = (hi[0][1] - lo[0][1]) / (hi[1][1] - lo[1][1])
        else:
            scale = (hi[0][0] - lo[0][0]) / (hi[1][0] - lo[1][0])
        du = lo[0][0] - scale * lo[1][0]
        dv = lo[0][1] - scale * lo[1][1]

        _move_plan_uvs(plan, lambda uv: (scale * uv[0] + du, scale * uv[1] + dv))
        # pinned loops drift by float noise
        for fi, corner, old in self.pins:
            k = self.loop_base[fi] + corner
            if k in plan.loop_uvs:
                plan.loop_uvs[k] = old


# the original was never unwrapped, only its proxy
class ProxyUVs(Transfer):
    # a missing piece is a hole in the proxy map
    allows_missing_pieces = False

    def _extract(self, target, output):
        return transfer_inputs(target, output)

    def _begin(self, inputs):
        self.task = TransferRun(
            tools_command(), get_extension_dir_path() / "proxy", *inputs
        )

    def _work_done(self):
        self.progress = self.task.progress
        return self.task.done()

    def _apply(self, target, result):
        seams, uvs = result
        data = target.data
        apply_seams(data, seams)
        if not data.uv_layers:
            data.uv_layers.new()
        set_loop_uvs(data, uvs)
        return 0


# the map goes onto a triangulated duplicate that stands in for the deleted output
class ProxyCopyUVs(ProxyUVs):
    repack_input = False

    def _target(self, input_mesh):
        # never linked to a collection, or the copy shows up beside the original
        target = input_mesh.copy()
        target.data = input_mesh.data.copy()
        triangulate_mesh(target.data)
        return target

    def _show(self):
        # no HideInput job exists when a transfer job is in the slot
        self.input_mesh.hide_set(True)
        # renamed only now, the deleted output held this name
        self.target.name = f"{self.input_mesh.name}_unwrapped"
        self.replacement = self.target

    def _discard(self):
        if check_exists(self.target):
            bpy.data.objects.remove(self.target, do_unlink=True)


# the island is read at full density to take the proxy's cuts
class ProxyIslandUVs(ProxyUVs):
    repack_input = False

    def __init__(self, input_mesh, faces, bbox, area):
        super().__init__(input_mesh)
        self.faces = faces
        self.bbox = bbox
        self.area = area
        self.orig_vert = []

    def _extract(self, target, output):
        dense, proxy = transfer_inputs(target, output)
        dense, self.orig_vert = dense_subset(dense, self.faces)
        return dense, proxy

    def _apply(self, target, result):
        seams, uvs = result
        data = target.data
        sizes = [data.polygons[fi].loop_total for fi in self.faces]
        polygons = [
            part.tolist() for part in numpy.split(uvs, numpy.cumsum(sizes)[:-1])
        ]
        move = uv_area_fit(polygons, self.area, self.bbox)
        apply_loop_uvs(
            data,
            numpy.array([move(uv) for points in polygons for uv in points]),
            self.faces,
        )

        ov = self.orig_vert.tolist()
        seams = {
            ((ov[a], ov[b]) if ov[a] < ov[b] else (ov[b], ov[a])) for a, b in seams
        }
        # the island boundary and the rest of the mesh keep their marks
        apply_interior_seams(data, _interior_edges(data, self.faces), seams)
        data.update()
        return 0


# the cut plane goes through the bounding box center, returns it and the diagonal
def symmetry_center(mesh):
    positions = vertex_position_array(mesh)
    low = positions.min(axis=0)
    high = positions.max(axis=0)
    return (low + high) / 2, float(numpy.linalg.norm(high - low))


def mirror_workdir():
    return get_extension_dir_path() / "mirror"


# whole when drifted off an exact mirror or with faces across the plane
# partners has each vertex's mirror vertex, -1 where there is none
MirrorPlan = collections.namedtuple("MirrorPlan", "axis whole partners")


# under Auto a mesh that doesn't mirror onto itself gets no plan
def mirror_plan(mesh, center, tolerance):
    # if props.symmetry == "AXES":
    #     return sorted(props.sym_axes)
    # the mirrored half takes the other half's uvs and overrides
    if any(any(values) for values in read_overrides(mesh).values()):
        return None
    # one axis, since the engine welds one plane
    found = symmetric_axis(
        tools_command(),
        mirror_workdir(),
        vertex_position_array(mesh),
        *face_vertex_arrays(mesh),
        center,
        tolerance,
    )
    if found is None:
        return None
    whole = not found.exact or found.crosses
    return MirrorPlan(found.axis, whole, found.partners)


# the bisect would put the plane seam on the paint, so the mesh mirrors whole
def paints_seams(obj, engine, props):
    return (
        engine.supports_guided
        and props.avoid_seams
        and any(weight > 0 for weight in seam_restriction_weights(obj).values())
    )


# kind is BISECT, WHOLE or MIRROR_UVS, partners is set for WHOLE only, kept_side is cut_on_axis's
SymmetryPlan = collections.namedtuple(
    "SymmetryPlan", "kind axis center tolerance partners kept_side"
)
UnwrapPlan = collections.namedtuple("UnwrapPlan", "symmetry splits_regions")


# mesh is the object's mesh before its transform, the plan is in that space
def unwrap_plan(engine, props, mesh, *, painted):
    symmetry = None
    if engine.uses_symmetry(props) and len(mesh.vertices) > 0:
        symmetry = _symmetry_plan(engine, mesh, painted)
    return UnwrapPlan(symmetry, engine.splits_regions(props))


def _symmetry_plan(engine, mesh, painted):
    center, diagonal = symmetry_center(mesh)
    tolerance = PLANE_SNAP * diagonal
    found = mirror_plan(mesh, center, tolerance)
    if found is None:
        return None
    if not engine.supports_plane_glue:
        # without the plane glue a mesh with faces across the plane can't get a seam on it
        if found.whole:
            return None
        return SymmetryPlan("MIRROR_UVS", found.axis, center, tolerance, None, 1)
    if not found.whole and not painted:
        return SymmetryPlan("BISECT", found.axis, center, tolerance, None, 1)
    partners = partner_map(found.partners)
    return SymmetryPlan("WHOLE", found.axis, center, tolerance, partners, 1)


# the segmentation isn't mirrored, so the kept half's regions are copied onto the other
def mirror_segmentation(positions, corners, totals, segmentation, symmetry):
    mirrored = symmetry is not None and symmetry.kind in ("BISECT", "WHOLE")
    if not mirrored or not segmentation.border_edges:
        return symmetry, segmentation
    kept_side, labels = symmetric_regions(
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
    face_pairs = edge_face_pairs(corners, totals)
    segmentation = segmentation._replace(
        border_edges=border_edges(face_pairs, labels),
        region_count=len(numpy.unique(labels[labels > 0])),
        labels=labels,
    )
    return symmetry._replace(kept_side=kept_side), segmentation


class Symmetrise:
    def __init__(
        self,
        axis,
        center,
        # overlap,
        tolerance,
        kept_side,
    ):
        self.axis = axis
        self.center = center
        # self.overlap = overlap
        # how far off the plane a vertex still counts as on it
        self.tolerance = tolerance
        self.kept_side = kept_side

    def cut(self, obj):
        cut_on_axis(obj, self.center, self.axis, self.tolerance, self.kept_side)

    # the mirrored half has its region borders at the kept half's mirror images
    def with_mirrored(self, positions):
        index = "XYZ".index(self.axis)
        flipped = positions.copy()
        flipped[:, index] = 2 * self.center[index] - flipped[:, index]
        return numpy.concatenate((positions, flipped))

    # the plane vertex indices the engine glues, and "keep" for a preseed whose charts are the seams
    def write_plane_sidecar(self, path, mesh, *, keep_charts):
        index = "XYZ".index(self.axis)
        distance = numpy.abs(vertex_position_array(mesh)[:, index] - self.center[index])
        on_plane = numpy.flatnonzero(distance <= self.tolerance).tolist()
        if not on_plane:
            return
        sidecar = path.parent / f"{path.stem}_plane"
        lines = ",".join(map(str, on_plane)) + "\n"
        if keep_charts:
            lines += "keep\n"
        sidecar.write_text(lines)

    # the plane vertices are on one uv line in the engine output and the mirror welds them
    def finish(self, output, matrix):
        mesh = output.data
        has_uvs = mesh.uv_layers.active is not None
        if has_uvs:
            # the modifier flips u about 0.5
            coords = loop_uvs(mesh)
            coords[:, 0] += 0.5
            set_loop_uvs(mesh, coords)

        mirror = output.modifiers.new("Mirror", "MIRROR")
        mirror.use_axis = tuple(axis == self.axis for axis in "XYZ")
        # the output vertices are the local ones moved by the export matrix's rotation and scale
        mirror.merge_threshold = self.tolerance * max(map(abs, matrix.to_scale()))
        empty = bpy.data.objects.new("Empty", None)
        empty.matrix_world = matrix @ mathutils.Matrix.Translation(self.center)
        mirror.mirror_object = empty

        # if not self.overlap:
        # the halves get separate islands
        mirror.use_mirror_u = True
        # a welded line keeps v shared across it
        mirror.use_mirror_v = False

        old_active = bpy.context.view_layer.objects.active
        bpy.context.view_layer.objects.active = output
        bpy.ops.object.modifier_apply(modifier=mirror.name)
        bpy.context.view_layer.objects.active = old_active
        bpy.data.objects.remove(empty, do_unlink=True)

        # the plane output is unnormalized so its line could sit at u = 0
        if has_uvs:
            coords = loop_uvs(mesh)
            low = coords.min(axis=0)
            size = (coords.max(axis=0) - low).max()
            if size > 0:
                set_loop_uvs(mesh, (coords - low) / size)


# the negative side gets the positive side's uvs mirrored after the engine unwraps the whole mesh
class MirrorUVs:
    def __init__(self, axis, center, tolerance):
        self.axis = axis
        self.center = center
        self.tolerance = tolerance

    def cut(self, obj):
        pass

    # the borders come from the whole mesh
    def with_mirrored(self, positions):
        return positions

    # the engine unwraps the whole mesh, no plane is glued
    def write_plane_sidecar(self, path, mesh, *, keep_charts):
        pass

    # the output vertices are the local ones moved by the export matrix's rotation and scale
    def finish(self, output, matrix):
        mesh = output.data
        corners, totals = face_vertex_arrays(mesh)
        local = matrix.to_3x3().inverted().to_4x4()
        targets, sources = mirrored_corner_sources(
            tools_command(),
            mirror_workdir(),
            transformed(vertex_position_array(mesh), local),
            corners,
            totals,
            self.center,
            self.axis,
            self.tolerance,
        )
        coords = loop_uvs(mesh)
        kept = numpy.ones(len(coords), dtype=bool)
        kept[targets] = False
        # flipped past the kept side's right edge
        right = coords[kept, 0].max()
        coords[targets, 0] = 2 * right - coords[sources, 0]
        coords[targets, 1] = coords[sources, 1]
        low = coords.min(axis=0)
        size = (coords.max(axis=0) - low).max()
        if size > 0:
            coords = (coords - low) / size
        set_loop_uvs(mesh, coords)
