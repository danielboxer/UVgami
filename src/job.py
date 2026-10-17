from collections import namedtuple
from enum import Enum

import bmesh
import bpy
import numpy

from .hard_surface import (
    apply_face_uvs,
    apply_interior_seams,
    apply_seams,
    apply_seams_except_faces,
)
from .logger import logger
from .objfile import merge_obj_files
from .proxy import FaceLookup, transfer_inputs
from .seams import (
    uv_area_fit,
)
from .seams.proxy_transfer import (
    LOOKUP_SHARE,
    dense_subset,
    transfer_from_lookup,
)
from .seams.uv_transfer import transfer_exact_job
from .seams.worker import WorkerProcess
from .utils.mesh import (
    check_exists,
    face_vertex_arrays,
    in_object_mode,
    new_bmesh,
    set_bmesh,
    set_loop_uvs,
    triangulate,
)

TransferReport = namedtuple(
    "TransferReport", ["applied", "split_count", "detail", "reason"], defaults=("",)
)


class Result(Enum):
    FINISHED = "finished"
    INVALID = "invalid"
    CANCELLED = "cancelled"
    STOPPED = "stopped"


def world_positions(obj):
    mesh = obj.data
    flat = numpy.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", flat)
    matrix = numpy.array(obj.matrix_world)
    return flat.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]


# in the flat form transfer_exact_job takes
def output_mesh_data(output):
    output_data = output.data
    coords = numpy.empty(len(output_data.loops) * 2)
    output_data.uv_layers.active.uv.foreach_get("vector", coords)
    return (
        world_positions(output),
        *face_vertex_arrays(output_data),
        coords.reshape(-1, 2),
    )


class Preserve:
    # welding the uv layout's coincident corners leaves seams as the only boundary
    @staticmethod
    def _seam_edges(bm):
        uvs = []
        uv_idcs = []
        mesh_verts = []
        uv_count = 0
        uv_layer = bm.loops.layers.uv.active

        for face in bm.faces:
            uv_i = []
            for loop in face.loops:
                uv = loop[uv_layer].uv
                uvs.append((uv.x, uv.y, 0))
                # every face point is added, so the index is new each time
                uv_i.append(uv_count)
                uv_count += 1
                # store the original mesh vertex so it can be accessed using the uvs
                mesh_verts.append(loop.vert)
            uv_idcs.append(uv_i)

        mesh_data = bpy.data.meshes.new("")
        mesh_data.from_pydata(uvs, [], uv_idcs)
        uvbm = bmesh.new()
        uvbm.from_mesh(mesh_data)

        uvvert_to_meshvert = {}
        for uv_v_idx, uv_v in enumerate(uvbm.verts):
            uvvert_to_meshvert[uv_v] = mesh_verts[uv_v_idx]

        # the faces will all be separate, so merging by distance joins them
        bmesh.ops.remove_doubles(uvbm, verts=uvbm.verts, dist=0.0001)

        seams = []
        for e in uvbm.edges:
            if e.is_boundary:
                m_v1 = uvvert_to_meshvert[e.verts[0]]
                m_v2 = uvvert_to_meshvert[e.verts[1]]
                for edge in m_v1.link_edges:
                    if edge.other_vert(m_v1) is m_v2:
                        seams.append(edge)
        uvbm.free()
        return seams

    def finish(self, unwrap, output, added_edges):
        bm = new_bmesh(output)

        e_dict = {}
        for edge in bm.edges:
            e_dict[(edge.verts[0].index, edge.verts[1].index)] = edge

        if not added_edges:
            added_edges = unwrap.added_edges

        # partial keeps the seams, so only non-seam edges may dissolve
        seams = self._seam_edges(bm) if unwrap.maintain_mode == "PARTIAL" else ()

        dissolve_edges = []
        for e in added_edges:
            bm_edge = None

            if e in e_dict:
                bm_edge = e_dict[e]
            elif (e[1], e[0]) in e_dict:
                bm_edge = e_dict[(e[1], e[0])]
            else:
                if (
                    logger.get_latest().errors
                    and logger.get_latest().errors[-1]
                    == "    Error removing added edge"
                ):
                    continue
                logger.add_data("errors", "Error removing added edge")
                continue

            if bm_edge not in seams:
                dissolve_edges.append(bm_edge)

        bmesh.ops.dissolve_edges(bm, edges=dissolve_edges)
        set_bmesh(bm, output)


class Join:
    _last_id = 0

    def __init__(self, expected):
        # a member's stem goes stale as soon as that piece settles
        Join._last_id += 1
        self.job_id = Join._last_id
        self.expected = expected
        # every piece in creation order, settled or not, for the queue ui
        self.members = []
        # the merge concatenates in creation order, not completion order
        self.finished = []
        self.reported = 0
        # a whole-group cancel drops the finished pieces instead of joining them
        self.discard = False
        self.is_expanded = False

    def record(self, unwrap, result):
        self.reported += 1
        if result is Result.FINISHED:
            self.finished.append(unwrap)
            self.finished.sort(key=self.members.index)

    def is_settled(self):
        return self.reported == self.expected

    def finish(self):
        unwraps = self.finished
        edge_path = unwraps[-1].edge_path

        # the merge writes into the first obj, and that is what gets imported
        path = merge_obj_files([u.output_path for u in unwraps])

        added_edges = []
        if unwraps[-1].preserve_job is not None:
            v_count = 0
            for e_idx, edges in enumerate([u.added_edges for u in unwraps]):
                for v1, v2 in edges:
                    added_edges.append((v1 + v_count, v2 + v_count))
                v_count += unwraps[e_idx].vertex_count

            edge_path = unwraps[0].edge_path
            v_count = unwraps[0].vertex_count
            e_paths = [u.edge_path for u in unwraps]
            with e_paths[0].open("a") as f:
                for e_idx, e_path in enumerate(e_paths[1:], 1):
                    with e_path.open() as f2:
                        for line in f2:
                            line = line.split()
                            f.write(
                                f"{int(line[0]) + v_count} {int(line[1]) + v_count}\n"
                            )
                    v_count += unwraps[e_idx].vertex_count

        return (path, edge_path, added_edges)


class HideInput:
    def finish(self, input_mesh):
        if check_exists(input_mesh):
            input_mesh.hide_set(True)


# the read is bpy-free and runs in a worker process
class Transfer:
    # the manager repacks the input in place of the deleted output
    repack_input = True
    allows_missing_pieces = True
    # a copy of the input the result went onto, in place of the output
    replacement = None
    # poll() does a slice of the work itself
    works_in_poll = False

    def __init__(self):
        self.input_mesh = None
        self.output = None
        self.target = None
        self.task = None
        self.progress = 0.0
        self.loop_count = 0
        self.settled = False

    # None means poll() finishes it, a report means it failed before starting
    def start(self, input_mesh, output):
        if not check_exists(input_mesh) or not check_exists(output):
            return TransferReport(False, 0, "input or output object missing")
        if output.data.uv_layers.active is None:
            return TransferReport(False, 0, "output mesh has no uv layer")
        self.input_mesh = input_mesh
        self.output = output
        self.target = self._target(input_mesh)
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
            # an undo while it ran swapped the mesh out under us
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
            world_positions(target),
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
                to_split.append((face, parts, face.material_index, face.smooth))

        # delete first so a new piece can never collide with the face it replaces
        bmesh.ops.delete(
            bm, geom=[face for face, _, _, _ in to_split], context="FACES_ONLY"
        )
        bm.verts.ensure_lookup_table()
        for _, parts, material_index, smooth in to_split:
            for verts, uvs in parts:
                new_face = bm.faces.new([bm.verts[v] for v in verts])
                new_face.material_index = material_index
                new_face.smooth = smooth
                for loop, uv in zip(new_face.loops, uvs):
                    loop[uv_layer].uv = uv

        for edge in bm.edges:
            if edge in kept_edges:
                continue
            a, b = edge.verts[0].index, edge.verts[1].index
            edge.seam = ((a, b) if a < b else (b, a)) in plan.seam_edges

        set_bmesh(bm, input_mesh)


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


# rides TransferUVs' position matching, fed only the island's faces
class IslandUVs(TransferUVs):
    repack_input = False
    allows_missing_pieces = False

    def __init__(self, faces, bbox, area, mirrored=False):
        super().__init__()
        self.faces = faces
        self.bbox = bbox
        self.area = area
        self.mirrored = mirrored
        self.orig_vert = []
        self.loop_base = {}
        self.loop_counts = []

    # a mirrored island was exported with u negated
    def _unmirror(self, plan):
        if not self.mirrored:
            return
        for k in plan.loop_uvs:
            u, v = plan.loop_uvs[k]
            plan.loop_uvs[k] = (-u, v)
        for fi, parts in plan.split_faces.items():
            plan.split_faces[fi] = [
                (verts, [(-u, v) for u, v in part_uvs]) for verts, part_uvs in parts
            ]

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

    # the loops a split face came from are dead
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
        move = uv_area_fit(polygons, self.area, self.bbox)
        for k in plan.loop_uvs:
            plan.loop_uvs[k] = move(plan.loop_uvs[k])
        for fi, parts in plan.split_faces.items():
            plan.split_faces[fi] = [
                (verts, [move(uv) for uv in part_uvs]) for verts, part_uvs in parts
            ]

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

        uvs = [None] * len(data.polygons)
        for i, fi in enumerate(self.faces):
            base = self.loop_base[fi]
            uvs[fi] = [plan.loop_uvs[base + c] for c in range(self.loop_counts[i])]
        apply_face_uvs(data, uvs, self.faces)

        apply_interior_seams(data, interior, seams)
        data.update()
        return 0

    # like _apply, but rebuilds the island faces a uv cut runs through
    def _apply_island_splits(self, input_mesh, plan, split_faces, seams, interior):
        bm = new_bmesh(input_mesh)
        uv_layer = bm.loops.layers.uv.verify()
        bm.faces.ensure_lookup_table()

        to_split = []
        for fi in self.faces:
            face = bm.faces[fi]
            parts = split_faces.get(fi)
            if parts is None:
                base = self.loop_base[fi]
                for c, loop in enumerate(face.loops):
                    loop[uv_layer].uv = plan.loop_uvs[base + c]
            else:
                to_split.append((face, parts, face.material_index, face.smooth))

        bmesh.ops.delete(
            bm, geom=[face for face, _, _, _ in to_split], context="FACES_ONLY"
        )
        bm.verts.ensure_lookup_table()
        ov = self.orig_vert
        for _, parts, material_index, smooth in to_split:
            for verts, uvs in parts:
                new_face = bm.faces.new([bm.verts[ov[v]] for v in verts])
                new_face.material_index = material_index
                new_face.smooth = smooth
                for loop, uv in zip(new_face.loops, uvs):
                    loop[uv_layer].uv = uv

        for edge in bm.edges:
            a, b = edge.verts[0].index, edge.verts[1].index
            key = (a, b) if a < b else (b, a)
            if key in seams:
                edge.seam = True
            elif key in interior:
                edge.seam = False

        set_bmesh(bm, input_mesh)


# the engine held the patch border in place, the pins undo its normalization
class AreaUVs(IslandUVs):
    def __init__(self, faces, pins, mirrored):
        super().__init__(faces, None, None, mirrored)
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

        # the output is the solved map scaled into the unit box
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

        def move(uv):
            return (scale * uv[0] + du, scale * uv[1] + dv)

        for k in plan.loop_uvs:
            plan.loop_uvs[k] = move(plan.loop_uvs[k])
        for fi, parts in plan.split_faces.items():
            plan.split_faces[fi] = [
                (verts, [move(uv) for uv in part_uvs]) for verts, part_uvs in parts
            ]
        # pinned loops drift by float noise, restore them so the border welds
        for fi, corner, old in self.pins:
            k = self.loop_base[fi] + corner
            if k in plan.loop_uvs:
                plan.loop_uvs[k] = old


# the window is frozen for this long per poll
LOOKUP_SLICE_SECONDS = 0.05


# the original was never unwrapped itself, so it is cut where the map is torn
class ProxyUVs(Transfer):
    # a missing piece is a hole in the proxy map
    allows_missing_pieces = False

    def _extract(self, target, output):
        return transfer_inputs(target, output)

    @property
    def works_in_poll(self):
        return self.task is None

    def _begin(self, inputs):
        self._inputs = inputs
        self._lookup = FaceLookup(*inputs)

    def _work_done(self):
        if self.task is not None:
            self.progress = max(self.progress, self.task.progress)
            return self.task.done()
        self._lookup.advance(LOOKUP_SLICE_SECONDS)
        self.progress = LOOKUP_SHARE * self._lookup.fraction
        if self._lookup.done:
            self.task = WorkerProcess(
                transfer_from_lookup,
                *self._inputs,
                self._lookup.face_of_vertex,
            )
            self._inputs = self._lookup = None
        return False

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
        bm = new_bmesh(target)
        triangulate(bm, target.data)
        set_bmesh(bm, target)
        return target

    def _show(self):
        # no HideInput job exists when a transfer job holds the slot
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

    def __init__(self, faces, bbox, area):
        super().__init__()
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
        uvs_by_face = [None] * len(data.polygons)
        for fi, points in zip(self.faces, polygons):
            uvs_by_face[fi] = [move(uv) for uv in points]
        apply_face_uvs(data, uvs_by_face, self.faces)

        ov = self.orig_vert.tolist()
        seams = {
            ((ov[a], ov[b]) if ov[a] < ov[b] else (ov[b], ov[a])) for a, b in seams
        }
        # the island boundary and the rest of the mesh keep their marks
        apply_interior_seams(data, _interior_edges(data, self.faces), seams)
        data.update()
        return 0
