from collections import namedtuple
from enum import Enum

import bmesh
import bpy
import numpy

from .hard_surface import (
    apply_interior_seams,
    apply_loop_uvs,
    apply_seams_except_faces,
)
from .logger import logger
from .no_bpy import fit_uvs_keeping_area, sorted_pair
from .no_bpy.uv_transfer import TransferError, transfer_exact_job
from .no_bpy.worker import WorkerProcess
from .objfile import merge_obj_files
from .symmetry import symmetry_plan
from .utils.mesh import (
    check_exists,
    in_object_mode,
    loop_uvs,
    loop_vertices_and_face_sizes,
    new_bmesh,
    set_bmesh,
    set_loop_uvs,
    world_positions,
)

TransferReport = namedtuple(
    "TransferReport", ["applied", "split_count", "detail", "reason"], defaults=("",)
)


class Result(Enum):
    FINISHED = "finished"
    INVALID = "invalid"
    CANCELLED = "cancelled"
    STOPPED = "stopped"


def output_mesh_data(output):
    output_data = output.data
    corners, totals = loop_vertices_and_face_sizes(output_data)
    return {
        "output_positions": world_positions(output_data, output.matrix_world),
        "output_corners": corners,
        "output_totals": totals,
        "output_loop_uvs": loop_uvs(output_data),
    }


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


REMOVE_EDGE_ERROR = "Error removing added edge"


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
                    logger.add_error(REMOVE_EDGE_ERROR)
                continue
            # optcuts never cuts locked edges unless the input uv map has a seam there
            if not _carries_seam(edge, uv_layer):
                dissolve_edges.append(edge)

        bmesh.ops.dissolve_edges(bm, edges=dissolve_edges)
        set_bmesh(bm, output)


class Join:
    _last_id = 0

    def __init__(self, expected):
        Join._last_id += 1
        self.job_id = Join._last_id
        self.expected = expected
        self.members = []
        self.finished = []
        self.reported = 0
        self.discard = False
        self.is_expanded = False

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

        path = merge_obj_files([u.output_path for u in unwraps])

        added_edges = []
        offset = 0
        for unwrap in unwraps:
            export = unwrap.export
            added_edges.extend((a + offset, b + offset) for a, b in export.added_edges)
            offset += export.vertex_count

        return path, added_edges


class HideInput:
    def __init__(self, input_mesh):
        self.input_mesh = input_mesh

    def finish(self):
        if check_exists(self.input_mesh):
            self.input_mesh.hide_set(True)


class Transfer:
    repack_input = True
    allows_missing_pieces = True
    replacement = None

    def __init__(self, input_mesh):
        self.input_mesh = input_mesh
        self.output = None
        self.target = None
        self.task = None
        self.loop_count = 0
        self.settled = False

    @property
    def progress(self):
        return self.task.progress

    def start(self, output):
        if not check_exists(self.input_mesh) or not check_exists(output):
            return TransferReport(
                applied=False, split_count=0, detail="input or output object missing"
            )
        if output.data.uv_layers.active is None:
            return TransferReport(
                applied=False, split_count=0, detail="output mesh has no uv layer"
            )
        self.output = output
        self.target = self._target(self.input_mesh)
        inputs = in_object_mode(self.target, self._extract, self.target, output)
        self.loop_count = len(self.target.data.loops)
        self._begin(inputs)
        return None

    def poll(self):
        if not self.task.done():
            return None
        try:
            result = self.task.result()
        except TransferError as error:
            return self._fail(str(error), error.reason)
        if not check_exists(self.target) or not check_exists(self.output):
            return self._fail("input or output object missing")
        if len(self.target.data.loops) != self.loop_count:
            # undo during a run replaces the mesh
            return self._fail("mesh changed during the unwrap")
        split_count = in_object_mode(self.target, self._apply, self.target, result)
        bpy.data.objects.remove(self.output, do_unlink=True)
        self._show()
        return TransferReport(applied=True, split_count=split_count, detail="")

    def cancel(self):
        self.settled = True
        if self.task is not None:
            self.task.close()
        self._discard_start_extras()
        if check_exists(self.output):
            bpy.data.objects.remove(self.output, do_unlink=True)

    def _fail(self, detail, reason=""):
        self._discard_start_extras()
        return TransferReport(
            applied=False, split_count=0, detail=detail, reason=reason
        )

    def _target(self, input_mesh):
        return input_mesh

    def _discard_start_extras(self):
        pass

    def _show(self):
        self.input_mesh.hide_set(False)


class TransferUVs(Transfer):
    def _extract(self, target, output):
        corners, totals = loop_vertices_and_face_sizes(target.data)
        return {
            "input_positions": world_positions(target.data, target.matrix_world),
            "input_corners": corners,
            "input_totals": totals,
            **output_mesh_data(output),
        }

    def _begin(self, inputs):
        self.task = WorkerProcess(
            transfer_exact_job,
            **inputs,
            repack=self.repack_input,
            partial=self.allows_missing_pieces,
        )

    def _apply(self, target, plan):
        if plan.split_faces:
            self._apply_with_splits(target, plan)
            return len(plan.split_faces)

        data = target.data
        if not data.uv_layers:
            data.uv_layers.new(name="UVMap")

        coords = loop_uvs(data)
        coords[list(plan.loop_uvs)] = list(plan.loop_uvs.values())
        set_loop_uvs(data, coords)

        apply_seams_except_faces(data, plan.untouched_faces, plan.seam_edges)
        data.update()
        return 0

    def _apply_with_splits(self, input_mesh, plan):
        bm = new_bmesh(input_mesh)
        uv_layer = bm.loops.layers.uv.verify()

        loop_index = 0
        to_split = []
        kept_edges = set()
        for face_index, face in enumerate(bm.faces):
            parts = plan.split_faces.get(face_index)
            if face_index in plan.untouched_faces:
                kept_edges.update(face.edges)
                loop_index += len(face.loops)
            elif parts is None:
                for loop in face.loops:
                    loop[uv_layer].uv = plan.loop_uvs[loop_index]
                    loop_index += 1
            else:
                loop_index += len(face.loops)
                to_split.append((face, parts))

        _rebuild_split_faces(bm, uv_layer, to_split)
        for edge in bm.edges:
            if edge in kept_edges:
                continue
            key = sorted_pair(edge.verts[0].index, edge.verts[1].index)
            edge.seam = key in plan.seam_edges

        set_bmesh(bm, input_mesh)


def _rebuild_split_faces(bm, uv_layer, to_split):
    pieces = [(parts, face.material_index, face.smooth) for face, parts in to_split]
    # bm.faces.new raises on a face that already exists
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
    for loop in plan.loop_uvs:
        plan.loop_uvs[loop] = move(plan.loop_uvs[loop])
    for face, parts in plan.split_faces.items():
        plan.split_faces[face] = [
            (verts, [move(uv) for uv in part_uvs]) for verts, part_uvs in parts
        ]


def edges_inside_faces(data, faces):
    owner_count = {}
    for face in faces:
        vertices = data.polygons[face].vertices
        size = len(vertices)
        for corner in range(size):
            key = sorted_pair(vertices[corner], vertices[(corner + 1) % size])
            owner_count[key] = owner_count.get(key, 0) + 1
    return {key for key, count in owner_count.items() if count == 2}


def island_edges_on_mesh(island_vertices, edges):
    return {sorted_pair(island_vertices[a], island_vertices[b]) for a, b in edges}


def apply_island_uvs_and_seams(data, faces, uvs, seams):
    apply_loop_uvs(data, uvs, faces)
    apply_interior_seams(data, edges_inside_faces(data, faces), seams)
    data.update()


class IslandUVs(TransferUVs):
    repack_input = False
    allows_missing_pieces = False

    def __init__(self, input_mesh, faces, bbox, area, *, mirrored=False):
        super().__init__(input_mesh)
        self.faces = faces
        self.bbox = bbox
        self.area = area
        self.mirrored = mirrored
        self.orig_vert = []
        self.loop_base = {}
        self.loop_counts = []

    def _unmirror(self, plan):
        if self.mirrored:
            _move_plan_uvs(plan, lambda uv: (-uv[0], uv[1]))

    def _extract(self, target, output):
        data = target.data
        matrix = target.matrix_world

        used = sorted(
            {vertex for face in self.faces for vertex in data.polygons[face].vertices}
        )
        self.orig_vert = used
        local = {vertex: index for index, vertex in enumerate(used)}

        positions = world_positions(data, matrix)[used]
        corners = []
        self.loop_counts = []
        base = 0
        for face in self.faces:
            polygon = data.polygons[face]
            self.loop_base[face] = base
            self.loop_counts.append(polygon.loop_total)
            base += polygon.loop_total
            corners.extend(local[vertex] for vertex in polygon.vertices)

        return {
            "input_positions": positions,
            "input_corners": numpy.array(corners, dtype=numpy.int64),
            "input_totals": numpy.array(self.loop_counts, dtype=numpy.int64),
            **output_mesh_data(output),
        }

    def _fit(self, plan):
        self._unmirror(plan)
        polygons = []
        for index, count in enumerate(self.loop_counts):
            parts = plan.split_faces.get(index)
            if parts is None:
                base = self.loop_base[self.faces[index]]
                polygons.append(
                    [plan.loop_uvs[base + corner] for corner in range(count)]
                )
            else:
                polygons.extend(part_uvs for _, part_uvs in parts)
        _move_plan_uvs(plan, fit_uvs_keeping_area(polygons, self.area, self.bbox))

    def _apply(self, target, plan):
        self._fit(plan)
        data = target.data
        seams = island_edges_on_mesh(self.orig_vert, plan.seam_edges)

        split_faces = {
            self.faces[face]: parts for face, parts in plan.split_faces.items()
        }
        if split_faces:
            interior = edges_inside_faces(data, self.faces)
            self._apply_island_splits(target, plan, split_faces, seams, interior)
            return len(plan.split_faces)

        uvs = [
            plan.loop_uvs[self.loop_base[face] + corner]
            for index, face in enumerate(self.faces)
            for corner in range(self.loop_counts[index])
        ]
        apply_island_uvs_and_seams(data, self.faces, numpy.array(uvs), seams)
        return 0

    def _apply_island_splits(self, input_mesh, plan, split_faces, seams, interior):
        bm = new_bmesh(input_mesh)
        uv_layer = bm.loops.layers.uv.verify()
        bm.faces.ensure_lookup_table()

        original_vertex = self.orig_vert
        to_split = []
        for face_index in self.faces:
            face = bm.faces[face_index]
            parts = split_faces.get(face_index)
            if parts is None:
                base = self.loop_base[face_index]
                for corner, loop in enumerate(face.loops):
                    loop[uv_layer].uv = plan.loop_uvs[base + corner]
            else:
                parts = [
                    ([original_vertex[vertex] for vertex in verts], uvs)
                    for verts, uvs in parts
                ]
                to_split.append((face, parts))

        _rebuild_split_faces(bm, uv_layer, to_split)
        for edge in bm.edges:
            key = sorted_pair(edge.verts[0].index, edge.verts[1].index)
            if key in seams:
                edge.seam = True
            elif key in interior:
                edge.seam = False

        set_bmesh(bm, input_mesh)


class AreaUVs(IslandUVs):
    def __init__(self, input_mesh, faces, pins, *, mirrored):
        super().__init__(input_mesh, faces, None, None, mirrored=mirrored)
        self.pins = pins

    def _fit(self, plan):
        self._unmirror(plan)
        # optcuts scales its output into the unit box
        pairs = []
        for face, corner, old in self.pins:
            new = plan.loop_uvs.get(self.loop_base[face] + corner)
            if new is not None:
                pairs.append((old, new))
        if len(pairs) < 2:
            return

        low = min(pairs, key=lambda pair: pair[0][0])
        high = max(pairs, key=lambda pair: pair[0][0])
        if high[1][0] == low[1][0]:
            low = min(pairs, key=lambda pair: pair[0][1])
            high = max(pairs, key=lambda pair: pair[0][1])
            scale = (high[0][1] - low[0][1]) / (high[1][1] - low[1][1])
        else:
            scale = (high[0][0] - low[0][0]) / (high[1][0] - low[1][0])
        shift_u = low[0][0] - scale * low[1][0]
        shift_v = low[0][1] - scale * low[1][1]

        _move_plan_uvs(
            plan, lambda uv: (scale * uv[0] + shift_u, scale * uv[1] + shift_v)
        )
        # pinned loops drift by float noise
        for face, corner, old in self.pins:
            loop = self.loop_base[face] + corner
            if loop in plan.loop_uvs:
                plan.loop_uvs[loop] = old


def unwrap_symmetry(engine, props, mesh, *, painted):
    if not engine.uses_symmetry(props) or len(mesh.vertices) == 0:
        return None
    return symmetry_plan(engine, mesh, painted)
