import contextlib
import functools
import shutil
import time
from collections import deque, namedtuple

import bmesh
import bpy
import mathutils
import numpy

from ..engines import active_engine, get_engine
from ..handler import handle_error
from ..job import (
    HideInput,
    Join,
    MirrorUVs,
    Preserve,
    ProxyCopyUVs,
    ProxyUVs,
    Result,
    Symmetrise,
    TransferUVs,
    mirror_segmentation,
    mirror_workdir,
    paints_seams,
    unwrap_plan,
)
from ..logger import logger
from ..manager import Preparing, manager
from ..part_overrides import (
    piece_keeps_uvs,
    # piece_priority_override,
    read_overrides,
)
from ..proxy import (
    make_proxy,
    mirror_proxy,
    needs_proxy,
    part_triangles,
    triangle_count,
)
from ..seams import (
    EngineError,
    face_edges,
    island_layout,
    islands_overlap,
    uv_island_groups,
)
from ..seams.mirror import mirror_partners
from ..seams.worker import WorkerError
from ..segmentation import (
    SegmentWeld,
    make_segmented_proxy,
    segment_work,
    split_on_borders,
)
from ..similar import find_twins, write_twin_output
from ..ui.panels import describe_settings, unwrap_settings
from ..unwrap import Unwrap
from ..utils.geometry import applied_matrix
from ..utils.io import export_obj
from ..utils.mesh import (
    check_exists,
    deselect_all,
    face_vertex_arrays,
    get_shading_modifiers,
    is_shading_modifier,
    mark_not_unwrapped,
    new_bmesh,
    set_bmesh,
    triangulate,
    vertex_position_array,
    world_positions,
)
from ..utils.paths import get_preferences, unused_input_path
from ..utils.ui import tag_redraw
from ..utils.uvgami_tools import tools_command
from .guides import seam_restriction_weights

# finish is called with the done task
PendingWork = namedtuple("PendingWork", ["task", "obj", "finish"])

# seconds of work per tick before yielding to the event loop
TICK_BUDGET = 0.033


# two faces walking one shared edge in the same direction
def has_inconsistent_winding(mesh):
    corners = numpy.empty(len(mesh.loops), dtype=numpy.int64)
    mesh.loops.foreach_get("vertex_index", corners)
    totals = numpy.empty(len(mesh.polygons), dtype=numpy.int64)
    mesh.polygons.foreach_get("loop_total", totals)
    ends = numpy.cumsum(totals)
    next_corner = numpy.arange(1, len(corners) + 1)
    next_corner[ends - 1] = ends - totals
    directed = corners * (len(mesh.vertices) + 1) + corners[next_corner]
    return len(numpy.unique(directed)) < len(directed)


# mixed winding is optcuts exit 115
def fix_inconsistent_winding(obj):
    if not has_inconsistent_winding(obj.data):
        return
    bm = new_bmesh(obj)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    set_bmesh(bm, obj)


# the engine re-cuts a mirrored or stacked chart, losing its seams
def normalize_uvs(mesh):
    layer = mesh.uv_layers.active
    face_count = len(mesh.polygons)
    totals = numpy.empty(face_count, dtype=numpy.int64)
    mesh.polygons.foreach_get("loop_total", totals)
    starts = numpy.empty(face_count, dtype=numpy.int64)
    mesh.polygons.foreach_get("loop_start", starts)
    loop_verts = numpy.empty(len(mesh.loops), dtype=numpy.int64)
    mesh.loops.foreach_get("vertex_index", loop_verts)
    coords = numpy.empty(len(mesh.loops) * 2)
    layer.uv.foreach_get("vector", coords)
    coords = coords.reshape(-1, 2)

    # the grouping helpers take plain lists
    vert_list = loop_verts.tolist()
    uv_list = coords.tolist()
    faces = []
    uvs = []
    for start, total in zip(starts.tolist(), totals.tolist()):
        faces.append(tuple(vert_list[start : start + total]))
        uvs.append(uv_list[start : start + total])

    groups = uv_island_groups(faces, uvs, face_edges(faces))

    # shoelace area per face
    following = numpy.arange(1, len(coords) + 1)
    following[starts + totals - 1] = starts
    cross = coords[:, 0] * coords[following, 1] - coords[following, 0] * coords[:, 1]
    face_areas = 0.5 * numpy.add.reduceat(cross, starts) if face_count else []

    island_of_face = numpy.empty(face_count, dtype=numpy.int64)
    for index, group in enumerate(groups):
        island_of_face[group] = index
    loop_island = numpy.repeat(island_of_face, totals)
    order = numpy.argsort(loop_island, kind="stable")
    counts = numpy.bincount(loop_island, minlength=len(groups))
    island_loops = numpy.split(order, numpy.cumsum(counts)[:-1])

    boxes = []
    areas = []
    for group, loops in zip(groups, island_loops):
        points = coords[loops]
        boxes.append(
            (
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                float(points[:, 0].max()),
                float(points[:, 1].max()),
            )
        )
        areas.append(float(face_areas[group].sum()))

    if all(a >= 0 for a in areas) and not islands_overlap(boxes):
        return

    # nan marks an island that keeps its handedness
    flips = numpy.full(face_count, numpy.nan)
    offsets = numpy.empty((face_count, 2))
    for group, (flip, du, dv) in zip(groups, island_layout(boxes, areas)):
        if flip is not None:
            flips[group] = flip
        offsets[group] = (du, dv)

    loop_flip = numpy.repeat(flips, totals)
    mirrored = ~numpy.isnan(loop_flip)
    coords[mirrored, 0] = loop_flip[mirrored] - coords[mirrored, 0]
    coords += numpy.repeat(offsets, totals, axis=0)
    layer.uv.foreach_set("vector", coords.ravel())
    mesh.update()


class InputExporter:
    def __init__(self, engine, pieces, start_objects, temp_collection):
        self.engine = engine
        self.pieces = pieces
        self.remaining = deque(pieces)
        self.start_objects = start_objects
        self.temp_collection = temp_collection

    def tick(self):
        # exports need object mode
        if bpy.context.mode != "OBJECT":
            return 0.2

        # Cancel All ended the session mid-export
        if self.pieces and not manager.is_active:
            # called first so a failure below can't leave the count stuck
            manager.finished_adding()
            for obj, _ in self.remaining:
                bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.collections.remove(self.temp_collection)
            return None

        try:
            # a copy, a slider moved during the export doesn't reach the engine
            props = manager.props
            start = time.monotonic()
            while self.remaining:
                self._export_object(*self.remaining.popleft(), props)
                if time.monotonic() - start >= TICK_BUDGET:
                    return 0.0

            self._finish()
            return None

        except Exception as e:
            handle_error(e, "START", objects=self.start_objects)
            # handle_error doesn't delete these during a live session
            for piece in list(self.temp_collection.objects):
                bpy.data.objects.remove(piece, do_unlink=True)
            bpy.data.collections.remove(self.temp_collection)
            manager.finished_adding()
            # the session can't finish while a piece has no result
            for _, unwrap in self.pieces:
                if unwrap.result is None and not unwrap.is_exported:
                    manager.record_result(unwrap, Result.INVALID)
            return None

    def _export_object(self, obj, unwrap, props):
        if unwrap.result is not None:
            # cancelled while waiting to export
            bpy.data.objects.remove(obj, do_unlink=True)
            return
        if unwrap.copy_of is not None:
            self._export_twin(obj, unwrap, props)
            return

        path = unwrap.path
        fix_inconsistent_winding(obj)
        new_edges = self._triangulate_mesh(obj, unwrap, path, props)

        matrix = unwrap.matrix

        # seams and uvs were built before separation, see create_jobs
        if unwrap.has_uvs and not unwrap.keeps_uvs:
            normalize_uvs(obj.data)
        # a transfer writes onto the original, where the winding doesn't matter
        vt_verts = export_obj(
            obj,
            path,
            unwrap.has_uvs or unwrap.keeps_uvs,
            flip_mirrored=unwrap.keeps_output,
            matrix=matrix,
        )

        # an imported uv map keeps its layout, so the engine only mirrors it
        symmetrize_job = unwrap.symmetrize_job
        if (
            symmetrize_job is not None
            and not unwrap.keeps_uvs
            and (unwrap.preseeded or not unwrap.has_uvs)
        ):
            symmetrize_job.write_plane_sidecar(
                path, obj.data, keep_charts=unwrap.preseeded
            )

        unwrap.set_export_data(
            guide_path=self._create_guide_file(obj, path, props, vt_verts),
            origin=matrix.translation,
            added_edges=new_edges,
            **self._mesh_metadata(obj),
        )

        bpy.data.objects.remove(obj, do_unlink=True)

        # the input file is already the result
        if unwrap.keeps_uvs:
            shutil.copyfile(path, unwrap.output_path)
            manager.record_result(unwrap, Result.FINISHED)

    # a reordered twin's own indices point at the wrong vertices
    def _export_twin(self, obj, unwrap, props):
        representative = unwrap.copy_of
        if unwrap.copy_reordered:
            unwrap.set_export_data(
                origin=representative.origin,
                materials=representative.materials,
                added_edges=representative.added_edges,
                vertex_count=representative.vertex_count,
                material_indices=representative.material_indices,
                vertex_groups=representative.vertex_groups,
                face_smooth=representative.face_smooth,
                shading_modifiers=representative.shading_modifiers,
                overrides=representative.overrides,
            )
        else:
            new_edges = self._triangulate_mesh(obj, unwrap, unwrap.path, props)
            unwrap.set_export_data(
                # pieces of one object share a transform
                origin=representative.origin,
                added_edges=new_edges,
                **self._mesh_metadata(obj),
            )

        bpy.data.objects.remove(obj, do_unlink=True)

        # the representative finished before this twin was ready
        if representative.result is Result.FINISHED:
            write_twin_output(
                representative.output_path, unwrap.output_path, unwrap.copy_matrix
            )
            manager.record_result(unwrap, Result.FINISHED)

    def _finish(self):
        bpy.data.collections.remove(self.temp_collection)
        manager.finished_adding()

    # a preseeded piece locks the added edges, keep quads also dissolves them
    def _triangulate_mesh(self, obj, unwrap, path, props):
        bm = new_bmesh(obj)
        if all(len(face.verts) == 3 for face in bm.faces):
            bm.free()
            return []

        # the panel hides the setting where it doesn't apply but the value persists
        keep_quads = props.keep_quads and self.engine.supports_keep_quads
        lock = keep_quads or unwrap.preseeded
        if keep_quads:
            unwrap.preserve_job = Preserve()
        if lock:
            old_edges = set(bm.edges)

        triangulate(bm, obj.data)

        new_edges = []
        if lock:
            new_edges = [
                (edge.verts[0].index, edge.verts[1].index)
                for edge in bm.edges
                if edge not in old_edges
            ]
            with (path.parent / f"{path.stem}_locked").open("w") as f:
                f.writelines(f"{a} {b}\n" for a, b in new_edges)

        set_bmesh(bm, obj)
        return new_edges

    # _weights raises the seam cost of faces, _importance raises their stretch cost
    def _create_guide_file(self, obj, path, props, vt_verts):
        if not (
            self.engine.supports_guided and props.avoid_seams
            # (props.avoid_seams or props.reduce_stretching)
        ):
            return None
        weights = seam_restriction_weights(obj)
        if not weights:
            return None

        if vt_verts is not None:
            # optcuts rebuilds a UV-carrying obj with one vertex per vt
            weights = {
                vt: weights[v] for vt, v in enumerate(vt_verts.tolist()) if v in weights
            }

        guide = ",".join(f"{index},{weight}" for index, weight in weights.items())
        guide_path = path.parent / f"{path.stem}_weights"
        with guide_path.open("w") as f:
            f.write(f"{guide}\n")
        # if props.reduce_stretching:
        #     importance_path = path.parent / f"{path.stem}_importance"
        #     with importance_path.open("w") as f:
        #         f.write(f"{guide}\n")

        return guide_path

    # the export data read off the piece itself
    def _mesh_metadata(self, obj):
        # keep empty slots as None so the per-face indices below stay valid
        materials = [
            slot.material.name if slot.material else None for slot in obj.material_slots
        ]

        material_indices = numpy.empty(len(obj.data.polygons), dtype=numpy.int32)
        obj.data.polygons.foreach_get("material_index", material_indices)

        face_smooth = numpy.empty(len(obj.data.polygons), dtype=bool)
        obj.data.polygons.foreach_get("use_smooth", face_smooth)

        # one pass over the vertices for every group
        members = [([], []) for _ in obj.vertex_groups]
        for vertex in obj.data.vertices:
            for element in vertex.groups:
                indices, weights = members[element.group]
                indices.append(vertex.index)
                weights.append(element.weight)
        vertex_groups = {
            group.name: (numpy.array(indices, dtype=numpy.int64), numpy.array(weights))
            for group, (indices, weights) in zip(obj.vertex_groups, members)
        }

        return {
            "materials": materials,
            "material_indices": material_indices.tolist(),
            "vertex_groups": vertex_groups,
            "face_smooth": face_smooth.tolist(),
            "vertex_count": len(obj.data.vertices),
            "shading_modifiers": get_shading_modifiers(obj),
            "overrides": read_overrides(obj.data),
        }


def invalid_piece_reason(obj, *, had_uv_layer):
    if len(obj.data.polygons) == 0:
        return "No Polygons"
    if piece_keeps_uvs(obj.data) and (
        not had_uv_layer or obj.data.uv_layers.active is None
    ):
        return "No UVs To Keep"
    return None


# proxied is true when the mesh was decimated
def input_job(props, proxied, input_mesh):
    if proxied:
        return ProxyUVs(input_mesh) if props.transfer_uvs else ProxyCopyUVs(input_mesh)
    if props.transfer_uvs:
        return TransferUVs(input_mesh)
    return None


# copy is the unlinked duplicate that gets unwrapped, original is the selected object
# parts is part_triangles of the copy, set when it is proxied
MeshInput = namedtuple("MeshInput", ["copy", "original", "name", "proxied", "parts"])


# the engine's own job for each symmetry plan, None for a whole mesh the preseed mirrors
def symmetry_job(symmetry):
    if symmetry is None or symmetry.kind == "WHOLE":
        return None
    if symmetry.kind == "MIRROR_UVS":
        return MirrorUVs(symmetry.axis, symmetry.center, symmetry.tolerance)
    return Symmetrise(
        symmetry.axis,
        mathutils.Vector(symmetry.center),
        # props.sym_overlap,
        symmetry.tolerance,
        symmetry.kept_side,
    )


# segmentation and the preseed run in worker processes, the rest across timer ticks
class SessionBuilder:
    def __init__(
        self,
        engine,
        segment_command,
        input_path,
        inputs,
        start_objects,
        temp_collection,
    ):
        self.engine = engine
        self.segment_command = segment_command
        self.input_path = input_path
        self.inputs = {entry.copy: entry for entry in inputs}
        self.remaining = deque(entry.copy for entry in inputs)
        self.start_objects = start_objects
        self.temp_collection = temp_collection
        self.pieces = []
        self.piece_unwrap = {}
        self.pending = None
        # cancelled while its worker runs, dropped when the worker exits
        self.cancelled = set()
        # queue ui placeholders until each object's pieces exist
        self.preparing = {
            entry.copy: Preparing(
                entry.name, functools.partial(self.cancel_object, entry.copy)
            )
            for entry in inputs
        }
        manager.preparing.extend(self.preparing.values())

    def tick(self):
        # separation and uv writes need object mode
        if bpy.context.mode != "OBJECT":
            return 0.2
        try:
            return self._advance()
        except Exception as e:
            handle_error(e, "START", objects=self.start_objects)
            # an undo past the session start removes the collection datablock
            if check_exists(self.temp_collection):
                # handle_error doesn't delete these during a live session
                for piece in list(self.temp_collection.objects):
                    bpy.data.objects.remove(piece, do_unlink=True)
                bpy.data.collections.remove(self.temp_collection)
            manager.finished_adding()
            for entry in self.preparing.values():
                manager.drop_preparing(entry)
            self.preparing.clear()
            # settle the pieces already added, none of them exported yet
            for unwrap in self.piece_unwrap.values():
                if unwrap.result is None:
                    manager.record_result(unwrap, Result.INVALID)
            return None

    def _drop_preparing(self, obj):
        entry = self.preparing.pop(obj, None)
        if entry is not None:
            manager.drop_preparing(entry)

    # a running worker stops only at its next cancel check
    def cancel_object(self, obj):
        if self.pending is not None and self.pending.obj is obj:
            self.pending.task.cancel()
            self.cancelled.add(obj)
        else:
            self.remaining = deque(
                entry for entry in self.remaining if entry is not obj
            )
            self._drop_object(obj)

    def _drop_object(self, obj):
        self._drop_preparing(obj)
        if check_exists(obj):
            bpy.data.objects.remove(obj, do_unlink=True)

    # finish is called with the task once its worker exits
    def _wait(self, task, obj, finish):
        self.pending = PendingWork(task, obj, finish)
        return 0.1

    def _advance(self):
        if self.pending is not None:
            task, obj, finish = self.pending
            if not task.done():
                return 0.1
            self.pending = None
            if obj in self.cancelled:
                # it may have finished before it saw the flag
                self.cancelled.discard(obj)
                task.close()
                self._drop_object(obj)
                return 0.0
            if not check_exists(obj):
                task.close()
                raise RuntimeError("Undo removed the working copy mid unwrap")
            return finish(task)
        if not self.remaining:
            return self._finish()

        obj = self.remaining.popleft()
        props = manager.props
        plan = unwrap_plan(
            self.engine,
            props,
            obj.data,
            painted=paints_seams(obj, self.engine, props),
        )
        if not plan.splits_regions:
            return self._start_preseed(obj, plan.symmetry)
        worker = segment_work(
            obj.data,
            world_positions(obj.data, applied_matrix(obj)),
            self.segment_command,
            self.engine.unset_surface_mode(props),
        )
        return self._wait(
            worker,
            obj,
            functools.partial(self._finish_segmenting, obj=obj, symmetry=plan.symmetry),
        )

    def _finish_segmenting(self, task, *, obj, symmetry):
        try:
            segmentation = task.result()
        except (EngineError, WorkerError) as error:
            logger.add_data(
                "errors", f"{self.inputs[obj].name}: segmentation failed, {error}"
            )
            return self._start_preseed(obj, symmetry)
        border_positions = None
        parts = None
        symmetry, segmentation = mirror_segmentation(
            vertex_position_array(obj.data),
            *face_vertex_arrays(obj.data),
            segmentation,
            symmetry,
        )
        if segmentation.border_edges:
            border_positions, parts = split_on_borders(obj, segmentation.border_edges)
        # the preseed's classifier calls a small segment hard surface
        return self._start_preseed(
            obj,
            symmetry,
            surface_modes=segmentation.surface_modes,
            parts=parts,
            border_positions=border_positions,
        )

    # parts is part_triangles from before the segment split
    def _start_preseed(
        self,
        obj,
        symmetry,
        surface_modes=None,
        parts=None,
        border_positions=None,
    ):
        props = manager.props
        proxied = self.inputs[obj].proxied
        if proxied:
            # the copy's own parts hold until the segment split
            if parts is None:
                parts = self.inputs[obj].parts
            surface_modes = self._make_proxy(obj, props, parts, surface_modes, symmetry)
        plane = None
        if symmetry is not None and symmetry.kind == "WHOLE":
            partners = symmetry.partners
            # the proxy and the region split both renumber the vertices
            if proxied or border_positions is not None:
                partners = mirror_partners(
                    tools_command(),
                    mirror_workdir(),
                    vertex_position_array(obj.data),
                    symmetry.center,
                    symmetry.axis,
                )
            plane = ("XYZ".index(symmetry.axis), partners)

        separate = functools.partial(
            self._separate,
            obj,
            symmetrize_job=symmetry_job(symmetry),
            had_uv_layer=obj.data.uv_layers.active is not None,
            border_positions=border_positions,
        )
        # the seams package reads region widths off the full model
        work = self.engine.preseed_work(obj, props, plane, surface_modes)
        if work is None:
            return separate(preseeded=False)
        worker, apply = work
        return self._wait(
            worker,
            obj,
            functools.partial(
                self._finish_preseed, obj=obj, apply=apply, separate=separate
            ),
        )

    # returns surface_modes for the decimated faces
    def _make_proxy(self, obj, props, parts, surface_modes, symmetry):
        # the decimate only runs on an object in the scene
        bpy.context.scene.collection.objects.link(obj)
        if surface_modes is None:
            make_proxy(obj, props.proxy_faces, parts)
        else:
            surface_modes = make_segmented_proxy(
                obj, props.proxy_faces, parts, surface_modes
            )
        bpy.context.scene.collection.objects.unlink(obj)
        if symmetry is None:
            return surface_modes
        source_faces = mirror_proxy(
            obj,
            symmetry.axis,
            mathutils.Vector(symmetry.center),
            symmetry.tolerance,
            symmetry.kept_side,
        )
        if surface_modes is not None:
            surface_modes = surface_modes[source_faces]
        return surface_modes

    def _finish_preseed(self, task, *, obj, apply, separate):
        try:
            result = task.result()
        except EngineError as error:
            # each part then fails in the engine or gets the engine's own seams
            logger.add_data(
                "errors",
                f"{self.inputs[obj].name}: hard surface seams failed, {error}",
            )
            result = None
        return separate(preseeded=apply(result))

    def _input_jobs(self, props, obj):
        entry = self.inputs[obj]
        transfer_uvs_job = input_job(props, entry.proxied, entry.original)
        # the hide job doesn't depend on the unwrapped objects
        hide_job = HideInput(entry.original) if transfer_uvs_job is None else None
        return hide_job, transfer_uvs_job

    def _add_piece(
        self, obj, input_name, piece_name, path, has_uvs, props, preseeded, jobs
    ):
        uses_uvs = (
            self.engine.piece_uses_uvs(obj, props, has_uvs)
            and obj.data.uv_layers.active is not None
        )
        keeps_uvs = piece_keeps_uvs(obj.data)
        unwrap = Unwrap(
            name=piece_name,
            input_name=input_name,
            path=path,
            preseeded=preseeded and uses_uvs and not keeps_uvs,
            **jobs,
        )
        unwrap.has_uvs = uses_uvs
        unwrap.keeps_uvs = keeps_uvs
        # unwrap.priority_override = piece_priority_override(obj.data)
        self.piece_unwrap[obj] = unwrap
        manager.add(unwrap)

    def _separate(
        self, obj, *, preseeded, symmetrize_job, had_uv_layer, border_positions
    ):
        props = manager.props
        if symmetrize_job is not None:
            symmetrize_job.cut(obj)
        has_uvs = preseeded or self.engine.uses_import_uvs(props)
        # relink for the ops selection, all within one tick so nothing shows
        bpy.context.scene.collection.objects.link(obj)
        self.temp_collection.objects.unlink(obj)
        bpy.context.view_layer.update()
        # the pieces share this transform, their own matrix is never updated
        matrix = obj.matrix_world.copy()
        weld_job = None
        if border_positions is not None:
            if symmetrize_job is not None:
                border_positions = symmetrize_job.with_mirrored(border_positions)
            weld_job = SegmentWeld(border_positions, matrix)
        deselect_all()
        obj.select_set(True)

        bpy.ops.mesh.separate(type="LOOSE")
        pieces = bpy.context.selected_objects
        unwrap_name = self.inputs[obj].name
        valid = []
        for piece in pieces:
            reason = invalid_piece_reason(piece, had_uv_layer=had_uv_layer)
            if reason is None:
                valid.append(piece)
            else:
                mark_not_unwrapped(piece, reason, unwrap_name)
                manager.moved_to_invalid = True

        hide_job, transfer_uvs_job = self._input_jobs(props, obj)
        jobs = {
            "join_job": Join(len(valid)) if len(pieces) > 1 else None,
            "hide_job": hide_job,
            "symmetrize_job": symmetrize_job,
            "transfer_uvs_job": transfer_uvs_job,
            "segment_weld_job": weld_job,
        }
        claimed = manager.claimed_paths()
        for index, piece in enumerate(valid):
            piece_name = unwrap_name
            if jobs["join_job"] is not None:
                piece_name = f"{unwrap_name}_{index + 1}"
            path = unused_input_path(self.input_path, piece_name, claimed)
            claimed.add(path)
            self._add_piece(
                piece, unwrap_name, piece_name, path, has_uvs, props, preseeded, jobs
            )
        if props.stack_similar and len(valid) > 1:
            self._stack_twins(valid)

        self._drop_preparing(obj)

        deselect_all()
        for piece in valid:
            unwrap = self.piece_unwrap[piece]
            unwrap.matrix = matrix
            self.pieces.append((piece, unwrap))
            for coll in piece.users_collection:
                coll.objects.unlink(piece)
            self.temp_collection.objects.link(piece)
        return 0.0

    def _stack_twins(self, valid):
        # the new pieces' matrix_world needs an evaluation first
        bpy.context.view_layer.update()
        # a kept piece can't take a twin's uvs
        engine_pieces = [o for o in valid if not self.piece_unwrap[o].keeps_uvs]
        # a representative always precedes its twins in valid order
        for twin_obj, (rep_obj, twin_matrix, exact) in find_twins(
            engine_pieces
        ).items():
            twin = self.piece_unwrap[twin_obj]
            representative = self.piece_unwrap[rep_obj]
            twin.copy_of = representative
            twin.copy_matrix = twin_matrix
            twin.copy_reordered = not exact
            representative.twins.append(twin)

    def _finish(self):
        exporter = InputExporter(
            engine=self.engine,
            pieces=self.pieces,
            start_objects=self.start_objects,
            temp_collection=self.temp_collection,
        )
        # this builder's count entry passes to the exporter
        bpy.app.timers.register(exporter.tick)
        return None


class UVGAMI_OT_start(bpy.types.Operator):
    bl_idname = "uvgami.start"
    bl_label = "Unwrap"
    bl_description = "Start UV unwrap process"

    def execute(self, context):
        start_objects = set(bpy.data.objects)
        temp_collection = None
        builder_registered = False

        try:
            info = logger.new_info()
            props = context.scene.uvgami
            engine = active_engine(props.engine)
            if engine is None:
                _, error = get_engine("OPTCUTS").validate(get_preferences())
                self.report({"ERROR"}, error)
                logger.discard_info()
                return {"CANCELLED"}
            info.engine = engine.describe()
            info.settings = describe_settings(props, unwrap_settings(props))

            # a mesh added to a running session gets the first one's settings
            if manager.is_active:
                self.report({"ERROR"}, "Finish or cancel the current unwrap first")
                logger.discard_info()
                return {"CANCELLED"}

            prefs = get_preferences()
            if self._autosave_blocks(prefs):
                logger.discard_info()
                return {"CANCELLED"}
            engine_ctx, error = engine.validate(prefs)
            if error is not None:
                self.report({"ERROR"}, error)
                logger.discard_info()
                return {"CANCELLED"}
            segment_command = None
            if engine.splits_regions(props):
                try:
                    segment_command = [*tools_command(), "segment"]
                except RuntimeError as error:
                    self.report({"ERROR"}, str(error))
                    logger.discard_info()
                    return {"CANCELLED"}

            # leaving edit mode flushes the mesh the copies below take
            active = context.active_object
            if active is not None and active.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")

            inputs, reports = prepare_meshes(context, engine)
            if not inputs:
                reason = reports[0][1] if reports else "No object selected"
                self.report({"ERROR"}, reason)
                logger.discard_info()
                return {"CANCELLED"}

            input_path = manager.prepare_input_dir()
            deselect_all()

            # an unlinked collection doesn't flash in the viewport between ticks
            temp_collection = bpy.data.collections.new("UVgami Temp")
            for entry in inputs:
                for coll in entry.copy.users_collection:
                    coll.objects.unlink(entry.copy)
                temp_collection.objects.link(entry.copy)

            builder = SessionBuilder(
                engine=engine,
                segment_command=segment_command,
                input_path=input_path,
                inputs=inputs,
                start_objects=start_objects,
                temp_collection=temp_collection,
            )
            bpy.app.timers.register(builder.tick)
            builder_registered = True

            # before the first piece exists, the bar is up while the builder works
            manager.pieces_still_arriving += 1
            manager.start(engine, engine_ctx)
            tag_redraw()

            for level, text in reports:
                self.report({level}, text)

        except Exception as e:
            handle_error(e, "START", objects=start_objects)
            # once the builder is registered tick() removes the collection
            if not builder_registered and temp_collection is not None:
                bpy.data.collections.remove(temp_collection)

        return {"FINISHED"}

    def _autosave_blocks(self, prefs):
        if not prefs.autosave:
            return False

        if bpy.data.is_saved:
            bpy.ops.wm.save_mainfile()
            return False

        bpy.ops.wm.save_as_mainfile("INVOKE_DEFAULT")
        self.report(
            {"WARNING"},
            "Autosave is turned on. Save the file before starting UVgami",
        )
        return True


# returns the MeshInput of each mesh to unwrap and the (level, text) reports
def prepare_meshes(context, engine):
    props = context.scene.uvgami

    inputs = []
    input_sizes = []
    skipped = set()
    seen_mesh_data = set()
    applied_modifiers = False
    depsgraph = context.evaluated_depsgraph_get()
    for obj in context.selected_objects:
        if obj.type != "MESH":
            skipped.add("non mesh objects")
            continue
        if len(obj.data.polygons) == 0:
            skipped.add("objects with zero polygons")
            continue
        # the transfer writes into obj.data
        if props.transfer_uvs and obj.data in seen_mesh_data:
            skipped.add("duplicates sharing a mesh")
            continue
        seen_mesh_data.add(obj.data)

        # unlinked duplicate, so the original is never touched
        copy_object = obj.copy()
        copy_object.data = obj.data.copy()
        copy_object.animation_data_clear()

        obj.users_collection[0].objects.link(copy_object)

        # counted on the modifier bake, the mesh a plain run unwraps
        evaluated = obj.evaluated_get(depsgraph)
        triangles = triangle_count(evaluated)
        input_sizes.append((obj.name, triangles))
        # the budget reads the mesh make_proxy will decimate
        parts = part_triangles(copy_object.data) if engine.uses_proxy(props) else None
        proxied = parts is not None and needs_proxy(
            copy_object, props.proxy_faces, parts
        )
        # the engine input must be the input mesh itself, not a modifier bake of it
        if proxied or props.transfer_uvs:
            copy_object.modifiers.clear()
        elif _apply_modifiers(context, copy_object):
            applied_modifiers = True

        inputs.append(
            MeshInput(copy_object, obj, obj.name, proxied, parts if proxied else None)
        )

    logger.get_latest().objects = input_sizes

    reports = []
    if skipped:
        reports.append(("WARNING", f"Input contains {', '.join(sorted(skipped))}"))
    if applied_modifiers:
        reports.append(("WARNING", "Modifiers are applied to the unwrapped copy"))

    return inputs, reports


def _apply_modifiers(context, obj):
    context.view_layer.objects.active = obj
    applied = False
    # applying one frees its entry in obj.modifiers
    names = [m.name for m in obj.modifiers if not is_shading_modifier(m)]
    for name in names:
        # a disabled modifier can't be applied
        with contextlib.suppress(RuntimeError):
            bpy.ops.object.modifier_apply(modifier=name)
            applied = True
    return applied
