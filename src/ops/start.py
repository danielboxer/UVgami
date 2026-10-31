import functools
import shutil
import time
from collections import deque, namedtuple

import bmesh
import bpy
import numpy

from ..engines import ENGINES, active_engine
from ..handler import handle_start_error
from ..job import (
    HideInput,
    Join,
    Preserve,
    Result,
    TransferUVs,
    unwrap_symmetry,
)
from ..logger import logger
from ..manager import Preparing, manager
from ..no_bpy import (
    EngineError,
    boxes_overlap,
    faces_by_edge,
    island_layout,
    islands_from_uvs,
)
from ..no_bpy.mesh import moved_island_uvs, next_corners_in_face
from ..no_bpy.worker import WorkerError
from ..part_overrides import (
    piece_keeps_uvs,
    # piece_priority_override,
    read_overrides,
)
from ..proxy import (
    ProxyCopyUVs,
    ProxyUVs,
    make_proxy,
    mirror_proxy,
    needs_proxy,
    part_triangles,
    triangle_count,
)
from ..similar import find_twins, write_twin_output
from ..symmetry import (
    MirrorUVs,
    Symmetrise,
    mesh_mirror_partners,
    mirror_segmentation,
    paints_seams,
)
from ..ui.props import describe_settings, unwrap_settings
from ..unwrap import ExportData, Unwrap
from ..utils.io import export_obj
from ..utils.mesh import (
    applied_matrix,
    check_exists,
    deselect_all,
    get_shading_modifiers,
    is_shading_modifier,
    loop_starts,
    loop_uvs,
    loop_vertices_and_face_sizes,
    mark_not_unwrapped,
    mesh_arrays,
    new_bmesh,
    set_bmesh,
    set_loop_uvs,
    triangulate,
    world_positions,
)
from ..utils.paths import get_preferences, unused_input_path
from ..utils.ui import tag_redraw
from ..utils.uvgami_tools import tools_command
from .guides import painted_weight_by_vertex


def has_inconsistent_winding(mesh):
    corners, totals = loop_vertices_and_face_sizes(mesh)
    next_corner = next_corners_in_face(totals)
    directed = corners * (len(mesh.vertices) + 1) + corners[next_corner]
    return len(numpy.unique(directed)) < len(directed)


def fix_inconsistent_winding(obj):
    # mixed winding is optcuts exit 115
    if not has_inconsistent_winding(obj.data):
        return
    bm = new_bmesh(obj)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    set_bmesh(bm, obj)


def normalize_uvs(mesh):
    face_count = len(mesh.polygons)
    loop_vertices, totals = loop_vertices_and_face_sizes(mesh)
    starts = loop_starts(mesh)
    coords = loop_uvs(mesh)

    vert_list = loop_vertices.tolist()
    uv_list = coords.tolist()
    faces = []
    uvs = []
    for start, total in zip(starts.tolist(), totals.tolist()):
        faces.append(tuple(vert_list[start : start + total]))
        uvs.append(uv_list[start : start + total])

    groups = islands_from_uvs(faces, uvs, faces_by_edge(faces))

    following = next_corners_in_face(totals)
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

    # optcuts re-cuts mirrored or stacked charts
    if all(a >= 0 for a in areas) and not boxes_overlap(boxes):
        return

    moved = moved_island_uvs(coords, groups, island_layout(boxes, areas), totals)
    set_loop_uvs(mesh, moved)
    mesh.update()


def discard_session_pieces(temp_collection, unwraps):
    # undo past the session start removes the collection datablock
    if check_exists(temp_collection):
        for piece in list(temp_collection.objects):
            bpy.data.objects.remove(piece, do_unlink=True)
        bpy.data.collections.remove(temp_collection)
    manager.finished_adding()
    for unwrap in unwraps:
        if unwrap.result is None and not unwrap.is_exported:
            manager.record_result(unwrap, Result.INVALID)


TICK_BUDGET = 0.033


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

        if self.pieces and not manager.is_active:
            manager.finished_adding()
            for obj, _ in self.remaining:
                bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.collections.remove(self.temp_collection)
            return None

        try:
            props = manager.props
            start = time.monotonic()
            while self.remaining:
                self._export_object(*self.remaining.popleft(), props)
                if time.monotonic() - start >= TICK_BUDGET:
                    return 0.0

            bpy.data.collections.remove(self.temp_collection)
            manager.finished_adding()
            return None

        except Exception as error:
            handle_start_error(error, self.start_objects)
            discard_session_pieces(
                self.temp_collection, [unwrap for _, unwrap in self.pieces]
            )
            return None

    def _export_object(self, obj, unwrap, props):
        if unwrap.result is not None:
            bpy.data.objects.remove(obj, do_unlink=True)
            return
        if unwrap.copy_of is not None:
            self._export_twin(obj, unwrap, props)
            return

        path = unwrap.path
        fix_inconsistent_winding(obj)
        new_edges = self._triangulate_mesh(obj, unwrap, props)

        matrix = unwrap.matrix

        if unwrap.has_uvs and not unwrap.keeps_uvs:
            normalize_uvs(obj.data)
        vertex_of_uv = export_obj(
            obj,
            path,
            export_uv=unwrap.has_uvs or unwrap.keeps_uvs,
            flip_mirrored=unwrap.keeps_output,
            matrix=matrix,
        )

        symmetrize_job = unwrap.symmetrize_job
        if (
            symmetrize_job is not None
            and not unwrap.keeps_uvs
            and (unwrap.preseeded or not unwrap.has_uvs)
        ):
            symmetrize_job.write_plane_sidecar(
                path, obj.data, keep_charts=unwrap.preseeded
            )

        self._write_guide_file(obj, path, props, vertex_of_uv)
        unwrap.export = ExportData(
            origin=matrix.translation.copy(),
            added_edges=new_edges,
            **self._mesh_metadata(obj),
        )

        bpy.data.objects.remove(obj, do_unlink=True)

        if unwrap.keeps_uvs:
            shutil.copyfile(path, unwrap.output_path)
            manager.record_result(unwrap, Result.FINISHED)

    def _export_twin(self, obj, unwrap, props):
        representative = unwrap.copy_of
        if unwrap.copy_reordered:
            unwrap.export = representative.export
        else:
            new_edges = self._triangulate_mesh(obj, unwrap, props)
            unwrap.export = ExportData(
                origin=representative.export.origin,
                added_edges=new_edges,
                **self._mesh_metadata(obj),
            )

        bpy.data.objects.remove(obj, do_unlink=True)

        if representative.result is Result.FINISHED:
            write_twin_output(
                representative.output_path, unwrap.output_path, unwrap.copy_matrix
            )
            manager.record_result(unwrap, Result.FINISHED)

    def _triangulate_mesh(self, obj, unwrap, props):
        bm = new_bmesh(obj)
        if all(len(face.verts) == 3 for face in bm.faces):
            bm.free()
            return []

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
            path = unwrap.path
            with (path.parent / f"{path.stem}_locked").open("w") as locked_file:
                locked_file.writelines(f"{a} {b}\n" for a, b in new_edges)

        set_bmesh(bm, obj)
        return new_edges

    def _write_guide_file(self, obj, path, props, vertex_of_uv):
        if not (
            self.engine.supports_guided and props.avoid_seams
            # (props.avoid_seams or props.reduce_stretching)
        ):
            return
        weights = painted_weight_by_vertex(obj)
        if not weights:
            return

        if vertex_of_uv is not None:
            # optcuts rebuilds a UV-carrying obj with one vertex per vt
            weights = {
                uv: weights[vertex]
                for uv, vertex in enumerate(vertex_of_uv.tolist())
                if vertex in weights
            }

        guide = ",".join(f"{index},{weight}" for index, weight in weights.items())
        guide_path = path.parent / f"{path.stem}_weights"
        with guide_path.open("w") as guide_file:
            guide_file.write(f"{guide}\n")
        # if props.reduce_stretching:
        #     importance_path = path.parent / f"{path.stem}_importance"
        #     with importance_path.open("w") as f:
        #         f.write(f"{guide}\n")

    def _mesh_metadata(self, obj):
        materials = [
            slot.material.name if slot.material else None for slot in obj.material_slots
        ]

        material_indices = numpy.empty(len(obj.data.polygons), dtype=numpy.int32)
        obj.data.polygons.foreach_get("material_index", material_indices)

        face_smooth = numpy.empty(len(obj.data.polygons), dtype=bool)
        obj.data.polygons.foreach_get("use_smooth", face_smooth)

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


MeshInput = namedtuple("MeshInput", ["copy", "original", "name", "proxy_parts"])
PreparedMeshes = namedtuple("PreparedMeshes", ["inputs", "reports", "input_sizes"])


def symmetry_job(symmetry):
    if symmetry is None or symmetry.kind == "WHOLE":
        return None
    if symmetry.kind == "MIRROR_UVS":
        return MirrorUVs(symmetry)
    return Symmetrise(
        symmetry,
        # props.sym_overlap,
    )


PendingWork = namedtuple("PendingWork", ["task", "obj", "finish"])


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
        self.cancelled = set()
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
        except Exception as error:
            handle_start_error(error, self.start_objects)
            discard_session_pieces(self.temp_collection, self.piece_unwrap.values())
            for entry in self.preparing.values():
                manager.drop_preparing(entry)
            self.preparing.clear()
            return None

    def _drop_preparing(self, obj):
        entry = self.preparing.pop(obj, None)
        if entry is not None:
            manager.drop_preparing(entry)

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
        symmetry = unwrap_symmetry(
            self.engine,
            props,
            obj.data,
            painted=paints_seams(obj, self.engine, props),
        )
        if not self.engine.splits_regions(props):
            return self._start_preseed(obj, symmetry)
        from ..segmentation import segment_work

        worker = segment_work(
            obj.data,
            world_positions(obj.data, applied_matrix(obj)),
            self.segment_command,
            self.engine.unset_surface_mode(props),
        )
        return self._wait(
            worker,
            obj,
            functools.partial(self._finish_segmenting, obj=obj, symmetry=symmetry),
        )

    def _finish_segmenting(self, task, *, obj, symmetry):
        try:
            segmentation = task.result()
        except (EngineError, WorkerError) as error:
            logger.add_error(f"{self.inputs[obj].name}: segmentation failed, {error}")
            return self._start_preseed(obj, symmetry)
        border_positions = None
        parts = None
        symmetry, segmentation = mirror_segmentation(
            *mesh_arrays(obj.data),
            segmentation,
            symmetry,
        )
        if segmentation.border_edges:
            from ..segmentation import split_on_borders

            border_positions, parts = split_on_borders(obj, segmentation.border_edges)
        return self._start_preseed(
            obj,
            symmetry,
            surface_modes=segmentation.surface_modes,
            parts=parts,
            border_positions=border_positions,
        )

    def _start_preseed(
        self,
        obj,
        symmetry,
        surface_modes=None,
        parts=None,
        border_positions=None,
    ):
        props = manager.props
        proxied = self.inputs[obj].proxy_parts is not None
        if proxied:
            if parts is None:
                parts = self.inputs[obj].proxy_parts
            surface_modes = self._make_proxy(obj, props, parts, surface_modes, symmetry)
        plane = None
        if symmetry is not None and symmetry.kind == "WHOLE":
            partners = symmetry.partners
            if proxied or border_positions is not None:
                partners = mesh_mirror_partners(
                    obj.data, symmetry.center, symmetry.axis
                )
            plane = ("XYZ".index(symmetry.axis), partners)

        separate = functools.partial(
            self._separate,
            obj,
            symmetrize_job=symmetry_job(symmetry),
            had_uv_layer=obj.data.uv_layers.active is not None,
            border_positions=border_positions,
        )
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

    def _make_proxy(self, obj, props, parts, surface_modes, symmetry):
        # Decimate modifiers only evaluate on objects in the scene
        bpy.context.scene.collection.objects.link(obj)
        if surface_modes is None:
            make_proxy(obj, props.proxy_faces, parts)
        else:
            from ..segmentation import make_segmented_proxy

            surface_modes = make_segmented_proxy(
                obj, props.proxy_faces, parts, surface_modes
            )
        bpy.context.scene.collection.objects.unlink(obj)
        if symmetry is None:
            return surface_modes
        source_faces = mirror_proxy(obj, symmetry)
        if surface_modes is not None:
            surface_modes = surface_modes[source_faces]
        return surface_modes

    def _finish_preseed(self, task, *, obj, apply, separate):
        try:
            result = task.result()
        except (EngineError, WorkerError) as error:
            logger.add_error(
                f"{self.inputs[obj].name}: hard surface seams failed, {error}"
            )
            result = None
        return separate(preseeded=apply(result))

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
        bpy.context.scene.collection.objects.link(obj)
        self.temp_collection.objects.unlink(obj)
        bpy.context.view_layer.update()
        matrix = obj.matrix_world.copy()
        weld_job = None
        if border_positions is not None:
            from ..segmentation import SegmentWeld

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

        entry = self.inputs[obj]
        if entry.proxy_parts is not None and props.transfer_uvs:
            transfer_uvs_job = ProxyUVs(entry.original)
        elif entry.proxy_parts is not None:
            transfer_uvs_job = ProxyCopyUVs(entry.original)
        elif props.transfer_uvs:
            transfer_uvs_job = TransferUVs(entry.original)
        else:
            transfer_uvs_job = None
        jobs = {
            "join_job": Join(len(valid)) if len(pieces) > 1 else None,
            "hide_job": HideInput(entry.original) if transfer_uvs_job is None else None,
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
            for collection in piece.users_collection:
                collection.objects.unlink(piece)
            self.temp_collection.objects.link(piece)
        return 0.0

    def _stack_twins(self, valid):
        # matrix_world on new objects is stale until view_layer.update
        bpy.context.view_layer.update()
        engine_pieces = [
            piece for piece in valid if not self.piece_unwrap[piece].keeps_uvs
        ]
        for twin_obj, (representative_obj, twin_matrix, exact) in find_twins(
            engine_pieces
        ).items():
            twin = self.piece_unwrap[twin_obj]
            representative = self.piece_unwrap[representative_obj]
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
        bpy.app.timers.register(exporter.tick)
        return None


class UVGAMI_OT_start(bpy.types.Operator):
    bl_idname = "uvgami.start"
    bl_label = "Unwrap"
    bl_description = "Start UV unwrap process"

    def execute(self, context):
        props = context.scene.uvgami
        engine = active_engine(props.engine)
        try:
            if engine is None:
                ENGINES["OPTCUTS"].validate()
                raise EngineError("The selected engine is not installed")
            engine_ctx = engine.validate()
            segment_command = tools_command() if engine.splits_regions(props) else None
        except EngineError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}

        if manager.is_active:
            self.report({"ERROR"}, "Finish or cancel the current unwrap first")
            return {"CANCELLED"}
        if self._autosave_blocks(get_preferences()):
            return {"CANCELLED"}

        start_objects = set(bpy.data.objects)
        temp_collection = None
        builder_registered = False

        try:
            info = logger.new_info()
            info.engine = engine.describe()
            info.settings = describe_settings(props, unwrap_settings(props))

            # leaving edit mode flushes the mesh the copies below take
            active = context.active_object
            if active is not None and active.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")

            prepared = prepare_meshes(context, engine)
            info.objects = prepared.input_sizes
            inputs = prepared.inputs
            if not inputs:
                reasons = prepared.reports
                reason = reasons[0][1] if reasons else "No object selected"
                self.report({"ERROR"}, reason)
                logger.discard_refused_run()
                return {"CANCELLED"}

            input_path = manager.prepare_input_dir()
            deselect_all()

            # unlinked collections don't show in the viewport between ticks
            temp_collection = bpy.data.collections.new("UVgami Temp")
            for entry in inputs:
                for collection in entry.copy.users_collection:
                    collection.objects.unlink(entry.copy)
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

            manager.pieces_still_arriving += 1
            manager.start(engine, engine_ctx)
            tag_redraw()

            for level, text in prepared.reports:
                self.report({level}, text)

        except Exception as error:
            handle_start_error(error, start_objects)
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


def prepare_meshes(context, engine):
    props = context.scene.uvgami

    inputs = []
    input_sizes = []
    skipped = set()
    seen_mesh_data = set()
    applied_modifiers = False
    unapplied_modifiers = []
    depsgraph = context.evaluated_depsgraph_get()
    for obj in context.selected_objects:
        if obj.type != "MESH":
            skipped.add("non mesh objects")
            continue
        if len(obj.data.polygons) == 0:
            skipped.add("objects with zero polygons")
            continue
        if props.transfer_uvs and obj.data in seen_mesh_data:
            skipped.add("duplicates sharing a mesh")
            continue
        seen_mesh_data.add(obj.data)

        copy_object = obj.copy()
        copy_object.data = obj.data.copy()
        copy_object.animation_data_clear()

        obj.users_collection[0].objects.link(copy_object)

        evaluated = obj.evaluated_get(depsgraph)
        triangles = triangle_count(evaluated)
        input_sizes.append((obj.name, triangles))
        parts = part_triangles(copy_object.data) if engine.uses_proxy(props) else None
        proxied = parts is not None and needs_proxy(
            copy_object, props.proxy_faces, parts
        )
        if proxied or props.transfer_uvs:
            copy_object.modifiers.clear()
        else:
            applied, failed = _apply_modifiers(context, copy_object)
            applied_modifiers |= applied
            unapplied_modifiers += [f"{obj.name} {failure}" for failure in failed]

        inputs.append(MeshInput(copy_object, obj, obj.name, parts if proxied else None))

    reports = []
    if skipped:
        reports.append(("WARNING", f"Input contains {', '.join(sorted(skipped))}"))
    if applied_modifiers:
        reports.append(("WARNING", "Modifiers are applied to the unwrapped copy"))
    if unapplied_modifiers:
        reports.append(
            ("WARNING", f"Modifiers not applied: {', '.join(unapplied_modifiers)}")
        )

    return PreparedMeshes(inputs, reports, input_sizes)


def _apply_modifiers(context, obj):
    context.view_layer.objects.active = obj
    applied = False
    failed = []
    # applying one frees its entry in obj.modifiers
    names = [
        modifier.name for modifier in obj.modifiers if not is_shading_modifier(modifier)
    ]
    for name in names:
        # blender can't apply modifiers hidden in the viewport
        if not obj.modifiers[name].show_viewport:
            continue
        try:
            bpy.ops.object.modifier_apply(modifier=name)
        except RuntimeError as error:
            failed.append(f"{name}: {str(error).strip()}")
            continue
        applied = True
    return applied, failed
