import contextlib
import threading
import time
from collections import deque

import bpy
import numpy

from ..engines import active_engine
from ..handler import handle_error
from ..job import (
    HideInput,
    Join,
    Preserve,
    ProxyUVs,
    Result,
    Symmetrise,
    TransferUVs,
)
from ..logger import logger
from ..manager import manager
from ..progress_bar import progress_bar
from ..proxy import make_proxy
from ..seams import (
    face_edges,
    island_layout,
    islands_overlap,
    uv_island_groups,
)
from ..similar import find_twins, write_twin_output
from ..unwrap import Unwrap
from ..utils.geometry import apply_transforms, calc_center
from ..utils.io import export_obj
from ..utils.mesh import (
    check_collection,
    check_exists,
    deselect_all,
    move_to_collection,
    new_bmesh,
    set_bmesh,
    triangulate,
)
from ..utils.paths import (
    clear_io_dir,
    engine_file_stem,
    get_io_dir_paths,
    get_preferences,
)
from ..utils.ui import tag_redraw
from .guides import SEAM_RESTRICTIONS_GROUP

# process objects for at most this long per tick before yielding to the event loop
TICK_BUDGET = 0.033


def normalize_uvs(mesh):
    """The engine reads a mirrored island as inverted and stacked islands as
    self-intersecting, and re-cuts those charts, losing their seams. Mirror
    them back and lay the islands side by side, the output layout is the
    engine's repack either way."""
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

    # the grouping helpers walk plain lists, so convert in bulk once instead of
    # indexing the rna collections per loop
    vert_list = loop_verts.tolist()
    uv_list = coords.tolist()
    faces = []
    uvs = []
    for start, total in zip(starts.tolist(), totals.tolist()):
        faces.append(tuple(vert_list[start : start + total]))
        uvs.append(uv_list[start : start + total])

    groups = uv_island_groups(faces, uvs, face_edges(faces))

    # shoelace per face, each loop paired with the next one around its own face
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

    # nan marks an island that keeps its handedness, so it is left alone below
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
    """Writes separated objects to engine input files across timer ticks so the
    UI stays responsive."""

    def __init__(
        self,
        engine,
        engine_ctx,
        piece_unwrap,
        piece_has_uvs,
        separated_objects,
        start_objects,
        temp_collection,
    ):
        self.engine = engine
        self.engine_ctx = engine_ctx
        self.piece_unwrap = piece_unwrap
        self.piece_has_uvs = piece_has_uvs
        self.remaining = deque(separated_objects)
        self.start_objects = start_objects
        self.temp_collection = temp_collection

    def tick(self):
        # exports need object mode, the user may have entered edit mode between ticks
        if bpy.context.mode != "OBJECT":
            return 0.2

        # the session we joined was cancelled mid-export (Cancel All), so drop the
        # remaining pieces instead of exporting into a dead session
        if self.piece_unwrap and not manager.is_active:
            # release the hold first so a failure below can't wedge the manager
            manager.hold_count -= 1
            for obj in self.remaining:
                bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.collections.remove(self.temp_collection)
            return None

        try:
            props = bpy.context.scene.uvgami
            start = time.monotonic()
            while self.remaining:
                self._export_object(self.remaining.popleft(), props)
                if time.monotonic() - start >= TICK_BUDGET:
                    # yield to the event loop, resume next pass
                    return 0.0

            self._finish()
            return None

        except Exception as e:
            handle_error(e, "START", objects=self.start_objects)
            # handle_error leaves these alone during a live session
            for piece in list(self.temp_collection.objects):
                bpy.data.objects.remove(piece, do_unlink=True)
            bpy.data.collections.remove(self.temp_collection)
            manager.hold_count -= 1
            # settle the pieces that never got exported so their groups and
            # the session can still finish
            for unwrap in self.piece_unwrap.values():
                if unwrap.result is None and not unwrap.is_exported:
                    manager.record_result(unwrap, Result.INVALID)
            # if no session owns the bar (nothing added yet), tear it down
            if not manager.is_active:
                progress_bar.remove()
            return None

    def _export_object(self, obj, props):
        unwrap = self.piece_unwrap[obj]
        if unwrap.result is not None:
            # cancelled while waiting to export
            bpy.data.objects.remove(obj, do_unlink=True)
            return
        if unwrap.copy_of is not None:
            self._export_twin(obj, unwrap, props)
            return

        path = unwrap.path
        edge_path, new_edges = self._triangulate_mesh(obj, unwrap, path, props)

        # relink to the scene so matrix_world is evaluated for the world-space
        # export, all within one tick so no redraw shows the object
        bpy.context.scene.collection.objects.link(obj)
        bpy.context.view_layer.update()

        # seams and uvs were built before separation, see create_jobs
        has_uvs = self.piece_has_uvs[obj]
        if has_uvs:
            normalize_uvs(obj.data)
        vt_verts = export_obj(obj, path, has_uvs)

        guide_path = self._create_guide_file(obj, path, props, vt_verts)

        materials, material_indices, vertex_groups, shade_smooth = (
            self._get_mesh_metadata(obj)
        )

        unwrap.set_export_data(
            guide_path=guide_path,
            edge_path=edge_path,
            origin=obj.matrix_world.translation,
            materials=materials,
            added_edges=new_edges,
            vertex_count=len(obj.data.vertices),
            material_indices=material_indices,
            vertex_groups=vertex_groups,
            shade_smooth=shade_smooth,
        )

        bpy.data.objects.remove(obj, do_unlink=True)

    def _export_twin(self, obj, unwrap, props):
        """A twin writes no engine input, but its metadata must line up with
        the representative's output faces: an index-matched twin triangulates
        the same way, a reordered one reuses the representative's data
        outright since its own indices point at the wrong vertices."""
        representative = unwrap.copy_of
        if unwrap.copy_reordered:
            unwrap.set_export_data(
                edge_path=representative.edge_path,
                origin=representative.origin,
                materials=representative.materials,
                added_edges=representative.added_edges,
                vertex_count=representative.vertex_count,
                material_indices=representative.material_indices,
                vertex_groups=representative.vertex_groups,
                shade_smooth=representative.shade_smooth,
            )
        else:
            edge_path, new_edges = self._triangulate_mesh(
                obj, unwrap, unwrap.path, props
            )
            materials, material_indices, vertex_groups, shade_smooth = (
                self._get_mesh_metadata(obj)
            )
            unwrap.set_export_data(
                edge_path=edge_path,
                # pieces of one object share a transform, and the
                # representative exported before this twin
                origin=representative.origin,
                materials=materials,
                added_edges=new_edges,
                vertex_count=len(obj.data.vertices),
                material_indices=material_indices,
                vertex_groups=vertex_groups,
                shade_smooth=shade_smooth,
            )

        bpy.data.objects.remove(obj, do_unlink=True)

        # the representative finished before this twin was ready, settle now
        if representative.result is Result.FINISHED:
            write_twin_output(
                representative.output_path, unwrap.output_path, unwrap.copy_matrix
            )
            manager.record_result(unwrap, Result.FINISHED)

    def _finish(self):
        # a session with no valid pieces never started in _separate; start it
        # anyway so the empty run still finishes and reports
        if not manager.is_active:
            manager.engine = self.engine
            manager.engine_ctx = self.engine_ctx
            manager.start()
        bpy.data.collections.remove(self.temp_collection)
        manager.hold_count -= 1

    def _triangulate_mesh(self, obj, unwrap, path, props):
        """Triangulate the mesh if needed, tracking added edges for untriangulation."""
        new_edges = []
        bm = new_bmesh(obj)

        must_triangulate = False
        ngon_dict = {}
        for face_idx, face in enumerate(bm.faces):
            if len(face.edges) > 3:
                must_triangulate = True
                # n-gon vertices are only needed in full mode
                if props.maintain_mode == "PARTIAL":
                    break

            if len(face.edges) > 4:
                # found n-gon
                for vert in face.verts:
                    if vert.index not in ngon_dict:
                        ngon_dict[vert.index] = set()
                    ngon_dict[vert.index].add(face_idx)

        # the panel hides the setting on engines without preserve, but the value
        # persists, so check it here too: added edges are input vertex indices
        # and engines that renumber vertices would dissolve the wrong ones
        untriangulate = props.untriangulate and self.engine.supports_preserve

        edge_path = None
        if must_triangulate:
            if untriangulate:
                unwrap.preserve_job = Preserve()
                old_edges = set(bm.edges)

            triangulate(bm)

            if untriangulate:
                # write added edges to file
                edge_path = path.parent / f"{path.stem}_edges"
                with edge_path.open("w") as f:
                    for bm_e in set(bm.edges).difference(old_edges):
                        edge = (bm_e.verts[0].index, bm_e.verts[1].index)
                        if (
                            # if both vertices are ngon vertices
                            edge[0] in ngon_dict
                            and edge[1] in ngon_dict
                            # and they are from the same ngon
                            and len(ngon_dict[edge[0]].intersection(ngon_dict[edge[1]]))
                            > 0
                        ):
                            # edge is inside ngon, don't dissolve
                            # because ngons aren't rerouted
                            continue
                        new_edges.append(edge)
                        f.write(f"{edge[0]} {edge[1]}\n")

            set_bmesh(bm, obj)
        else:
            bm.free()

        return edge_path, new_edges

    def _create_guide_file(self, obj, path, props, vt_verts):
        """Write the per-vertex weight sidecars from the restriction group.
        _weights repels seams, _importance protects faces from stretching."""
        weights = {}
        if (
            self.engine.supports_guided
            and (props.avoid_seams or props.reduce_stretching)
            and SEAM_RESTRICTIONS_GROUP in obj.vertex_groups
        ):
            group_idx = obj.vertex_groups[SEAM_RESTRICTIONS_GROUP].index
            for v in obj.data.vertices:
                for g in v.groups:
                    if g.group == group_idx:
                        weights[v.index] = g.weight
                        break

        if not weights:
            return None

        if vt_verts is not None:
            # optcuts rebuilds a UV-carrying obj with one vertex per vt, so
            # vertex-indexed weights would land on the wrong vertices
            weights = {
                vt: weights[v] for vt, v in enumerate(vt_verts.tolist()) if v in weights
            }

        guide = ",".join(f"{index},{weight}" for index, weight in weights.items())
        guide_path = None
        if props.avoid_seams:
            guide_path = path.parent / f"{path.stem}_weights"
            with guide_path.open("w") as f:
                f.write(f"{guide}\n")
        if props.reduce_stretching:
            importance_path = path.parent / f"{path.stem}_importance"
            with importance_path.open("w") as f:
                f.write(f"{guide}\n")

        return guide_path

    def _get_mesh_metadata(self, obj):
        """Gather materials and shading info from the mesh."""
        # empty slots are kept as None so the per-face indices below still
        # point at the same material once the output rebuilds the slots
        materials = [
            slot.material.name if slot.material else None for slot in obj.material_slots
        ]

        # per-face indices, so they can be restored after import
        material_indices = [0] * len(obj.data.polygons)
        obj.data.polygons.foreach_get("material_index", material_indices)

        shade_smooth = obj.data.polygons[0].use_smooth

        vertex_groups = {}
        for group in obj.vertex_groups:
            weights = {}
            for v in obj.data.vertices:
                for g in v.groups:
                    if g.group == group.index:
                        weights[v.index] = g.weight
                        break
            vertex_groups[group.name] = weights

        return materials, material_indices, vertex_groups, shade_smooth


def input_job(props):
    """The job that finishes an unwrap against the original input mesh."""
    if props.use_proxy:
        return ProxyUVs(props.transfer_uvs)
    if props.transfer_uvs:
        return TransferUVs()
    return None


class SessionBuilder:
    """Runs each object's preseed in a worker thread, then separates it and
    hands every piece to an InputExporter, all across timer ticks so the UI
    stays live through the slow part."""

    def __init__(
        self,
        engine,
        engine_ctx,
        input_path,
        names,
        input_objs,
        objects,
        start_objects,
        temp_collection,
    ):
        self.engine = engine
        self.engine_ctx = engine_ctx
        self.input_path = input_path
        self.names = names
        self.input_objs = input_objs
        self.remaining = deque(enumerate(objects))
        self.start_objects = start_objects
        self.temp_collection = temp_collection
        self.separated_objects = []
        self.piece_unwrap = {}
        self.piece_has_uvs = {}
        self.pending = None
        # queue ui placeholders until each object's pieces exist
        self.preparing = [names[obj.name][0] for _, obj in self.remaining]
        manager.preparing.extend(self.preparing)

    def tick(self):
        # separation and uv writes need object mode
        if bpy.context.mode != "OBJECT":
            return 0.2
        try:
            return self._advance()
        except Exception as e:
            handle_error(e, "START", objects=self.start_objects)
            # an undo past the session start kills the collection datablock,
            # and a second ReferenceError here would wedge the hold count
            if check_exists(self.temp_collection):
                # handle_error leaves these alone during a live session
                for piece in list(self.temp_collection.objects):
                    bpy.data.objects.remove(piece, do_unlink=True)
                bpy.data.collections.remove(self.temp_collection)
            manager.hold_count -= 1
            for name in self.preparing:
                manager.preparing.remove(name)
            self.preparing.clear()
            # settle the pieces already added, none of them exported yet
            for unwrap in self.piece_unwrap.values():
                if unwrap.result is None:
                    manager.record_result(unwrap, Result.INVALID)
            if not manager.is_active:
                progress_bar.remove()
            return None

    def _advance(self):
        if self.pending is not None:
            thread, box, apply, obj, index, symmetrize_job = self.pending
            if thread.is_alive():
                return 0.1
            self.pending = None
            if "error" in box:
                raise box["error"]
            if not check_exists(obj):
                raise RuntimeError("Undo removed the working copy mid unwrap")
            props = bpy.context.scene.uvgami
            preseeded = apply(box.get("result"))
            if symmetrize_job is not None:
                symmetrize_job.cut(obj)
            has_uvs = preseeded or (
                props.import_uvs and self.engine.supports_import_uvs
            )
            self._separate(obj, index, has_uvs, symmetrize_job, preseeded=preseeded)
            return 0.0
        if not self.remaining:
            return self._finish()

        index, obj = self.remaining.popleft()
        props = bpy.context.scene.uvgami
        symmetrize_job = None
        if props.use_symmetry:
            apply_transforms(obj)
            symmetrize_job = Symmetrise(
                props.sym_axes, calc_center(obj), props.sym_merge
            )

        # seams and uvs are built on the whole mesh, before the symmetry cut
        # and separation: the seams package reads region widths off the full
        # model, a bisected half merges its regions away, and a small loose
        # part run alone shatters (auto width tunes to the piece)
        work = self.engine.preseed_work(obj, props)
        if work is None:
            has_uvs = self.engine.prepare_uvs(obj, props)
            if symmetrize_job is not None:
                symmetrize_job.cut(obj)
            self._separate(obj, index, has_uvs, symmetrize_job)
            return 0.0
        compute, apply = work
        box = {}

        def run():
            try:
                box["result"] = compute()
            except BaseException as error:  # rethrown on the main thread
                box["error"] = error

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self.pending = (thread, box, apply, obj, index, symmetrize_job)
        return 0.1

    def _input_jobs(self, props, index):
        """The hide or transfer job that finishes against the input mesh."""
        transfer_uvs_job = input_job(props)
        hide_job = None
        if transfer_uvs_job is not None:
            manager.input[transfer_uvs_job] = self.input_objs[index]
        else:
            # the hide job can come after join because it doesn't depend
            # on the unwrapped objects
            hide_job = HideInput()
            manager.input[hide_job] = self.input_objs[index]
        return hide_job, transfer_uvs_job

    def _add_piece(self, obj, input_name, piece_name, jobs, has_uvs, props, preseeded):
        """Create the piece's session record before its input file exists, so
        cancels and the queue ui see the whole session upfront."""
        path = self.input_path / f"{engine_file_stem(piece_name)}.obj"
        # names can repeat across pieces and session extends, and the output
        # file is keyed by stem, so claim a stem no other unwrap holds
        claimed = {u.path for u in manager.active}
        claimed.update(u.path for u, _ in manager.results)
        while path.is_file() or path in claimed:
            path = path.parent / f"{path.stem}1.obj"

        uses_uvs = (
            self.engine.piece_uses_uvs(obj, props, has_uvs)
            and obj.data.uv_layers.active is not None
        )
        unwrap = Unwrap(
            name=piece_name,
            input_name=input_name,
            path=path,
            jobs=jobs,
            maintain_mode=props.maintain_mode,
            preseeded=preseeded and uses_uvs,
        )
        self.piece_unwrap[obj] = unwrap
        self.piece_has_uvs[obj] = uses_uvs
        manager.add(unwrap)

    def _separate(self, obj, index, has_uvs, symmetrize_job, preseeded=False):
        props = bpy.context.scene.uvgami
        # relink for the ops selection, all within one tick so nothing shows
        bpy.context.scene.collection.objects.link(obj)
        self.temp_collection.objects.unlink(obj)
        bpy.context.view_layer.update()
        deselect_all()
        obj.select_set(True)

        bpy.ops.mesh.separate(type="LOOSE")
        s = bpy.context.selected_objects
        added = []
        if len(s) > 1:
            unwrap_name = self.names[obj.name][0]
            valid = []
            for o in s:
                # check for 0 polygons again
                if len(o.data.polygons) == 0:
                    collection = check_collection(
                        "UVgami Not Unwrapped", bpy.context.scene.collection
                    )
                    move_to_collection(o, collection)
                    o.name = f"{unwrap_name}: No Polygons"
                else:
                    valid.append(o)

            join_job = Join(len(valid))
            hide_job, transfer_uvs_job = self._input_jobs(props, index)
            for obj_idx, o in enumerate(valid):
                jobs = (None, join_job, hide_job, symmetrize_job, transfer_uvs_job)
                piece_name = f"{unwrap_name}_{obj_idx + 1}"
                self._add_piece(
                    o, unwrap_name, piece_name, jobs, has_uvs, props, preseeded
                )
                added.append(o)
            if props.stack_similar:
                # the new pieces' matrix_world needs an evaluation first
                bpy.context.view_layer.update()
                # a representative always precedes its twins in valid order,
                # so it exports and settles first
                for twin_obj, (rep_obj, matrix, exact) in find_twins(valid).items():
                    twin = self.piece_unwrap[twin_obj]
                    representative = self.piece_unwrap[rep_obj]
                    twin.copy_of = representative
                    twin.copy_matrix = matrix
                    twin.copy_reordered = not exact
                    representative.twins.append(twin)
        else:
            # object didn't need to be separated
            unwrap_name = self.names[obj.name][0]
            hide_job, transfer_uvs_job = self._input_jobs(props, index)
            jobs = (None, None, hide_job, symmetrize_job, transfer_uvs_job)
            self._add_piece(
                obj, unwrap_name, unwrap_name, jobs, has_uvs, props, preseeded
            )
            added.append(obj)

        if unwrap_name in self.preparing:
            self.preparing.remove(unwrap_name)
            manager.preparing.remove(unwrap_name)

        deselect_all()
        for piece in added:
            self.separated_objects.append(piece)
            for coll in piece.users_collection:
                coll.objects.unlink(piece)
            self.temp_collection.objects.link(piece)

        # the pieces are queued, start the session so the bar and the queue ui
        # track them while the export catches up
        if added and not manager.is_active:
            manager.engine = self.engine
            manager.engine_ctx = self.engine_ctx
            manager.start()

    def _finish(self):
        exporter = InputExporter(
            engine=self.engine,
            engine_ctx=self.engine_ctx,
            piece_unwrap=self.piece_unwrap,
            piece_has_uvs=self.piece_has_uvs,
            separated_objects=self.separated_objects,
            start_objects=self.start_objects,
            temp_collection=self.temp_collection,
        )
        # the exporter takes over the session hold this builder was carrying
        bpy.app.timers.register(exporter.tick)
        return None


class UVGAMI_OT_start(bpy.types.Operator):
    bl_idname = "uvgami.start"
    bl_label = "Unwrap"
    bl_description = "Start UV unwrap process"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_variables()

    def reset_variables(self):
        self.engine = None
        self.engine_ctx = None
        self.input_path = None

        self.input_objs = None

        self.objects = None
        self.names = None
        self.reports = []

        self.temp_collection = None

    def execute(self, context):
        start_objects = set(bpy.data.objects)
        builder_registered = False

        try:
            logger.new_info()
            self.engine = active_engine(context.scene.uvgami.engine)
            if self.engine is None:
                self.report({"ERROR"}, "No engine installed")
                return {"CANCELLED"}

            # the manager holds one engine for the whole session, so pieces added
            # to a running session would export for this engine and run on the old
            if manager.is_active and manager.engine is not self.engine:
                self.report(
                    {"ERROR"},
                    "Finish or cancel the current unwrap before switching engine",
                )
                return {"CANCELLED"}

            if self.check_for_errors() is not None:
                return {"CANCELLED"}
            if self._prepare_unwrap_session(context) is not None:
                return {"CANCELLED"}

            builder = SessionBuilder(
                engine=self.engine,
                engine_ctx=self.engine_ctx,
                input_path=self.input_path,
                names=self.names,
                input_objs=self.input_objs,
                objects=self.objects,
                start_objects=start_objects,
                temp_collection=self.temp_collection,
            )
            bpy.app.timers.register(builder.tick)
            builder_registered = True
            # the builder, then the exporter it hands off to, hold the session
            # open until every piece is added
            manager.hold_count += 1

            # show the progress bar now instead of after every piece exports
            if get_preferences().show_progress_bar:
                progress_bar.start()
            tag_redraw()

            for level, text in self.reports:
                self.report({level}, text)

        except Exception as e:
            handle_error(e, "START", objects=start_objects)
            # tick() owns the collection once registered; only remove it here
            # if the builder never started ticking
            if not builder_registered and self.temp_collection is not None:
                bpy.data.collections.remove(self.temp_collection)

        # these variables should only be used while operator is running
        self.reset_variables()
        return {"FINISHED"}

    def _prepare_unwrap_session(self, context):
        # the builder's separation ops need object mode, and leaving edit mode
        # flushes the mesh so the copies below don't take pre-edit geometry
        active = context.active_object
        if active is not None and active.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        self.input_objs = context.selected_objects
        self.objects, self.names, self.reports = self.prepare_meshes(context)
        if len(self.objects) == 0:
            # there are no valid meshes
            reason = self.reports[0][1] if self.reports else "No object selected"
            self.report({"ERROR"}, reason)
            return {"CANCELLED"}

        self.input_path, _ = self.prepare_io_folders()
        deselect_all()

        # stash the copies in a collection not linked to the scene so they
        # don't flash in the viewport or outliner between builder ticks
        self.temp_collection = bpy.data.collections.new("UVgami Temp")
        for obj in self.objects:
            for coll in obj.users_collection:
                coll.objects.unlink(obj)
            self.temp_collection.objects.link(obj)
        return None

    def check_for_errors(self):
        prefs = get_preferences()

        if self._run_autosave_check(prefs) is not None:
            return {"CANCELLED"}

        engine_ctx, error = self.engine.validate(prefs)
        if error is not None:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}
        self.engine_ctx = engine_ctx

        return None

    def _run_autosave_check(self, prefs):
        if not prefs.autosave:
            return None

        if bpy.data.is_saved:
            bpy.ops.wm.save_mainfile()
            return None

        bpy.ops.wm.save_as_mainfile("INVOKE_DEFAULT")
        self.report(
            {"WARNING"},
            "Autosave is turned on. Save the file before starting UVgami",
        )
        return {"CANCELLED"}

    def prepare_meshes(self, context):
        props = context.scene.uvgami

        objects = []
        names = {}
        skipped = set()
        applied_modifiers = False
        warn = get_preferences().show_warnings
        # the result comes from the input mesh itself, so the engine has to see
        # that mesh and not a modifier bake of it
        keep_modifiers = input_job(props) is not None
        for obj in self.input_objs:
            if obj.type != "MESH":
                skipped.add("non mesh objects")
                continue
            if len(obj.data.polygons) == 0:
                skipped.add("objects with zero polygons")
                continue

            # unlinked duplicate, so the original is never touched
            copy_object = obj.copy()
            copy_object.data = obj.data.copy()
            copy_object.animation_data_clear()

            obj.users_collection[0].objects.link(copy_object)

            if keep_modifiers:
                copy_object.modifiers.clear()
            elif self._apply_modifiers(context, copy_object):
                applied_modifiers = True

            if props.use_proxy:
                make_proxy(copy_object, props.proxy_faces)

            # save name, format: input name, unwrap name
            names[copy_object.name] = [obj.name, obj.name]
            objects.append(copy_object)

        reports = []
        if skipped:
            reports.append(("WARNING", f"Input contains {', '.join(sorted(skipped))}"))
        if warn and applied_modifiers:
            reports.append(("INFO", "Modifiers were applied to the unwrapped copy"))

        return objects, names, reports

    def _apply_modifiers(self, context, obj):
        """True when anything was baked into the copy."""
        context.view_layer.objects.active = obj
        applied = False
        for modifier in obj.modifiers:
            if "Smooth by Angle" in modifier.name:
                # don't apply auto smooth modifier
                continue

            # a disabled modifier can't be applied
            with contextlib.suppress(RuntimeError):
                bpy.ops.object.modifier_apply(modifier=modifier.name)
                applied = True
        return applied

    def prepare_io_folders(self):
        input_path, output_path = get_io_dir_paths()
        if not manager.is_active:
            clear_io_dir(input_path)
            clear_io_dir(output_path)

        return input_path, output_path
