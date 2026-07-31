import time
from collections import deque

import bmesh
import bpy
import numpy

from ..engines import get_engine
from ..handler import handle_error
from ..job import HideInput, Join, Preserve, ProxyUVs, Symmetrise, TransferUVs
from ..logger import logger
from ..manager import manager
from ..objfile import remap_weights_to_vt
from ..progress_bar import progress_bar
from ..proxy import make_proxy
from ..unwrap import Unwrap
from ..utils.geometry import apply_transforms, calc_center, cut, cut_on_axes
from ..utils.io import export_obj
from ..utils.mesh import (
    check_collection,
    deselect_all,
    move_to_collection,
    new_bmesh,
    set_bmesh,
    triangulate,
)
from ..utils.paths import get_extension_dir_path, get_preferences
from ..utils.ui import tag_redraw
from .guides import SEAM_RESTRICTIONS_GROUP

# process objects for at most this long per tick before yielding to the event loop
TICK_BUDGET = 0.033


class InputExporter:
    """Writes separated objects to engine input files across timer ticks so the
    UI stays responsive."""

    def __init__(
        self,
        engine,
        engine_ctx,
        input_path,
        names,
        jobs,
        piece_has_uvs,
        separated_objects,
        old_active,
        old_mode,
        start_objects,
        temp_collection,
    ):
        self.engine = engine
        self.engine_ctx = engine_ctx
        self.input_path = input_path
        self.names = names
        self.jobs = jobs
        self.piece_has_uvs = piece_has_uvs
        self.remaining = deque(separated_objects)
        self.old_active = old_active
        self.old_mode = old_mode
        self.start_objects = start_objects
        self.temp_collection = temp_collection
        self.added_any = False

    def tick(self):
        # exports need object mode, the user may have entered edit mode between ticks
        if bpy.context.mode != "OBJECT":
            return 0.2

        # the session we joined was cancelled mid-export (Cancel All), so drop the
        # remaining pieces instead of exporting into a dead session
        if self.added_any and not manager.is_active:
            # release the hold first so a failure below can't wedge the manager
            manager.hold_count -= 1
            manager.pending_count -= len(self.remaining)
            for obj in self.remaining:
                bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.collections.remove(self.temp_collection)
            bpy.context.view_layer.objects.active = self.old_active
            bpy.ops.object.mode_set(mode=self.old_mode)
            return None

        try:
            props = bpy.context.scene.uvgami
            start = time.monotonic()
            while self.remaining:
                self._export_object(self.remaining.popleft(), props)
                if time.monotonic() - start >= TICK_BUDGET:
                    # yield to the event loop, resume next pass
                    return 0.0

            self._finish(props)
            return None

        except Exception as e:
            # handle_error removes objects created since start, incl the temp
            # pieces, so only the collection data-block is left to remove
            handle_error(e, "START", objects=self.start_objects)
            bpy.data.collections.remove(self.temp_collection)
            manager.hold_count -= 1
            # return the pieces that never made it into the session
            manager.pending_count -= len(self.remaining)
            # if no session owns the bar (nothing added yet), tear it down
            if not manager.is_active:
                progress_bar.remove()
            return None

    def _export_object(self, obj, props):
        join = self.jobs[obj]["join"]
        if join is not None and join.cancel_requested:
            # whole-group cancel: drop the piece instead of exporting it
            self._skip_piece(obj)
            return

        # consume upfront so a failure below can't double count this piece,
        # the error path only subtracts what's still in the queue
        manager.pending_count -= 1
        # get unwrap name
        unwrap_name = self.names[obj.name][1]
        path = self.input_path / f"{bpy.path.clean_name(unwrap_name)}.obj"
        # if path to file already exists, find a unique name
        while path.is_file():
            path = path.parent / (f"{path.stem}1.obj")

        edge_path, new_edges = self._triangulate_mesh(obj, path, props)

        # relink to the scene so select_set works, objects not in the view layer
        # can't be selected, all within one tick so no redraw shows the object
        bpy.context.scene.collection.objects.link(obj)
        bpy.context.view_layer.update()

        # export uses selected-objects mode and the user can change selection between
        # ticks, deselect so only this object lands in the obj file
        deselect_all()
        # seams and uvs were built before separation, see create_jobs
        has_uvs = self.piece_has_uvs[obj]
        export_obj(obj, path, has_uvs)

        guide_path = self._create_guide_file(obj, path, props, has_uvs)

        materials, material_indices, vertex_groups, shade_smooth = (
            self._get_mesh_metadata(obj)
        )

        unwrap = Unwrap(
            name=unwrap_name,
            input_name=self.names[obj.name][0],
            path=path,
            guide_path=guide_path,
            edge_path=edge_path,
            jobs=(
                self.jobs[obj]["preserve"],
                self.jobs[obj]["join"],
                self.jobs[obj]["hide"],
                self.jobs[obj]["symmetrize"],
                self.jobs[obj]["transfer_uvs"],
            ),
            origin=obj.matrix_world.translation,
            materials=materials,
            added_edges=new_edges,
            vertex_count=len(obj.data.vertices),
            material_indices=material_indices,
            vertex_groups=vertex_groups,
            shade_smooth=shade_smooth,
            merge_cuts=props.use_cuts and not props.use_symmetry,
            maintain_mode=props.maintain_mode,
        )
        manager.add(unwrap)
        if not manager.is_active:
            manager.engine = self.engine
            manager.engine_ctx = self.engine_ctx
            manager.start()
            # start() only counted the queue; include still-pending pieces
            # in the bar total (the active case was counted in execute)
            manager.starting_count += manager.pending_count
        self.added_any = True

        bpy.data.objects.remove(obj, do_unlink=True)

    def _skip_piece(self, obj):
        # release each job so the group's finished runners still satisfy
        # is_completed() and get imported
        jobs = [j for j in self.jobs[obj].values() if j is not None]
        manager.release_jobs(jobs)
        bpy.data.objects.remove(obj, do_unlink=True)
        # drop it from the bar denominator
        manager.pending_count -= 1
        manager.starting_count -= 1

    def _finish(self, props):
        # pieces were added as they exported; only start here if none were
        # (e.g. every piece had zero polygons)
        if not manager.is_active:
            manager.engine = self.engine
            manager.engine_ctx = self.engine_ctx
            manager.start()
        bpy.context.view_layer.objects.active = self.old_active
        bpy.ops.object.mode_set(mode=self.old_mode)
        bpy.data.collections.remove(self.temp_collection)
        manager.hold_count -= 1

    def _triangulate_mesh(self, obj, path, props):
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
                self.jobs[obj]["preserve"] = Preserve(1)
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

    def _create_guide_file(self, obj, path, props, has_uvs):
        """Write the per-vertex seam weight file from the painted guide.
        Higher weight repels seams."""
        weights = {}
        if (
            self.engine.supports_guided
            and props.use_guided_mode
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

        if has_uvs:
            # optcuts rebuilds a UV-carrying obj with one vertex per vt, so
            # vertex-indexed weights would land on the wrong vertices
            weights = remap_weights_to_vt(path, weights)

        guide = ",".join(f"{index},{weight}" for index, weight in weights.items())
        guide_path = path.parent / f"{path.stem}_weights"
        with guide_path.open("w") as f:
            f.write(f"{guide}\n")

        return guide_path

    def _get_mesh_metadata(self, obj):
        """Gather materials and shading info from the mesh."""
        # get materials
        materials = [slot.material.name for slot in obj.material_slots if slot.material]

        # get per-face material indices so they can be restored after import
        material_indices = [0] * len(obj.data.polygons)
        obj.data.polygons.foreach_get("material_index", material_indices)

        # check smooth shading
        shade_smooth = True if obj.data.polygons[0].use_smooth else False

        # get vertex groups
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

        self.old_active = None
        self.old_mode = None
        self.input_objs = None

        self.objects = None
        self.names = None
        self.report_msg = None

        self.separated_objects = None
        self.jobs = None
        self.piece_has_uvs = None
        self.temp_collection = None

    def execute(self, context):
        start_objects = set(bpy.data.objects)
        exporter_registered = False

        try:
            logger.new_info()
            self.engine = get_engine(context.scene.uvgami.engine)

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

            exporter = InputExporter(
                engine=self.engine,
                engine_ctx=self.engine_ctx,
                input_path=self.input_path,
                names=self.names,
                jobs=self.jobs,
                piece_has_uvs=self.piece_has_uvs,
                separated_objects=self.separated_objects,
                old_active=self.old_active,
                old_mode=self.old_mode,
                start_objects=start_objects,
                temp_collection=self.temp_collection,
            )
            bpy.app.timers.register(exporter.tick)
            exporter_registered = True
            # the exporter holds the session open until it finishes adding pieces
            manager.hold_count += 1
            # count every piece in the bar total upfront so the finished
            # ratio doesn't shrink as pieces get added
            manager.pending_count += len(self.separated_objects)
            if manager.is_active:
                manager.starting_count += len(self.separated_objects)

            # show the progress bar now instead of after every piece exports
            if get_preferences().show_progress_bar:
                progress_bar.start()
            tag_redraw()

            if self.report_msg == "Input contain":
                self.report({"INFO"}, "UV unwrap in progress")
            else:
                self.report({"WARNING"}, f"UV unwrap in progress. {self.report_msg}")

        except Exception as e:
            handle_error(e, "START", objects=start_objects)
            # tick() owns the collection once registered; only remove it here
            # if the exporter never started ticking
            if not exporter_registered and self.temp_collection is not None:
                bpy.data.collections.remove(self.temp_collection)

        # these variables should only be used while operator is running
        self.reset_variables()
        return {"FINISHED"}

    def _prepare_unwrap_session(self, context):
        self.old_active = context.active_object
        self.input_objs = context.selected_objects

        # check if there is an active object selected
        if not (self.old_active and self.old_active in self.input_objs):
            self.report({"ERROR"}, "No active object selected")
            return {"CANCELLED"}

        self.old_mode = self.old_active.mode
        self.objects, self.names, self.report_msg = self.prepare_meshes(context)
        if len(self.objects) == 0:
            # there are no valid meshes
            self.report({"ERROR"}, self.report_msg)
            return {"CANCELLED"}

        self.input_path, _ = self.prepare_io_folders()
        self.jobs, self.separated_objects, self.piece_has_uvs = self.create_jobs(
            context
        )
        deselect_all()

        # stash pieces in a collection not linked to the scene so they don't flash
        # in the viewport or outliner between export ticks
        self.temp_collection = bpy.data.collections.new("UVgami Temp")
        for obj in self.separated_objects:
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
        messages = [False, False]
        for obj in self.input_objs:
            if obj.type != "MESH":
                messages[0] = True
                continue
            if len(obj.data.polygons) == 0:
                messages[1] = True
                continue

            # make unlinked duplicate of object
            copy_object = obj.copy()
            copy_object.data = obj.data.copy()
            copy_object.animation_data_clear()

            # link to scene
            object_collection = obj.users_collection[0]
            object_collection.objects.link(copy_object)

            self._apply_modifiers(context, copy_object)
            if props.use_proxy:
                make_proxy(copy_object, props.proxy_faces)
            self._apply_cuts_if_needed(copy_object, obj, props)

            # save name, format: input name, unwrap name
            names[copy_object.name] = [obj.name, obj.name]
            objects.append(copy_object)

        report_msg = "Input contains"
        if messages[0]:
            report_msg += " non mesh objects,"
        if messages[1]:
            report_msg += " objects with zero polygons "
        # remove comma or space at end
        report_msg = report_msg[:-1]

        return objects, names, report_msg

    def _apply_modifiers(self, context, obj):
        context.view_layer.objects.active = obj
        for modifier in obj.modifiers:
            if "Smooth by Angle" in modifier.name:
                # don't apply auto smooth modifier
                continue

            try:
                bpy.ops.object.modifier_apply(modifier=modifier.name)
            except RuntimeError:
                # if the modifier is disabled, don't apply
                pass

    def _apply_cuts_if_needed(self, target_obj, source_obj, props):
        if not (props.use_cuts and not props.use_symmetry):
            return

        bm = new_bmesh(target_obj)
        if props.cut_type == "EVEN":
            self._apply_even_cuts(source_obj, target_obj, bm, props)
        else:
            self._apply_seam_cuts(source_obj, bm)
        set_bmesh(bm, target_obj)

    def _apply_even_cuts(self, source_obj, target_obj, bm, props):
        # make even cuts on axes
        apply_transforms(target_obj)

        axes = props.cut_axes
        cuts = props.cuts

        axis_count = len(axes) if len(axes) != 0 else 3
        d = cuts // axis_count
        r = cuts % axis_count

        # distribute cuts
        x_num = d if r == 0 else d + 1
        y_num = d if r != 2 else d + 1
        z_num = d

        center = calc_center(source_obj)
        if not axes or "X" in axes:
            cut(x_num, center, target_obj.dimensions.x, 0, bm)
        if not axes or "Y" in axes:
            cut(y_num, center, target_obj.dimensions.y, 1, bm)
        if not axes or "Z" in axes:
            cut(z_num, center, target_obj.dimensions.z, 2, bm)

    def _apply_seam_cuts(self, source_obj, bm):
        seams = numpy.zeros(len(bm.edges), dtype=bool)
        source_obj.data.edges.foreach_get("use_seam", seams)
        bm_seams = numpy.array(bm.edges)[seams]
        bmesh.ops.split_edges(bm, edges=bm_seams)

    def prepare_io_folders(self):
        input_path = get_extension_dir_path() / "input"
        input_path.mkdir(exist_ok=True)
        output_path = input_path.parent / "output"
        output_path.mkdir(exist_ok=True)
        # io folder clean up
        if not manager.is_active:
            for file in input_path.iterdir():
                file.unlink()
            for file in output_path.iterdir():
                file.unlink()

        return input_path, output_path

    def _input_job(self, props, count):
        """The job that finishes an unwrap against the original input mesh."""
        if props.use_proxy:
            return ProxyUVs(count)
        if props.transfer_uvs:
            return TransferUVs(count)
        return None

    def create_jobs(self, context):
        props = context.scene.uvgami

        jobs = {}
        separated_objects = []
        piece_has_uvs = {}

        # objects can't be in edit mode
        context.view_layer.objects.active = self.old_active
        if self.old_mode == "EDIT":
            bpy.ops.object.mode_set(mode="OBJECT")

        for object_idx, obj in enumerate(self.objects):
            deselect_all()
            obj.select_set(True)

            symmetrize_job = None
            if props.use_symmetry:
                # bisect if symmetry on
                axes = props.sym_axes
                apply_transforms(obj)
                obj_center = calc_center(obj)
                symmetrize_job = Symmetrise(1, axes, obj_center, props.sym_merge)
                cut_on_axes(obj, obj_center, axes)

            # seams and uvs are built on the whole mesh before separation:
            # the seams package reads region widths off the full model, and a small
            # loose part run alone shatters (auto width tunes to the piece)
            has_uvs = self.engine.prepare_uvs(obj, props)
            deselect_all()
            obj.select_set(True)

            # separate objects
            bpy.ops.mesh.separate(type="LOOSE")
            s = context.selected_objects
            if len(s) > 1:
                # get input name
                unwrap_name = self.names[obj.name][0]
                join_job = Join(len(s))
                hide_job = None
                transfer_uvs_job = self._input_job(props, len(s))

                if transfer_uvs_job is not None:
                    manager.input[transfer_uvs_job] = self.input_objs[object_idx]
                else:
                    # the hide job can come after join because it doesn't depend
                    # on the unwrapped objects
                    # the count is > 1 because all the separated objs need to
                    # finish before hiding the original
                    hide_job = HideInput(len(s))
                    manager.input[hide_job] = self.input_objs[object_idx]

                for obj_idx, o in enumerate(s):
                    # check for 0 polygons again
                    if len(o.data.polygons) == 0:
                        join_job.count -= 1
                        if hide_job is not None:
                            hide_job.count -= 1
                        if transfer_uvs_job is not None:
                            transfer_uvs_job.count -= 1
                        collection = check_collection(
                            "UVgami Not Unwrapped", context.scene.collection
                        )
                        move_to_collection(o, collection)
                        o.name = f"{unwrap_name}: No Polygons"
                    else:
                        # add ids to separated objects
                        jobs[o] = {
                            "join": join_job,
                            "preserve": None,
                            "hide": hide_job,
                            "symmetrize": symmetrize_job,
                            "transfer_uvs": transfer_uvs_job,
                        }
                        piece_has_uvs[o] = self.engine.piece_uses_uvs(
                            o, props, has_uvs
                        )
                        separated_objects.append(o)
                        self.names[o.name] = [
                            unwrap_name,
                            f"{unwrap_name}_{obj_idx + 1}",
                        ]
            else:
                # object didn't need to be separated
                jobs[obj] = {
                    "join": None,
                    "preserve": None,
                    "hide": None,
                    "symmetrize": symmetrize_job,
                    "transfer_uvs": None,
                }
                transfer_uvs_job = self._input_job(props, 1)
                if transfer_uvs_job is not None:
                    jobs[obj]["transfer_uvs"] = transfer_uvs_job
                    manager.input[transfer_uvs_job] = self.input_objs[object_idx]
                else:
                    hide_job = HideInput(1)
                    jobs[obj]["hide"] = hide_job
                    manager.input[hide_job] = self.input_objs[object_idx]
                piece_has_uvs[obj] = self.engine.piece_uses_uvs(obj, props, has_uvs)
                separated_objects.append(obj)

        return jobs, separated_objects, piece_has_uvs
