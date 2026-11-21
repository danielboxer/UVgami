from collections import namedtuple

import bpy
import numpy
from bpy_extras.view3d_utils import region_2d_to_origin_3d, region_2d_to_vector_3d
from mathutils.bvhtree import BVHTree

from ..engines import active_engine
from ..engines.optcuts import SURFACE_MODES
from ..no_bpy import SURFACE_MODE_CODES, EngineError
from ..no_bpy.mesh import face_starts, lowest_index_part_labels
from ..no_bpy.preseed import SegmentRegions, classify_job
from ..no_bpy.worker import WorkerError, WorkerProcess
from ..part_overrides import (
    KEEP_ATTRIBUTE,
    OVERRIDE_ATTRIBUTES,
    SURFACE_MODE_ATTRIBUTE,
    face_values,
    surface_modes,
)
from ..proxy import (
    auto_proxy_triangle_counts,
    part_triangles,
    proxy_trigger_faces,
)
from ..segmentation import REGION_ATTRIBUTE, clear_regions, segment_work
from ..symmetry import mesh_mirror_partners, mirror_segmentation, paints_seams
from ..unwrap.job import start_unwrap_symmetry
from ..unwrap.manager import SettingsCopy, manager
from ..utils.mesh import (
    applied_matrix,
    check_exists,
    face_vertices,
    lifted_triangles,
    loop_vertices_and_face_sizes,
    mesh_arrays,
    validate_obj,
    vertex_position_array,
    world_positions,
)
from ..utils.paths import get_extension_dir_path
from ..utils.ui import count_label, draw_status_hints, tag_redraw
from ..utils.uvgami_tools import tools_command
from .plan_drawing import (
    FILL_NUDGE,
    Shape,
    draw_batches,
    plane_shapes,
    shape_batch,
)
from .segmentation import RegionEditor

PICKED_COLOR = (1.0, 0.9, 0.1, 0.45)
HARD_COLOR = (0.3, 0.3, 0.3, 0.5)
# no grey, which means hard surface, and no yellow, which means selected
REGION_COLORS = (
    (0.9, 0.3, 0.6, 0.5),
    (0.2, 0.8, 0.85, 0.5),
    (0.6, 0.4, 0.95, 0.5),
    (0.3, 0.8, 0.3, 0.5),
    (0.95, 0.5, 0.1, 0.5),
    (0.25, 0.45, 0.95, 0.5),
    (0.7, 0.45, 0.25, 0.5),
    (0.9, 0.2, 0.2, 0.5),
)
HOVERED_PART_COLOR = (1.0, 0.9, 0.1, 0.25)
TIMER_SECONDS = 0.1
ORGANIC = SURFACE_MODE_CODES["ORGANIC"]
HARD = SURFACE_MODE_CODES["HARD"]

PlanResult = namedtuple("PlanResult", "symmetry segments global_modes proxy_modes")


def plan_run_inputs(engine, props):
    return (
        engine,
        engine.unset_surface_mode(props),
        engine.proxy_surface_mode(props),
        props.proxy_mode,
        engine.splits_regions(props),
        props.material_seams,
        engine.uses_symmetry(props),
        engine.paint_is_seam_cost(props),
    )


class PlanRun:
    def __init__(self, obj, engine, props):
        mesh = obj.data
        self.engine = engine
        self.props = props
        self.worker = None
        self.global_classify = None
        self.proxy_classify = None
        self.symmetry_check = start_unwrap_symmetry(
            engine, props, mesh, painted=paints_seams(obj, engine, props)
        )
        try:
            self.worker = self._start_worker(mesh, applied_matrix(obj))
            self.global_classify = self._start_global_classify(mesh)
            self.proxy_classify = self._start_proxy_classify(mesh)
        except EngineError:
            self.close()
            raise
        self.local_positions = vertex_position_array(mesh)
        self.corners, self.totals = loop_vertices_and_face_sizes(mesh)

    def tasks(self):
        tasks = (
            self.worker,
            self.symmetry_check,
            self.global_classify,
            self.proxy_classify,
        )
        return [task for task in tasks if task is not None]

    def done(self):
        return all(task.done() for task in self.tasks())

    def result(self):
        symmetry = None
        if self.symmetry_check is not None:
            symmetry = self.symmetry_check.result()
        segments = self._segments(symmetry)
        proxy_modes = None
        if self.proxy_classify is not None:
            proxy_modes = self.proxy_classify.result()
        return PlanResult(symmetry, segments, self._global_modes(segments), proxy_modes)

    def close(self):
        for task in self.tasks():
            task.close()

    def _start_worker(self, mesh, matrix):
        unset_mode = self.engine.unset_surface_mode(self.props)
        if self.engine.splits_regions(self.props):
            return segment_work(
                mesh,
                world_positions(mesh, matrix),
                tools_command(),
                unset_mode,
                fine_cut=True,
                material_seams=self.props.material_seams,
            )
        if not self.engine.supports_hard_surface:
            return None
        return WorkerProcess(
            classify_job,
            tools_command(),
            get_extension_dir_path() / "segment",
            *mesh_arrays(mesh),
            surface_modes(mesh, unset_mode),
        )

    def _start_global_classify(self, mesh):
        if self.engine.unset_surface_mode(self.props) != "AUTO":
            return None
        if not any(face_values(mesh, name).any() for name in OVERRIDE_ATTRIBUTES):
            return None
        return WorkerProcess(
            classify_job,
            tools_command(),
            get_extension_dir_path() / "segment",
            *mesh_arrays(mesh),
            numpy.full(
                len(mesh.polygons), SURFACE_MODE_CODES["AUTO"], dtype=numpy.int64
            ),
        )

    def _start_proxy_classify(self, mesh):
        if not self.engine.classifies_for_proxy_alone(self.props):
            return None
        largest_part = int(part_triangles(mesh).triangle_counts.max())
        if largest_part <= proxy_trigger_faces(self.props):
            return None
        return WorkerProcess(
            classify_job,
            tools_command(),
            get_extension_dir_path() / "segment",
            *mesh_arrays(mesh),
            surface_modes(mesh, self.engine.proxy_surface_mode(self.props)),
        )

    def _segments(self, symmetry):
        if self.worker is None:
            return None
        result = self.worker.result()
        if self.engine.splits_regions(self.props):
            _, segments = mirror_segmentation(
                self.local_positions,
                self.corners,
                self.totals,
                result,
                symmetry,
            )
            return segments
        no_regions = numpy.zeros(len(result), dtype=numpy.int64)
        return SegmentRegions(result, [], 0, no_regions, no_regions, hand_edited=False)

    def _global_modes(self, segments):
        unset_mode = self.engine.unset_surface_mode(self.props)
        if unset_mode != "AUTO":
            return numpy.full(
                len(self.totals), SURFACE_MODE_CODES[unset_mode], dtype=numpy.int64
            )
        if self.global_classify is not None:
            return self.global_classify.result()
        return segments.surface_modes


SurfaceHit = namedtuple("SurfaceHit", "face point")
PlanState = namedtuple("PlanState", "regions overrides")
PickedOverrides = namedtuple("PickedOverrides", "surface_mode keeps_uvs")
PART_MODE_ITEMS = [
    ("GLOBAL", "Global", "Use the Mode setting above"),
    *(
        (mode, label, description)
        for mode, label, description in SURFACE_MODES
        if mode != "AUTO"
    ),
]
PART_MODE_OF_CODE = {
    0: "GLOBAL",
    **{SURFACE_MODE_CODES[mode]: mode for mode, _, _ in PART_MODE_ITEMS[1:]},
}


def picks_parts(event):
    return event.ctrl or event.oskey


class PlanSession:
    def __init__(self, obj, engine, props, result):
        mesh = obj.data
        self.obj = obj
        self.engine = engine
        self.props = props
        self.symmetry = result.symmetry
        self.global_modes = result.global_modes
        self.proxy_modes = result.proxy_modes
        positions = vertex_position_array(mesh)
        self.bvh = BVHTree.FromPolygons(positions.tolist(), face_vertices(mesh))
        self.part_counts = part_triangles(mesh)
        corners, totals = loop_vertices_and_face_sizes(mesh)
        self.triangles_per_face = totals - 2
        face_labels = self.part_counts.vertex_labels[corners[face_starts(totals)]]
        self.face_loose_part = numpy.searchsorted(
            self.part_counts.part_labels, face_labels
        )
        matrix = numpy.array(obj.matrix_world)
        triangle_corners, self.triangle_faces = lifted_triangles(mesh, positions, 0.0)
        self.triangle_corners = triangle_corners @ matrix[:3, :3].T + matrix[:3, 3]
        self.plane_shapes = []
        if self.symmetry is not None:
            self.plane_shapes = plane_shapes(self.symmetry, positions, obj.matrix_world)
        self.plane_batches = None

        self.regions = None
        segments = result.segments
        if segments is not None and segments.region_count:
            partners = {}
            if self.symmetry is not None:
                partners = mesh_mirror_partners(
                    mesh, self.symmetry.center, self.symmetry.axis
                )
            self.regions = RegionEditor(obj, segments, partners)

        self.overrides = {name: face_values(mesh, name) for name in OVERRIDE_ATTRIBUTES}
        self.picked = set()
        self.picked_batch = None
        self.hovered_loose_part = None
        self.hover_batches = {}
        self.done = []
        self.undone = []
        self._modes_changed()

    def _fill(self, faces, color):
        corners = self.triangle_corners[faces[self.triangle_faces]].reshape(-1, 3)
        return Shape("TRIS", corners, None, color, FILL_NUDGE, None)

    def _modes_changed(self):
        self.kept = self.overrides[KEEP_ATTRIBUTE] != 0
        overridden = self.overrides[SURFACE_MODE_ATTRIBUTE]
        self.modes = numpy.where(overridden != 0, overridden, self.global_modes)
        self._regions_changed()

    def _regions_changed(self):
        organic = (self.modes == ORGANIC) & ~self.kept
        labels = numpy.zeros(len(self.modes), dtype=numpy.int64)
        region_count = 0
        touching = {}
        if self.regions is not None:
            self.regions.hide_regions(~organic)
            labels = numpy.where(organic, self.regions.joined, 0)
            region_count = self.regions.part_count
            touching = self.regions.parts_of_part
        outside_regions = organic & (labels == 0)
        labels[outside_regions] = (
            region_count + 1 + self.face_loose_part[outside_regions]
        )
        drawn = numpy.unique(labels[labels > 0]).tolist()
        color_of_label = {}
        for label in sorted(drawn, key=lambda label: -len(touching.get(label, ()))):
            used = [
                color_of_label[other]
                for other in touching.get(label, ())
                if other in color_of_label
            ]
            taken = list(color_of_label.values())
            color_of_label[label] = min(
                range(len(REGION_COLORS)),
                key=lambda color: (used.count(color), taken.count(color)),
            )
        fills = [
            (labels == label, REGION_COLORS[color_of_label[label]]) for label in drawn
        ]
        if self.engine.supports_hard_surface:
            fills.append(((self.modes == HARD) & ~self.kept, HARD_COLOR))
        self.fill_shapes = [
            self._fill(faces, color) for faces, color in fills if faces.any()
        ]
        self.fill_batches = None
        self.lines = self._lines()

    def _lines(self):
        lines = []
        part_count = len(self.part_counts.part_labels)
        kept_count = len(numpy.unique(self.face_loose_part[self.kept]))
        unwrapped_count = part_count - kept_count
        counts = []
        if self.engine.supports_hard_surface:
            hard = (self.modes == HARD) & ~self.kept
            hard_count = len(numpy.unique(self.face_loose_part[hard]))
            if unwrapped_count - hard_count:
                counts.append(f"{unwrapped_count - hard_count} organic")
            if hard_count:
                counts.append(f"{hard_count} hard surface")
        if unwrapped_count:
            noun = "part" if unwrapped_count == 1 else "parts"
            counted = ", ".join(counts) if counts else str(unwrapped_count)
            lines.append(f"{counted} {noun}")
        if kept_count:
            verb = "keeps" if kept_count == 1 else "keep"
            lines.append(f"{count_label(kept_count, 'part')} {verb} UVs")

        if self.regions is not None:
            edited = REGION_ATTRIBUTE in self.obj.data.attributes
            lines.append(
                count_label(self.regions.part_count, "region")
                + (" (edited)" if edited else "")
            )

        if self.symmetry is not None:
            axis = self.symmetry.axis
            line = (
                f"Mostly symmetrical on {axis}"
                if self.symmetry.kind == "WHOLE"
                else f"Symmetrical on {axis}"
            )
            left_out = self.face_loose_part[~self.symmetry.symmetric_faces]
            if len(left_out):
                line += (
                    f", {count_label(len(numpy.unique(left_out)), 'part')} not mirrored"
                )
            lines.append(line)
        elif self.engine.uses_symmetry(self.props):
            lines.append("Not symmetrical")

        if self.engine.uses_proxy(self.props):
            pieces = self._unwrap_pieces()
            triangle_counts = numpy.bincount(
                pieces, weights=self.triangles_per_face
            ).astype(numpy.int64)
            if self.props.proxy_mode == "AUTO":
                modes = self.modes if self.proxy_modes is None else self.proxy_modes
                proxy_modes = numpy.where(self.kept, ORGANIC, modes)
                triangle_counts = auto_proxy_triangle_counts(
                    triangle_counts,
                    pieces,
                    proxy_modes,
                    self.engine.auto_proxy_hard_surface_faces,
                )
            trigger = proxy_trigger_faces(self.props)
            proxied = int((triangle_counts > trigger).sum())
            lines.append(count_label(proxied, "proxy part"))
        return lines

    def _unwrap_pieces(self):
        if self.regions is None:
            return self.face_loose_part
        organic = (self.modes == ORGANIC) & ~self.kept
        labels = numpy.where(organic, self.regions.joined, 0)
        face_pairs = self.regions.face_pairs
        first, second = labels[face_pairs.first], labels[face_pairs.second]
        # split_on_borders cuts only edges between two segmented regions
        attached = (first == second) | (first == 0) | (second == 0)
        pairs = numpy.stack([face_pairs.first, face_pairs.second], axis=1)[attached]
        _, pieces = numpy.unique(
            lowest_index_part_labels(len(labels), pairs), return_inverse=True
        )
        return pieces

    def set_props(self, props):
        self.props = props
        self.lines = self._lines()

    def _hit_under(self, region, event):
        x, y = event.mouse_region_x, event.mouse_region_y
        if not (0 <= x < region.width and 0 <= y < region.height):
            return None
        view = region.data
        to_local = self.obj.matrix_world.inverted()
        origin = to_local @ region_2d_to_origin_3d(region, view, (x, y))
        direction = to_local.to_3x3() @ region_2d_to_vector_3d(region, view, (x, y))
        location, _, face, _ = self.bvh.ray_cast(origin, direction)
        if face is None:
            return None
        return SurfaceHit(face, numpy.array(location))

    def hover(self, region, event):
        hovered_before = self.hovered_loose_part
        hit = self._hit_under(region, event)
        edits_regions = (
            hit is not None
            and not picks_parts(event)
            and self.regions is not None
            and self.regions.edits_face(hit.face)
        )
        regions_changed = False
        if self.regions is not None:
            region_hit = hit if edits_regions else None
            regions_changed = self.regions.hover(region, region_hit)
        self.hovered_loose_part = None
        if hit is not None and not edits_regions:
            self.hovered_loose_part = int(self.face_loose_part[hit.face])
        return regions_changed or self.hovered_loose_part != hovered_before

    def click(self, event):
        if self.hovered_loose_part is not None:
            self.pick(self.hovered_loose_part, extend=event.shift)
            return True
        if self.regions is not None and self.regions.has_hover():
            self._record()
            if self.regions.hovered is not None:
                self.regions.toggle(self.regions.hovered)
            else:
                self.regions.add_loops(self.regions.hovered_loops)
            self._regions_changed()
            return True
        self.picked = set()
        self.picked_batch = None
        return False

    def pick(self, part, *, extend):
        if extend or part in self.picked:
            self.picked ^= {part}
        else:
            self.picked = {part}
        self.picked_batch = None

    def _picked_faces(self):
        return numpy.isin(self.face_loose_part, list(self.picked))

    def picked_overrides(self):
        face = int(numpy.flatnonzero(self.face_loose_part == min(self.picked))[0])
        return PickedOverrides(
            PART_MODE_OF_CODE[int(self.overrides[SURFACE_MODE_ATTRIBUTE][face])],
            bool(self.overrides[KEEP_ATTRIBUTE][face]),
        )

    def write_picked(self, name, value):
        self._record()
        values = self.overrides[name].copy()
        values[self._picked_faces()] = value
        self.overrides = {**self.overrides, name: values}
        self._write_overrides()
        self._modes_changed()

    def _write_overrides(self):
        attributes = self.obj.data.attributes
        for name, values in self.overrides.items():
            attribute = attributes.get(name)
            if attribute is None:
                if not values.any():
                    continue
                attribute = attributes.new(name, OVERRIDE_ATTRIBUTES[name], "FACE")
            attribute.data.foreach_set("value", values)

    def _state(self):
        regions = None if self.regions is None else self.regions.state()
        return PlanState(regions, self.overrides)

    def _record(self):
        self.done.append(self._state())
        excess = len(self.done) - bpy.context.preferences.edit.undo_steps
        if excess > 0:
            del self.done[:excess]
        self.undone.clear()

    def undo(self):
        return self._step(self.done, self.undone)

    def redo(self):
        return self._step(self.undone, self.done)

    def _step(self, source, target):
        if not source:
            return False
        target.append(self._state())
        state = source.pop()
        if self.regions is not None:
            self.regions.restore(state.regions)
        self.overrides = state.overrides
        self._write_overrides()
        self._modes_changed()
        return True

    def draw_items(self):
        if self.fill_batches is None:
            self.fill_batches = [
                (shape, shape_batch(shape)) for shape in self.fill_shapes
            ]
        items = list(self.fill_batches)
        if self.picked:
            if self.picked_batch is None:
                shape = self._fill(self._picked_faces(), PICKED_COLOR)
                self.picked_batch = (shape, shape_batch(shape))
            items.append(self.picked_batch)
        part = self.hovered_loose_part
        if part is not None:
            if part not in self.hover_batches:
                shape = self._fill(self.face_loose_part == part, HOVERED_PART_COLOR)
                self.hover_batches[part] = (shape, shape_batch(shape))
            items.append(self.hover_batches[part])
        if self.regions is not None:
            items.extend(self.regions.draw_items())
        if self.plane_batches is None:
            self.plane_batches = [
                (shape, shape_batch(shape)) for shape in self.plane_shapes
            ]
        items.extend(self.plane_batches)
        return items


_run = None
_session = None
_draw_handler = None


def _draw():
    if _session is not None:
        draw_batches(_session.draw_items())


def stop_plan():
    global _run, _session, _draw_handler
    if _run is not None:
        _run.close()
        _run = None
    _session = None
    if _draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, "WINDOW")
        _draw_handler = None
    tag_redraw()


def _draw_status(header, context):
    hints = []
    if _session is not None and _session.regions is not None:
        hints.append(("Edit Regions", ["MOUSE_LMB"]))
    hints += [
        ("Select Part", ["EVENT_CTRL", "MOUSE_LMB"]),
        ("Undo", ["EVENT_CTRL", "EVENT_Z"]),
        ("Reset Regions", ["EVENT_R"]),
        ("Finish", ["EVENT_ESC"]),
    ]
    draw_status_hints(header.layout, hints, context)


def _draw_loading_status(header, context):
    draw_status_hints(header.layout, [("Cancel", ["EVENT_ESC"])], context)


class UVGAMI_OT_plan(bpy.types.Operator):
    bl_idname = "uvgami.plan"
    bl_label = "Plan"
    bl_description = (
        "Show and edit the part modes, regions and symmetry plane for the unwrap."
        " Click a border or edge loop to edit the regions,"
        " Ctrl+click a part to change its settings"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            _run is None
            and _session is None
            and not manager.is_active
            and context.mode in ("OBJECT", "EDIT_MESH")
            and obj is not None
            and obj.type == "MESH"
        )

    def invoke(self, context, event):
        obj = context.active_object
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        if not validate_obj(self, obj, report=True):
            return {"CANCELLED"}
        if context.area is None or context.area.type != "VIEW_3D":
            self.report({"ERROR"}, "Run Plan from the 3D Viewport")
            return {"CANCELLED"}
        self.obj = obj
        try:
            self._start_run(context)
        except EngineError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.timer = context.window_manager.event_timer_add(
            TIMER_SECONDS, window=context.window
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _start_run(self, context):
        global _run, _draw_handler
        props = context.scene.uvgami
        engine = active_engine(props.engine)
        self.run_inputs = plan_run_inputs(engine, props)
        _run = PlanRun(self.obj, engine, SettingsCopy(props))
        if _draw_handler is None:
            _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                _draw, (), "WINDOW", "POST_VIEW"
            )
        context.workspace.status_text_set(_draw_loading_status)
        tag_redraw()

    def _restart_run(self, context):
        global _run, _session
        if _run is not None:
            _run.close()
            _run = None
        _session = None
        try:
            self._start_run(context)
        except EngineError as error:
            self.report({"ERROR"}, str(error))
            return self._end(context, {"CANCELLED"})
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if not check_exists(self.obj) or context.mode != "OBJECT":
            return self._end(context, {"CANCELLED"})
        if _run is None and _session is None:
            return self._end(context, {"FINISHED"})
        if event.type == "TIMER":
            props = context.scene.uvgami
            if plan_run_inputs(active_engine(props.engine), props) != self.run_inputs:
                return self._restart_run(context)
            if _session is not None and _session.props.proxy_mode != props.proxy_mode:
                _session.set_props(SettingsCopy(props))
                tag_redraw()
            if _run is not None and _run.done() and not self._start_session(context):
                return self._end(context, {"CANCELLED"})
            return {"PASS_THROUGH"}
        if event.type in ("ESC", "RET") and event.value == "PRESS":
            return self._end(context, {"FINISHED"})
        session = _session
        if session is None:
            return {"PASS_THROUGH"}
        # blender's undo frees the mesh the session reads, so ctrl+z is caught over the sidebar too
        if event.type == "Z" and event.value == "PRESS" and picks_parts(event):
            if session.redo() if event.shift else session.undo():
                tag_redraw()
            return {"RUNNING_MODAL"}
        if event.type == "R" and event.value == "PRESS":
            clear_regions(self.obj.data)
            return self._restart_run(context)
        # blender's viewport region spans the whole area under an overlapping sidebar
        over_side_region = any(
            region.type != "WINDOW"
            and region.x <= event.mouse_x < region.x + region.width
            and region.y <= event.mouse_y < region.y + region.height
            for region in context.area.regions
        )
        if over_side_region:
            return {"PASS_THROUGH"}
        if event.type in ("MOUSEMOVE", "LEFT_CTRL", "RIGHT_CTRL", "OSKEY"):
            if session.hover(context.region, event):
                tag_redraw(("WINDOW",))
            return {"PASS_THROUGH"}
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            handled = session.click(event)
            tag_redraw()
            return {"RUNNING_MODAL"} if handled else {"PASS_THROUGH"}
        return {"PASS_THROUGH"}

    def _start_session(self, context):
        global _run, _session
        run, _run = _run, None
        try:
            _session = PlanSession(self.obj, run.engine, run.props, run.result())
        except (EngineError, WorkerError) as error:
            self.report({"ERROR"}, f"Plan: {error}")
            return False
        finally:
            run.close()
        context.workspace.status_text_set(_draw_status)
        tag_redraw()
        return True

    def _end(self, context, outcome):
        context.window_manager.event_timer_remove(self.timer)
        context.workspace.status_text_set(None)
        stop_plan()
        return outcome


class UVGAMI_OT_stop_plan(bpy.types.Operator):
    bl_idname = "uvgami.stop_plan"
    bl_label = "Exit Plan"
    bl_description = "Stop planning and remove the overlay from the viewport"

    @classmethod
    def poll(cls, context):
        return _run is not None or _session is not None

    def execute(self, context):
        stop_plan()
        return {"FINISHED"}


class UVGAMI_OT_set_part_mode(bpy.types.Operator):
    bl_idname = "uvgami.set_part_mode"
    bl_label = "Mode"
    bl_description = "Whether seams on the selected parts follow sharp features"

    mode: bpy.props.EnumProperty(name="Mode", items=PART_MODE_ITEMS)

    @classmethod
    def poll(cls, context):
        return _session is not None and bool(_session.picked)

    def execute(self, context):
        code = 0 if self.mode == "GLOBAL" else SURFACE_MODE_CODES[self.mode]
        _session.write_picked(SURFACE_MODE_ATTRIBUTE, code)
        tag_redraw()
        return {"FINISHED"}


class UVGAMI_OT_set_part_keep_uvs(bpy.types.Operator):
    bl_idname = "uvgami.set_part_keep_uvs"
    bl_label = "Keep UVs"
    bl_description = "Leave the selected parts out of the unwrap and keep their UV map"

    keep: bpy.props.BoolProperty(name="Keep UVs")

    @classmethod
    def poll(cls, context):
        return _session is not None and bool(_session.picked)

    def execute(self, context):
        _session.write_picked(KEEP_ATTRIBUTE, int(self.keep))
        tag_redraw()
        return {"FINISHED"}


PART_MODE_LABELS = {mode: label for mode, label, _ in PART_MODE_ITEMS}


def draw_unwrap_and_plan(layout, engine):
    box = layout if _session is None else layout.box()
    split = box.split(factor=0.7)
    split.scale_y = 2
    unwrap = split.row()
    unwrap.enabled = not manager.is_active
    unwrap.operator("uvgami.start", icon="UV")
    if _run is not None:
        split.operator("uvgami.stop_plan", text="", icon="SORTTIME")
        return
    if _session is None:
        split.operator("uvgami.plan", text="Plan")
        return
    split.operator("uvgami.stop_plan", text="Plan", depress=True)
    column = box.column(align=True)
    column.active = False
    for line in _session.lines:
        column.label(text=line)

    if not _session.picked:
        return
    picked = _session.picked_overrides()
    parts = box.box()
    parts.label(
        text=f"{count_label(len(_session.picked), 'part')} selected", icon="MOD_EXPLODE"
    )
    split = parts.split(factor=0.7)
    split.label(icon="PINNED", text="Keep UVs")
    keep_icon = "CHECKBOX_HLT" if picked.keeps_uvs else "CHECKBOX_DEHLT"
    keep = split.operator(
        "uvgami.set_part_keep_uvs", text="", icon=keep_icon, emboss=False
    )
    keep.keep = not picked.keeps_uvs
    if engine.supports_hard_surface:
        row = parts.row()
        row.active = not picked.keeps_uvs
        row.label(icon="MOD_BEVEL", text="Mode")
        row.operator_menu_enum(
            "uvgami.set_part_mode", "mode", text=PART_MODE_LABELS[picked.surface_mode]
        )
    # priority = parts.column(align=True)
    # priority.label(icon="SOLO_OFF", text="Priority")
    # priority.row(align=True).prop(overrides, "priority", expand=True)
    # row = parts.row()
    # # the scale is applied after Average Islands Scale
    # row.active = props.fix_scale
    # row.label(icon="FULLSCREEN_ENTER", text="UV Scale")
    # row.prop(overrides, "uv_scale", text="")
