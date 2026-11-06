import math
from collections import namedtuple

import bpy
import numpy
from bpy_extras.view3d_utils import (
    location_3d_to_region_2d,
    region_2d_to_origin_3d,
    region_2d_to_vector_3d,
)
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from ..engines import active_engine
from ..no_bpy import EngineError
from ..no_bpy.edge_loops import EdgeLoops, LoopRegion, drawn_loops, loop_side
from ..no_bpy.mesh import faces_from_arrays, undirected_edge_keys
from ..no_bpy.segmentation import (
    edge_face_pairs,
    joined_labels,
    nested_fine_labels,
    on_pairs_with_edges,
    pair_borders,
    pair_edges,
    pairs_between_regions,
)
from ..no_bpy.symmetry import face_partners, region_partners
from ..no_bpy.worker import WorkerError
from ..segmentation import (
    clear_regions,
    stored_loop_edges,
    write_regions,
)
from ..symmetry import mesh_mirror_partners, paints_seams
from ..unwrap.manager import SettingsCopy, manager
from ..utils.mesh import (
    applied_matrix,
    check_exists,
    edge_vertex_pairs,
    face_vertices,
    lifted_triangles,
    loop_vertices_and_face_sizes,
    validate_obj,
    vertex_position_array,
)
from ..utils.ui import tag_redraw
from .auto_preview import (
    BORDER_COLOR,
    FILL_NUDGE,
    LINE_NUDGE,
    LINE_WIDTH,
    SegmentationRun,
    Shape,
    draw_batches,
    shape_batch,
    stop_auto_preview,
)

PART_COLORS = (
    (0.25, 0.75, 0.35, 0.35),
    (0.95, 0.55, 0.15, 0.35),
    (0.25, 0.55, 0.95, 0.35),
    (0.85, 0.3, 0.6, 0.35),
    (0.9, 0.85, 0.2, 0.35),
    (0.3, 0.85, 0.85, 0.35),
    (0.6, 0.4, 0.9, 0.35),
    (0.7, 0.7, 0.7, 0.35),
)
OFF_BORDER_COLOR = (0.35, 0.35, 0.35, 0.45)
HOVER_COLOR = (1.0, 0.9, 0.1, 1.0)
TIMER_SECONDS = 0.1
BORDER_PICK_PIXELS = 30

_session = None
_segmentation_run = None
_draw_handler = None

LoopCandidate = namedtuple(
    "LoopCandidate", "split_region relabeled_faces vertices edges shape"
)
NO_LOOPS = ()
DrawnLoops = namedtuple("DrawnLoops", "edges loop_of_edge candidates shape")
NearestDrawnLoop = namedtuple("NearestDrawnLoop", "index closest")


def closest_points_on_segments(starts, ends, point):
    directions = ends - starts
    along = ((point - starts) * directions).sum(axis=1)
    along = numpy.clip(along / (directions**2).sum(axis=1), 0, 1)
    return starts + along[:, None] * directions


class EditSession:
    def __init__(self, obj, segments, partners):
        mesh = obj.data
        self.obj = obj
        self.positions = vertex_position_array(mesh)
        corners, totals = loop_vertices_and_face_sizes(mesh)
        self.face_corners = faces_from_arrays(corners, totals)
        self.face_of_vertex = numpy.full(len(self.positions), -1, dtype=numpy.int64)
        self.face_of_vertex[corners] = numpy.repeat(numpy.arange(len(totals)), totals)
        self.vertex_partners = numpy.full(len(self.positions), -1, dtype=numpy.int64)
        self.vertex_partners[list(partners)] = list(partners.values())
        self.face_partners = face_partners(corners, totals, self.vertex_partners)
        self.face_pairs = edge_face_pairs(corners, totals)
        self.edge_loops = EdgeLoops(self.positions, corners, totals)
        self.loop_edges = set()
        if segments.hand_edited:
            self.loop_edges = set(map(tuple, stored_loop_edges(mesh).tolist()))
        self.bvh = BVHTree.FromPolygons(self.positions.tolist(), face_vertices(mesh))
        edge_keys = undirected_edge_keys(*edge_vertex_pairs(mesh).T)
        self.edge_order = numpy.argsort(edge_keys)
        self.sorted_edge_keys = edge_keys[self.edge_order]

        matrix = numpy.array(obj.matrix_world)
        self.rotation_scale, self.offset = matrix[:3, :3], matrix[:3, 3]
        triangle_corners, self.triangle_faces = lifted_triangles(
            mesh, self.positions, 0.0
        )
        self.triangle_corners = self._world(triangle_corners)
        self._set_fine(nested_fine_labels(segments.labels, segments.fine_labels))
        if segments.hand_edited:
            self.on = on_pairs_with_edges(self.borders, segments.border_edges)
        else:
            self.on = pairs_between_regions(self.borders, segments.labels, self.fine)
        self.hovered = None
        self.hovered_mirror = None
        self.hovered_loop_key = None
        self.hovered_loops = NO_LOOPS
        self.hovered_region = None
        self._join_parts()

    def _world(self, points):
        return points @ self.rotation_scale.T + self.offset

    def _set_fine(self, fine):
        self.fine = fine
        self.region_partners = region_partners(fine, self.face_partners)
        self.borders = pair_borders(self.face_pairs, fine)
        self.pairs_of_region = {}
        for pair in self.borders:
            for region in pair:
                self.pairs_of_region.setdefault(region, []).append(pair)
        self.border_shapes = {
            pair: Shape(
                "LINES",
                self._world(self.positions[edges.ravel()]),
                None,
                OFF_BORDER_COLOR,
                LINE_NUDGE,
                LINE_WIDTH,
            )
            for pair, edges in self.borders.items()
        }
        # gpu batches are made inside the draw handler, where the gpu context is
        self.border_batches = {}
        self.loop_batches = {}
        self.loop_candidates = {}
        self.loop_regions = {}
        self.drawn_loops = {}
        self.drawn_loop_batches = {}

    def _join_parts(self):
        self.joined, self.part_count = joined_labels(self.fine, self.borders, self.on)
        triangle_parts = self.joined[self.triangle_faces]
        self.fill_shapes = []
        for part in range(1, self.part_count + 1):
            corners = self.triangle_corners[triangle_parts == part].reshape(-1, 3)
            color = PART_COLORS[(part - 1) % len(PART_COLORS)]
            self.fill_shapes.append(
                Shape("TRIS", corners, None, color, FILL_NUDGE, None)
            )
        self.fill_batches = None

    def hit_under(self, region, event):
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
        return face, numpy.array(location)

    def border_near(self, face, point):
        nearest, nearest_point, nearest_distance = None, None, numpy.inf
        for pair in self.pairs_of_region.get(int(self.fine[face]), []):
            edges = self.borders[pair]
            closest = closest_points_on_segments(
                self.positions[edges[:, 0]], self.positions[edges[:, 1]], point
            )
            distances = numpy.linalg.norm(closest - point, axis=1)
            index = int(distances.argmin())
            if distances[index] < nearest_distance:
                nearest, nearest_point = pair, closest[index]
                nearest_distance = distances[index]
        return nearest, nearest_point

    def _pixels_apart(self, region, point, other):
        to_world = self.obj.matrix_world
        first = location_3d_to_region_2d(region, region.data, to_world @ Vector(point))
        second = location_3d_to_region_2d(region, region.data, to_world @ Vector(other))
        if first is None or second is None:
            return math.inf
        return (first - second).length

    def _edge_near(self, face, point):
        vertices = numpy.array(self.face_corners[face])
        following = numpy.roll(vertices, -1)
        closest = closest_points_on_segments(
            self.positions[vertices], self.positions[following], point
        )
        index = int(numpy.linalg.norm(closest - point, axis=1).argmin())
        low, high = sorted((int(vertices[index]), int(following[index])))
        return int(self.fine[face]), low, high

    def _hover_loop_under(self, face, point):
        edge = self._edge_near(face, point)
        if edge not in self.loop_candidates:
            self.loop_candidates[edge] = self._loops_through(*edge)
        self.hovered_loop_key = edge
        self.hovered_loops = self.loop_candidates[edge]

    def hover(self, region, event):
        hovered_before, loops_before = self.hovered, self.hovered_loops
        region_before = self.hovered_region
        self.hovered, self.hovered_region = None, None
        self.hovered_loop_key, self.hovered_loops = None, NO_LOOPS
        hit = self.hit_under(region, event)
        if hit is not None:
            face, point = hit
            self.hovered_region = int(self.fine[face])
            drawn = self._drawn_loops(self.hovered_region)
            pair, closest = self.border_near(face, point)
            if drawn is not None:
                nearest = self._nearest_drawn_loop(drawn, pair, closest, point)
                if nearest is None:
                    self.hovered = pair
                elif (
                    self._pixels_apart(region, point, nearest.closest)
                    <= BORDER_PICK_PIXELS
                ):
                    self.hovered_loop_key = (self.hovered_region, nearest.index)
                    self.hovered_loops = drawn.candidates[nearest.index]
                else:
                    self._hover_loop_under(face, point)
            elif pair is not None and (
                self._pixels_apart(region, point, closest) <= BORDER_PICK_PIXELS
            ):
                self.hovered = pair
            else:
                self._hover_loop_under(face, point)
        self.hovered_mirror = None
        if self.hovered is not None:
            self.hovered_mirror = self._mirrored_border(self.hovered)
        return (
            self.hovered != hovered_before
            or self.hovered_loops is not loops_before
            or self.hovered_region != region_before
        )

    def _drawn_loops(self, region):
        if region in self.drawn_loops:
            return self.drawn_loops[region]
        loops = drawn_loops(self.edge_loops, self._loop_region(region))
        drawn = None
        if loops:
            edges = numpy.concatenate(
                [numpy.stack([loop, numpy.roll(loop, -1)], axis=1) for loop in loops]
            )
            shape = Shape(
                "LINES",
                self._world(self.positions[edges.ravel()]),
                None,
                OFF_BORDER_COLOR,
                LINE_NUDGE,
                LINE_WIDTH / 2,
            )
            loop_of_edge = numpy.repeat(
                numpy.arange(len(loops)), [len(loop) for loop in loops]
            )
            candidates = [self._loop_and_mirror(loop) for loop in loops]
            drawn = DrawnLoops(edges, loop_of_edge, candidates, shape)
        self.drawn_loops[region] = drawn
        return drawn

    def _nearest_drawn_loop(self, drawn, pair, closest, point):
        on_loops = closest_points_on_segments(
            self.positions[drawn.edges[:, 0]], self.positions[drawn.edges[:, 1]], point
        )
        distances = numpy.linalg.norm(on_loops - point, axis=1)
        index = int(distances.argmin())
        if pair is not None and numpy.linalg.norm(closest - point) <= distances[index]:
            return None
        return NearestDrawnLoop(int(drawn.loop_of_edge[index]), on_loops[index])

    def _mirrored_border(self, pair):
        low, high = (self.region_partners.get(region) for region in pair)
        if low is None or high is None:
            return None
        mirrored = (min(low, high), max(low, high))
        if mirrored == pair or mirrored not in self.borders:
            return None
        return mirrored

    def _loop_region(self, region):
        if region not in self.loop_regions:
            self.loop_regions[region] = LoopRegion(
                self.edge_loops, self.fine == region, self.face_pairs
            )
        return self.loop_regions[region]

    def _loops_through(self, region, tail, head):
        loop = self.edge_loops.find_loop(self._loop_region(region), tail, head)
        if loop is None:
            return NO_LOOPS
        return self._loop_and_mirror(loop)

    def _loop_and_mirror(self, loop):
        candidate = self._loop_candidate(loop)
        if candidate is None:
            return NO_LOOPS
        mirrored = self._mirrored_loop(loop)
        if mirrored is None:
            return (candidate,)
        mirror_candidate = self._loop_candidate(mirrored)
        if mirror_candidate is None:
            return (candidate,)
        return (candidate, mirror_candidate)

    def _mirrored_loop(self, loop):
        mirrored = self.vertex_partners[loop]
        if (mirrored < 0).any() or set(mirrored.tolist()) == set(loop):
            return None
        keys = undirected_edge_keys(mirrored, numpy.roll(mirrored, -1))
        if not numpy.isin(keys, self.sorted_edge_keys).all():
            return None
        return mirrored.tolist()

    def _loop_candidate(self, loop):
        region = int(self.fine[self.face_of_vertex[loop[0]]])
        loop_region = self._loop_region(region)
        if not all(loop_region.interior[vertex] for vertex in loop):
            return None
        side = loop_side(loop_region, loop)
        if side is None:
            return None
        edges = numpy.stack([loop, numpy.roll(loop, -1)], axis=1)
        shape = Shape(
            "LINES",
            self._world(self.positions[edges.ravel()]),
            None,
            HOVER_COLOR,
            LINE_NUDGE,
            LINE_WIDTH / 2,
        )
        return LoopCandidate(region, side, loop, numpy.sort(edges, axis=1), shape)

    def add_loops(self, candidates):
        for hovered in candidates:
            candidate = self._loop_candidate(hovered.vertices)
            if candidate is not None:
                self._split_along(candidate)
        self._changed()

    def _split_along(self, candidate):
        on_edges = pair_edges(self.borders, self.on)
        fine = self.fine.copy()
        new_label = int(fine.max()) + 1
        fine[candidate.relabeled_faces] = new_label
        self.loop_edges.update(map(tuple, candidate.edges.tolist()))
        self._set_fine(fine)
        self.on = on_pairs_with_edges(self.borders, on_edges)
        self.on.add((candidate.split_region, new_label))

    def _is_loop_border(self, pair):
        return set(map(tuple, self.borders[pair].tolist())) <= self.loop_edges

    def toggle(self, pair):
        pairs = {pair}
        mirrored = self._mirrored_border(pair)
        same_kind = mirrored is not None and (
            self._is_loop_border(mirrored) == self._is_loop_border(pair)
        )
        if same_kind:
            pairs.add(mirrored)
        if self._is_loop_border(pair):
            self._remove_loops(sorted(pairs))
        elif pair in self.on:
            self.on -= pairs
        else:
            self.on |= pairs
        self._changed()

    def _remove_loops(self, pairs):
        on_edges = pair_edges(self.borders, self.on - set(pairs))
        fine = self.fine
        for low, high in pairs:
            self.loop_edges -= set(map(tuple, self.borders[(low, high)].tolist()))
            fine = numpy.where(fine == high, low, fine)
        self._set_fine(fine)
        self.on = on_pairs_with_edges(self.borders, on_edges)

    def _changed(self):
        self.hovered = None
        self.hovered_mirror = None
        self.hovered_loop_key = None
        self.hovered_loops = NO_LOOPS
        self.hovered_region = None
        self._join_parts()
        self.write()

    def _edge_flags(self, edges):
        flags = numpy.zeros(len(self.edge_order), dtype=bool)
        if edges:
            keys = undirected_edge_keys(*numpy.array(edges).T)
            flags[self.edge_order[numpy.searchsorted(self.sorted_edge_keys, keys)]] = (
                True
            )
        return flags

    def write(self):
        write_regions(
            self.obj.data,
            self.fine,
            self._edge_flags(pair_edges(self.borders, self.on)),
            self._edge_flags(sorted(self.loop_edges)),
        )

    def draw_items(self):
        items = []
        for pair, shape in self.border_shapes.items():
            if pair in (self.hovered, self.hovered_mirror):
                color = HOVER_COLOR
            elif pair in self.on:
                color = BORDER_COLOR
            else:
                color = OFF_BORDER_COLOR
            if pair not in self.border_batches:
                self.border_batches[pair] = shape_batch(shape)
            items.append((shape._replace(color=color), self.border_batches[pair]))
        drawn = self.drawn_loops.get(self.hovered_region)
        if drawn is not None:
            if self.hovered_region not in self.drawn_loop_batches:
                self.drawn_loop_batches[self.hovered_region] = shape_batch(drawn.shape)
            items.append((drawn.shape, self.drawn_loop_batches[self.hovered_region]))
        for index, loop in enumerate(self.hovered_loops):
            key = (self.hovered_loop_key, index)
            if key not in self.loop_batches:
                self.loop_batches[key] = shape_batch(loop.shape)
            items.append((loop.shape, self.loop_batches[key]))
        if self.fill_batches is None:
            self.fill_batches = [shape_batch(shape) for shape in self.fill_shapes]
        items.extend(zip(self.fill_shapes, self.fill_batches))
        return items


def _draw():
    if _session is not None:
        draw_batches(_session.draw_items())


def start_session(obj, segments, partners):
    global _session, _draw_handler
    _session = EditSession(obj, segments, partners)
    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw, (), "WINDOW", "POST_VIEW"
        )
    tag_redraw()


def stop_segment_edit():
    global _session, _draw_handler, _segmentation_run
    if _segmentation_run is not None:
        _segmentation_run.close()
        _segmentation_run = None
    _session = None
    if _draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, "WINDOW")
        _draw_handler = None


def _draw_status(header, context):
    header.layout.label(text="Add Region", icon="MOUSE_LMB")
    header.layout.label(text="Reset", icon="EVENT_R")
    # blender sizes label icons one unit wide, narrower than the esc key icon
    header.layout.label(icon="EVENT_ESC")
    header.layout.separator()
    header.layout.label(text="Finish")


def _draw_loading_status(header, context):
    header.layout.label(icon="EVENT_ESC")
    header.layout.separator()
    header.layout.label(text="Cancel")


class UVGAMI_OT_edit_regions(bpy.types.Operator):
    bl_idname = "uvgami.edit_regions"
    bl_label = "Edit Regions"
    bl_description = (
        "Click a border to toggle it, or an edge loop to split a region."
        " Mirrored on symmetrical meshes"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return (
            _session is None
            and _segmentation_run is None
            and not manager.is_active
            and context.mode in ("OBJECT", "EDIT_MESH")
            and context.active_object is not None
            and context.active_object.type == "MESH"
        )

    def invoke(self, context, event):
        obj = context.active_object
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        if not validate_obj(self, obj, report=True):
            return {"CANCELLED"}
        if context.area is None or context.area.type != "VIEW_3D":
            self.report({"ERROR"}, "Run Edit Regions from the 3D Viewport")
            return {"CANCELLED"}
        props = context.scene.uvgami
        engine = active_engine(props.engine)
        if engine is None or not engine.splits_regions(props):
            self.report(
                {"ERROR"}, "Turn on Split Regions for an engine that supports it"
            )
            return {"CANCELLED"}
        stop_auto_preview()
        self.obj = obj
        try:
            self._segment(context)
        except EngineError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.timer = context.window_manager.event_timer_add(
            TIMER_SECONDS, window=context.window
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _segment(self, context):
        global _segmentation_run
        props = context.scene.uvgami
        engine = active_engine(props.engine)
        _segmentation_run = SegmentationRun(
            self.obj.data,
            applied_matrix(self.obj),
            engine,
            SettingsCopy(props),
            paints_seams(self.obj, engine, props),
            fine_cut=True,
        )
        context.workspace.status_text_set(_draw_loading_status)
        tag_redraw()

    def modal(self, context, event):
        if not check_exists(self.obj):
            return self._end(context, {"CANCELLED"})
        if _segmentation_run is None and _session is None:
            return self._end(context, {"FINISHED"})
        if event.type == "TIMER":
            if _segmentation_run is not None and _segmentation_run.done():
                self._start_session(context)
            return {"PASS_THROUGH"}
        if event.type in ("ESC", "RET") and event.value == "PRESS":
            return self._end(context, {"FINISHED"})
        session = _session
        if session is None:
            return {"PASS_THROUGH"}
        if event.type == "R" and event.value == "PRESS":
            clear_regions(self.obj.data)
            stop_segment_edit()
            try:
                self._segment(context)
            except EngineError as error:
                self.report({"ERROR"}, str(error))
                return self._end(context, {"CANCELLED"})
            tag_redraw()
            return {"RUNNING_MODAL"}
        # blender's viewport region spans the whole area under an overlapping sidebar
        over_side_region = any(
            region.type != "WINDOW"
            and region.x <= event.mouse_x < region.x + region.width
            and region.y <= event.mouse_y < region.y + region.height
            for region in context.area.regions
        )
        if over_side_region:
            return {"PASS_THROUGH"}
        if event.type == "MOUSEMOVE":
            if session.hover(context.region, event):
                tag_redraw(("WINDOW",))
            return {"PASS_THROUGH"}
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            if session.hovered is not None:
                session.toggle(session.hovered)
            elif session.hovered_loops:
                session.add_loops(session.hovered_loops)
            else:
                return {"PASS_THROUGH"}
            tag_redraw(("WINDOW",))
            return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}

    def _start_session(self, context):
        global _segmentation_run
        run, _segmentation_run = _segmentation_run, None
        try:
            symmetry, segments = run.result()
            partners = {}
            if symmetry is not None:
                partners = mesh_mirror_partners(
                    self.obj.data, symmetry.center, symmetry.axis
                )
        except (EngineError, WorkerError) as error:
            self.report({"ERROR"}, f"Edit Regions: {error}")
            self._end(context, {"CANCELLED"})
            return
        finally:
            run.close()
        if segments is None or not segments.region_count:
            self.report({"WARNING"}, "The mesh has no regions to edit")
            self._end(context, {"CANCELLED"})
            return
        start_session(self.obj, segments, partners)
        context.workspace.status_text_set(_draw_status)

    def _end(self, context, outcome):
        context.window_manager.event_timer_remove(self.timer)
        context.workspace.status_text_set(None)
        stop_segment_edit()
        tag_redraw()
        return outcome


class UVGAMI_OT_stop_edit_regions(bpy.types.Operator):
    bl_idname = "uvgami.stop_edit_regions"
    bl_label = "Stop Editing Regions"
    bl_description = "Finish editing the regions"

    @classmethod
    def poll(cls, context):
        return _session is not None or _segmentation_run is not None

    def execute(self, context):
        stop_segment_edit()
        return {"FINISHED"}


def draw_segment_edit(layout):
    row = layout.row()
    row.scale_y = 1.5
    if _segmentation_run is not None:
        row.operator("uvgami.stop_edit_regions", text="Loading", icon="SORTTIME")
        return
    if _session is not None:
        row.operator("uvgami.stop_edit_regions", icon="PANEL_CLOSE")
        return
    row.operator("uvgami.edit_regions", icon="RESTRICT_SELECT_OFF")
