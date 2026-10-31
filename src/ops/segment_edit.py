# import bpy
# import numpy
# from bpy_extras.view3d_utils import region_2d_to_origin_3d, region_2d_to_vector_3d
# from mathutils.bvhtree import BVHTree

# from ..engines import active_engine
# from ..engines.bundled import SEGMENT_ENGINE
# from ..manager import manager
# from ..no_bpy.segmentation import (
#     FlattenError,
#     border_edges,
#     edge_face_pairs,
#     merge_region,
# )
# from ..no_bpy.worker import WorkerError
# from ..segmentation import (
#     REGION_ATTRIBUTE,
#     clear_region_labels,
#     region_labels,
#     segment_work,
#     split_loop_work,
#     split_region_work,
#     write_region_labels,
# )
# from ..utils.mesh import (
#     check_exists,
#     loop_vertices_and_face_sizes,
#     face_vertices,
#     validate_obj,
#     vertex_position_array,
# )
# from ..utils.paths import get_preferences
# from ..utils.ui import tag_redraw
# from .auto_preview import (
#     BORDER_COLOR,
#     Preview,
#     Shape,
#     lifted_lines,
#     lifted_triangles,
#     show_preview,
# )

# REGION_COLORS = (
#     (0.25, 0.75, 0.35, 0.45),
#     (0.95, 0.55, 0.15, 0.45),
#     (0.25, 0.55, 0.95, 0.45),
#     (0.85, 0.3, 0.6, 0.45),
#     (0.9, 0.85, 0.2, 0.45),
#     (0.3, 0.85, 0.85, 0.45),
#     (0.6, 0.4, 0.9, 0.45),
#     (0.7, 0.7, 0.7, 0.45),
# )
# HINTS = (
#     "Click: split off the piece under the cursor    Ctrl Click: grow the last piece"
#     "    Alt Click: shrink it    Shift Click: merge into neighbor"
#     "    Ctrl Shift Click: split along the edge loop under the cursor    Esc: finish"
# )
# STEP_LIMIT_MESSAGES = {
#     "up": "the piece around the click is already the widest",
#     "down": "the piece around the click is already the narrowest",
# }
# TIMER_SECONDS = 0.1

# _session = None


# class EditSession:
#     def __init__(self, obj, segment_command):
#         self.obj = obj
#         self.segment_command = segment_command
#         mesh = obj.data
#         self.positions = vertex_position_array(mesh)
#         self.corners, self.totals = loop_vertices_and_face_sizes(mesh)
#         self.face_starts = numpy.cumsum(self.totals) - self.totals
#         self.face_pairs = edge_face_pairs(self.positions, self.corners, self.totals)
#         self.bvh = BVHTree.FromPolygons(self.positions.tolist(), face_vertices(mesh))
#         self.labels = region_labels(mesh)
#         self.worker = None
#         # called with the worker's result
#         self.finish = None
#         self.state = ""
#         # (labels before, seed face, piece) of the last split
#         self.last_split = None

#     def start(self, worker, finish, state):
#         self.worker = worker
#         self.finish = finish
#         self.state = state

#     def split(self, before, seed, piece, step):
#         self.start(
#             split_region_work(
#                 self.obj.data, self.segment_command, before, seed, piece, step
#             ),
#             lambda result: self._finish_split(result, before, seed, piece, step),
#             "Splitting...",
#         )

#     # a step that finds no other piece comes back with the piece it started from
#     def _finish_split(self, result, before, seed, piece, step):
#         if step is not None and result.piece == piece:
#             raise FlattenError(STEP_LIMIT_MESSAGES[step])
#         self.set_labels(result.labels)
#         self.last_split = (before, seed, result.piece)

#     def split_loop(self, face, corner):
#         self.start(
#             split_loop_work(
#                 self.obj.data, self.segment_command, self.labels, face, corner
#             ),
#             self._finish_loop_split,
#             "Splitting...",
#         )

#     def _finish_loop_split(self, labels):
#         self.set_labels(labels)
#         self.last_split = None

#     # the face under the cursor and the hit point on it in mesh coordinates, None off the mesh
#     def hit_under(self, region, event):
#         x, y = event.mouse_region_x, event.mouse_region_y
#         if not (0 <= x < region.width and 0 <= y < region.height):
#             return None
#         view = region.data
#         to_local = self.obj.matrix_world.inverted()
#         origin = to_local @ region_2d_to_origin_3d(region, view, (x, y))
#         direction = to_local.to_3x3() @ region_2d_to_vector_3d(region, view, (x, y))
#         location, _, face, _ = self.bvh.ray_cast(origin, direction)
#         if face is None:
#             return None
#         return face, numpy.array(location)

#     # the corner whose edge to the next corner passes closest to the point
#     def nearest_corner(self, face, point):
#         start = self.face_starts[face]
#         corners = self.corners[start : start + self.totals[face]]
#         starts = self.positions[corners]
#         directions = self.positions[numpy.roll(corners, -1)] - starts
#         along = ((point - starts) * directions).sum(axis=1)
#         along = numpy.clip(along / (directions**2).sum(axis=1), 0, 1)
#         closest = starts + along[:, None] * directions
#         return int(numpy.argmin(numpy.linalg.norm(closest - point, axis=1)))

#     def set_labels(self, labels):
#         self.labels = labels
#         write_region_labels(self.obj.data, labels)
#         self.draw()

#     def draw(self):
#         matrix = numpy.array(self.obj.matrix_world)
#         rotation_scale, offset = matrix[:3, :3], matrix[:3, 3]
#         corners, triangle_faces = lifted_triangles(self.obj.data, self.positions)
#         corners = corners @ rotation_scale.T + offset
#         triangle_labels = self.labels[triangle_faces]
#         regions = numpy.unique(self.labels[self.labels > 0])
#         shapes = []
#         for region in regions:
#             shown = corners[triangle_labels == region].reshape(-1, 3)
#             color = REGION_COLORS[(region - 1) % len(REGION_COLORS)]
#             shapes.append(Shape("TRIS", shown, None, color))
#         edges = numpy.array(
#             border_edges(self.face_pairs, self.labels), dtype=numpy.int64
#         )
#         if len(edges):
#             lines = lifted_lines(self.obj.data, self.positions, edges)
#             lines = lines @ rotation_scale.T + offset
#             shapes.append(Shape("LINES", lines, None, BORDER_COLOR))
#         summary = f"{self.obj.name}: {len(regions)} regions, edited by hand"
#         show_preview(Preview(shapes, summary))

#     def close(self):
#         if self.worker is not None:
#             self.worker.close()
#             self.worker = None


# def stop_segment_edit():
#     global _session
#     if _session is not None:
#         _session.close()
#         _session = None


# class UVGAMI_OT_edit_regions(bpy.types.Operator):
#     bl_idname = "uvgami.edit_regions"
#     bl_label = "Edit Regions"
#     bl_description = (
#         "Change the regions of the active mesh by hand. Click to split off the piece"
#         " under the cursor, Ctrl click to grow it, Alt click to shrink it, Shift click"
#         " to merge a region into its neighbor, Ctrl Shift click to split along the"
#         " edge loop under the cursor"
#     )
#     bl_options = {"REGISTER", "UNDO"}

#     @classmethod
#     def poll(cls, context):
#         return (
#             _session is None
#             and not manager.is_active
#             and context.mode == "OBJECT"
#             and context.active_object is not None
#             and context.active_object.type == "MESH"
#         )

#     def invoke(self, context, event):
#         global _session
#         obj = context.active_object
#         if not validate_obj(self, obj, report=True):
#             return {"CANCELLED"}
#         if context.area is None or context.area.type != "VIEW_3D":
#             self.report({"ERROR"}, "Run Edit Regions from the 3D Viewport")
#             return {"CANCELLED"}
#         segment_path, error = SEGMENT_ENGINE.validate(get_preferences())
#         if error is not None:
#             self.report({"ERROR"}, f"Split Into Regions: {error}")
#             return {"CANCELLED"}

#         _session = EditSession(obj, SEGMENT_ENGINE.command_prefix(segment_path))
#         if _session.labels.any():
#             _session.draw()
#         else:
#             props = context.scene.uvgami
#             unset_mode = active_engine(props.engine).unset_surface_mode(props)
#             _session.start(
#                 segment_work(obj.data, _session.segment_command, unset_mode),
#                 lambda regions: _session.set_labels(regions.labels),
#                 "Segmenting...",
#             )
#         self.timer = context.window_manager.event_timer_add(
#             TIMER_SECONDS, window=context.window
#         )
#         context.window_manager.modal_handler_add(self)
#         self._show_state(context)
#         return {"RUNNING_MODAL"}

#     def modal(self, context, event):
#         session = _session
#         if session is None or not check_exists(session.obj):
#             return self._end(context, {"CANCELLED"})
#         if event.type == "TIMER":
#             self._poll_worker(context, session)
#             return {"PASS_THROUGH"}
#         if event.type in ("ESC", "RET") and event.value == "PRESS":
#             return self._end(context, {"FINISHED"})
#         if event.type != "LEFTMOUSE" or event.value != "PRESS":
#             return {"PASS_THROUGH"}
#         if session.worker is not None:
#             return {"RUNNING_MODAL"}
#         splits_along_loop = event.ctrl and event.shift
#         if (event.ctrl or event.alt) and not splits_along_loop:
#             if session.last_split is None:
#                 self.report(
#                     {"WARNING"},
#                     "Click a region first, then Ctrl click grows that piece"
#                     " and Alt click shrinks it",
#                 )
#                 return {"RUNNING_MODAL"}
#             before, seed, piece = session.last_split
#             session.split(before, seed, piece, "up" if event.ctrl else "down")
#             self._show_state(context)
#             return {"RUNNING_MODAL"}
#         hit = session.hit_under(context.region, event)
#         if hit is None:
#             return {"PASS_THROUGH"}
#         face, location = hit
#         region = int(session.labels[face])
#         if region == 0:
#             self.report({"WARNING"}, "That face is outside the regions")
#             return {"RUNNING_MODAL"}
#         if splits_along_loop:
#             session.split_loop(face, session.nearest_corner(face, location))
#             self._show_state(context)
#             return {"RUNNING_MODAL"}
#         if event.shift:
#             try:
#                 session.set_labels(
#                     merge_region(session.labels, session.face_pairs, region)
#                 )
#             except FlattenError as error:
#                 self.report({"WARNING"}, str(error))
#             session.last_split = None
#             return {"RUNNING_MODAL"}
#         session.split(session.labels, face, None, None)
#         self._show_state(context)
#         return {"RUNNING_MODAL"}

#     def _poll_worker(self, context, session):
#         if session.worker is None or not session.worker.done():
#             return
#         worker, session.worker = session.worker, None
#         session.state = ""
#         try:
#             session.finish(worker.result())
#         except (FlattenError, WorkerError) as error:
#             self.report({"WARNING"}, str(error))
#         self._show_state(context)

#     def _show_state(self, context):
#         state = _session.state if _session is not None else ""
#         context.workspace.status_text_set(f"{state}    {HINTS}".strip())

#     def _end(self, context, outcome):
#         context.window_manager.event_timer_remove(self.timer)
#         context.workspace.status_text_set(None)
#         stop_segment_edit()
#         tag_redraw()
#         return outcome


# class UVGAMI_OT_reset_regions(bpy.types.Operator):
#     bl_idname = "uvgami.reset_regions"
#     bl_label = "Reset Regions"
#     bl_description = "Forget the hand edited regions, the engine picks them again"
#     bl_options = {"REGISTER", "UNDO"}

#     @classmethod
#     def poll(cls, context):
#         obj = context.active_object
#         return (
#             _session is None
#             and context.mode == "OBJECT"
#             and obj is not None
#             and obj.type == "MESH"
#             and REGION_ATTRIBUTE in obj.data.attributes
#         )

#     def execute(self, context):
#         clear_region_labels(context.active_object.data)
#         return {"FINISHED"}


# def draw_segment_edit(layout, context):
#     if _session is not None:
#         layout.label(
#             text="Click: split, Ctrl: grow, Alt: shrink, Shift: merge, Esc: finish"
#         )
#         return
#     row = layout.row(align=True)
#     row.operator("uvgami.edit_regions", icon="RESTRICT_SELECT_OFF")
#     obj = context.active_object
#     if (
#         obj is not None
#         and obj.type == "MESH"
#         and REGION_ATTRIBUTE in obj.data.attributes
#     ):
#         row.operator("uvgami.reset_regions", text="", icon="LOOP_BACK")
