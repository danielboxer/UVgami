import blf
import bpy
import gpu
import numpy
from gpu_extras.batch import batch_for_shader

from ..manager import manager
from ..utils.io import obj_numeric_columns, parse_obj_faces
from ..utils.mesh import check_exists
from ..utils.ui import tag_redraw
from .stop import group_targets, piece_target

VIEWER_WORKSPACE = "UVgami Viewer"

CYCLE_KEYS = {"RIGHT_ARROW": 1, "LEFT_ARROW": -1}

ARROW_FONT_SIZE = 24
DONE_FONT_SIZE = 14
GRID_GAP = 8

EDITOR_BACKGROUND = 0.22

WIRE_WIDTH = 1.0
WIRE_OUTLINE_WIDTH = 3.0
# blender's uv edge core color
WIRE_COLOR = (0.54, 0.54, 0.54)
# wires over triangles this small cover the colors
WIRE_HIDDEN_PIXELS = 3.0
WIRE_FULL_PIXELS = 10.0

_handler = None
_text_handler = None
_wire_batches = []
_fill_batch = None
_wire_shader = None
_fill_shader = None
WIRE_EDGE_BUCKET = 2.0
# PACK_MARGIN = 0.01

_corner_angle = None


def _corner_angles(triangle_corners):
    to_next = triangle_corners[:, [1, 2, 0]] - triangle_corners
    to_previous = triangle_corners[:, [2, 0, 1]] - triangle_corners
    dot = numpy.einsum("ijk,ijk->ij", to_next, to_previous)
    squared = (to_next * to_next).sum(2) * (to_previous * to_previous).sum(
        2
    ) - dot * dot
    return numpy.arctan2(numpy.sqrt(numpy.maximum(squared, 0)), dot)


def load_input_mesh(path):
    global _corner_angle
    _corner_angle = None
    if not path.is_file():
        return
    text = path.read_text()
    positions = obj_numeric_columns(text, "v", 3).astype(numpy.float32)
    triangle_corners = positions[parse_obj_faces(text).loop_vertices.reshape(-1, 3)]
    # optcuts snapshots keep the input face order
    _corner_angle = _corner_angles(triangle_corners)


# def _island_of_vertex(edges, vertex_count):
#     """Island index per snapshot vertex, 0 to island count - 1. Min label
#     propagation with pointer jumping, a few numpy rounds even on a long strip."""
#     label = numpy.arange(vertex_count)
#     e0, e1 = edges[:, 0], edges[:, 1]
#     while True:
#         before = label
#         low = numpy.minimum(label[e0], label[e1])
#         label = label.copy()
#         numpy.minimum.at(label, e0, low)
#         numpy.minimum.at(label, e1, low)
#         while True:
#             jumped = label[label]
#             if numpy.array_equal(jumped, label):
#                 break
#             label = jumped
#         if numpy.array_equal(label, before):
#             return numpy.unique(label, return_inverse=True)[1]


# def _pack_islands(co, island, island_count):
#     """Shelf pack the islands by bounding box into the unit square. The engine
#     never packs its layout, so straight from the snapshot the islands are
#     scattered and small."""
#     low = numpy.full((island_count, 2), numpy.inf, dtype=numpy.float32)
#     high = numpy.full((island_count, 2), -numpy.inf, dtype=numpy.float32)
#     numpy.minimum.at(low, island, co)
#     numpy.maximum.at(high, island, co)
#     size = high - low
#     width = float(numpy.sqrt((size[:, 0] * size[:, 1]).sum()))
#     margin = PACK_MARGIN * width
#     offset = numpy.empty_like(low)
#     x = y = margin
#     shelf = extent = 0.0
#     for i in numpy.argsort(-size[:, 1]):
#         if x > margin and x + size[i, 0] > width:
#             x, y, shelf = margin, y + shelf + margin, 0.0
#         offset[i] = (x, y)
#         x += size[i, 0] + margin
#         shelf = max(shelf, size[i, 1])
#         extent = max(extent, x, y + shelf + margin)
#     return (co - low[island] + offset[island]) / extent


def _wire_batches_by_edge_length(positions, edges):
    length = numpy.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    bucket = numpy.floor(
        numpy.log(numpy.maximum(length, 1e-9)) / numpy.log(WIRE_EDGE_BUCKET)
    )
    batches = []
    for key in numpy.unique(bucket):
        chosen = bucket == key
        batch = batch_for_shader(
            _wire_shader, "LINES", {"pos": positions[edges[chosen].reshape(-1)]}
        )
        batches.append((float(numpy.median(length[chosen])), batch))
    return batches


def set_snapshot(uv_positions, uv_indices):
    global _wire_batches, _fill_batch, _wire_shader, _fill_shader
    if _wire_shader is None:
        # polyline sets width in the shader, past the driver's line limit of 1
        _wire_shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
        _fill_shader = gpu.shader.from_builtin("SMOOTH_COLOR")

    triangles = numpy.asarray(uv_indices, dtype=numpy.int32)
    co = numpy.asarray(uv_positions, dtype=numpy.float32)

    edges = numpy.concatenate((triangles[:, :2], triangles[:, 1:], triangles[:, ::2]))
    edges.sort(axis=1)
    edges = numpy.unique(edges, axis=0)
    # island = _island_of_vertex(edges, len(co))
    # co = _pack_islands(co, island, island.max() + 1)
    _wire_batches = _wire_batches_by_edge_length(co, edges)

    _fill_batch = None
    if _corner_angle is not None and len(_corner_angle) == len(triangles):
        uv_corners = co[triangles]
        # Blender's angle stretch overlay uses the same measurement
        angle_error = numpy.abs(_corner_angles(uv_corners) - _corner_angle) / numpy.pi
        weight = (1 - (1 - angle_error) ** 2).reshape(-1)
        # blender's weight ramp, ported from BKE_defvert_weight_to_rgb, GPL-2.0-or-later
        colors = numpy.ones((weight.size, 4), dtype=numpy.float32)
        colors[:, 0] = numpy.clip((weight - 0.5) * 4, 0, 1)
        colors[:, 1] = numpy.clip(numpy.minimum(weight, 1 - weight) * 4, 0, 1)
        colors[:, 2] = numpy.clip((0.5 - weight) * 4, 0, 1)
        colors[:, :3] *= (weight / 2 + 0.5)[:, None]
        colors[:, :3] += EDITOR_BACKGROUND
        numpy.clip(colors, 0, 1, out=colors)
        _fill_batch = batch_for_shader(
            _fill_shader, "TRIS", {"pos": uv_corners.reshape(-1, 2), "color": colors}
        )


def clear_snapshot():
    global _wire_batches, _fill_batch
    _wire_batches = []
    _fill_batch = None


def _draw():
    gpu.state.blend_set("NONE")
    if _fill_batch is not None:
        _fill_batch.draw(_fill_shader)
    gpu.state.blend_set("ALPHA")
    if _wire_batches:
        _wire_shader.bind()
        _wire_shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
        _wire_shader.uniform_bool("lineSmooth", True)
    view2d = bpy.context.region.view2d
    left = view2d.view_to_region(0.0, 0.0, clip=False)[0]
    right = view2d.view_to_region(1.0, 0.0, clip=False)[0]
    span = WIRE_FULL_PIXELS - WIRE_HIDDEN_PIXELS
    for typical_edge, batch in _wire_batches:
        pixels = typical_edge * (right - left)
        fade = 1.0
        if _fill_batch is not None:
            fade = min(max((pixels - WIRE_HIDDEN_PIXELS) / span, 0.0), 1.0)
        if fade <= 0:
            continue
        for width, color in (
            (WIRE_OUTLINE_WIDTH, (0.0, 0.0, 0.0)),
            (WIRE_WIDTH, WIRE_COLOR),
        ):
            _wire_shader.uniform_float("lineWidth", width)
            _wire_shader.uniform_float("color", (*color, fade))
            batch.draw(_wire_shader)
    gpu.state.blend_set("NONE")


def _trim_to_uv(window):
    screen = window.screen
    while len(screen.areas) > 1:
        smallest = min(screen.areas, key=lambda area: area.width * area.height)
        with bpy.context.temp_override(window=window, area=smallest):
            bpy.ops.screen.area_close()
    area = screen.areas[0]
    area.ui_type = "UV"
    # with show_uv on, every selected object's uv map is drawn under the snapshot
    area.spaces.active.uv_editor.show_uv = False
    region = next(region for region in area.regions if region.type == "WINDOW")
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.image.view_all(fit_view=True)


def _schedule_workspace_trim(window):
    attempts = [0]

    def tick():
        workspace = bpy.data.workspaces.get(VIEWER_WORKSPACE)
        if workspace is None or attempts[0] > 20:
            return None
        if window.workspace != workspace:
            attempts[0] += 1
            return 0.01
        _trim_to_uv(window)
        return None

    # workspace switches take effect on the next main loop tick
    bpy.app.timers.register(tick, first_interval=0.0)


def _draw_hint_text():
    font = 0
    blf.color(font, 1.0, 1.0, 1.0, 0.9)
    view2d = bpy.context.region.view2d
    if manager.viewer_done:
        right, bottom = view2d.view_to_region(1.0, 0.0, clip=False)
        blf.size(font, DONE_FONT_SIZE)
        blf.position(font, right + GRID_GAP, bottom, 0)
        blf.draw(font, "Done, click to exit")
    if any(
        unwrap.is_viewable and unwrap is not manager.current_viewer
        for unwrap in manager.active
    ):
        _draw_switch_arrows(font, view2d)


def _draw_switch_arrows(font, view2d):
    left, top = view2d.view_to_region(0.0, 1.0, clip=False)
    right = view2d.view_to_region(1.0, 1.0, clip=False)[0]
    blf.size(font, ARROW_FONT_SIZE)
    for arrow, edge, side in (("←", left, -1), ("→", right, 1)):
        width, height = blf.dimensions(font, arrow)
        x = edge + side * GRID_GAP - (width if side < 0 else 0)
        blf.position(font, x, top - height, 0)
        blf.draw(font, arrow)


def start_viewer_draw():
    global _handler, _text_handler
    if _handler is None:
        # POST_VIEW in the image editor draws in uv space
        _handler = bpy.types.SpaceImageEditor.draw_handler_add(
            _draw, (), "WINDOW", "POST_VIEW"
        )
        _text_handler = bpy.types.SpaceImageEditor.draw_handler_add(
            _draw_hint_text, (), "WINDOW", "POST_PIXEL"
        )


def stop_viewer_draw():
    global _handler, _text_handler, _corner_angle
    if _handler is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(_handler, "WINDOW")
        _handler = None
    if _text_handler is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(_text_handler, "WINDOW")
        _text_handler = None
    clear_snapshot()
    _corner_angle = None
    manager.viewer_done = False


class UVGAMI_OT_view_unwrap(bpy.types.Operator):
    bl_idname = "uvgami.view_unwrap"
    bl_label = "View Unwrap"
    bl_description = "Show the unwrap live in the UV editor"

    stem: bpy.props.StringProperty(options={"SKIP_SAVE"})
    job_id: bpy.props.IntProperty(options={"SKIP_SAVE"})

    def execute(self, context):
        if self.job_id:
            candidates = [
                unwrap for unwrap in group_targets(self.job_id) if unwrap.is_viewable
            ]
        else:
            candidates = piece_target(self.stem)
        if not candidates:
            return {"CANCELLED"}
        unwrap = candidates[0]
        manager.is_viewer_active = True
        manager.exit_viewer = False
        manager.viewer_done = False

        window = context.window
        self.old_workspace = window.workspace.name
        workspace = bpy.data.workspaces.get(VIEWER_WORKSPACE)
        if workspace is None:
            existing = set(bpy.data.workspaces.keys())
            bpy.ops.workspace.duplicate()
            new_name = next(
                name for name in bpy.data.workspaces.keys() if name not in existing
            )
            workspace = bpy.data.workspaces[new_name]
            workspace.name = VIEWER_WORKSPACE
            _schedule_workspace_trim(window)
        else:
            window.workspace = workspace

        # edit mode uv maps are drawn under the snapshot
        self.old_mode = None
        active = context.view_layer.objects.active
        if active is not None and active.mode != "OBJECT":
            self.old_mode = (active, active.mode)
            bpy.ops.object.mode_set(mode="OBJECT")

        start_viewer_draw()
        self._show(unwrap)
        self.report({"INFO"}, "Click to exit viewer")
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _show(self, unwrap):
        if manager.current_viewer is not None:
            manager.current_viewer.viewing = False
        manager.viewer_done = False
        clear_snapshot()
        load_input_mesh(unwrap.path)
        manager.current_viewer = unwrap
        unwrap.viewing = True
        tag_redraw(("WINDOW",))

    def _cycle(self, step):
        viewable = [unwrap for unwrap in manager.active if unwrap.is_viewable]
        if not viewable:
            return
        current = manager.current_viewer
        index = viewable.index(current) + step if current in viewable else 0
        target = viewable[index % len(viewable)]
        if target is not current:
            self._show(target)

    def modal(self, context, event):
        if event.type in CYCLE_KEYS and event.value == "PRESS":
            self._cycle(CYCLE_KEYS[event.type])
            return {"RUNNING_MODAL"}

        if event.type in {"LEFTMOUSE", "RIGHTMOUSE", "ESC"} or manager.exit_viewer:
            manager.is_viewer_active = False
            if manager.current_viewer is not None:
                manager.current_viewer.viewing = False
                manager.current_viewer = None
            stop_viewer_draw()

            # both ops are deferred a tick but process in order
            workspace = bpy.data.workspaces.get(VIEWER_WORKSPACE)
            if workspace is not None and context.window.workspace == workspace:
                bpy.ops.workspace.delete()
            restore = bpy.data.workspaces.get(self.old_workspace)
            if restore is not None:
                context.window.workspace = restore

            if self.old_mode is not None:
                obj, mode = self.old_mode
                if check_exists(obj) and obj.visible_get() and obj.mode == "OBJECT":
                    context.view_layer.objects.active = obj
                    bpy.ops.object.mode_set(mode=mode)

            return {"FINISHED"}

        return {"PASS_THROUGH"}
