import bpy
import gpu
import numpy
from gpu_extras.batch import batch_for_shader

from ..manager import manager
from ..utils.mesh import check_exists

# ui and (object, mode) to restore when the viewer closes
old_ui = None
old_mode = None

# POST_VIEW in the image editor draws in uv space, so snapshot uvs go
# straight into gpu batches, no viewer mesh or edit mode needed
_handler = None
_wire_batch = None
_fill_batch = None
_wire_shader = None
_fill_shader = None
# per-face 3d areas of the engine input, for stretch colors
_face_area = None


def load_input_mesh(path):
    """Per-face 3d areas of the input obj. The engine keeps face order, so
    snapshot faces line up with these by index."""
    global _face_area
    _face_area = None
    if not path.is_file():
        return
    verts = []
    tris = []
    with path.open() as f:
        for line in f:
            if line.startswith("v "):
                x, y, z = line[2:].split()
                verts.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                tris.append(
                    [int(token.partition("/")[0]) - 1 for token in line[2:].split()]
                )
    pts = numpy.asarray(verts, dtype=numpy.float32)[
        numpy.asarray(tris, dtype=numpy.int32)
    ]
    cross = numpy.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
    _face_area = numpy.linalg.norm(cross, axis=1) / 2


def _stretch_colors(uv_pts):
    """Per-corner colors from uv area against 3d area, weight paint style:
    blue is no stretch, red is 2x stretched or squashed."""
    edge1 = uv_pts[:, 1] - uv_pts[:, 0]
    edge2 = uv_pts[:, 2] - uv_pts[:, 0]
    uv_area = numpy.abs(edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]) / 2
    total_uv = uv_area.sum()
    total_3d = _face_area.sum()
    if total_uv == 0 or total_3d == 0:
        return None
    with numpy.errstate(divide="ignore", invalid="ignore"):
        ratio = (uv_area / total_uv) / (_face_area / total_3d)
        stretch = numpy.maximum(ratio, 1 / ratio) - 1
    stretch = numpy.clip(numpy.nan_to_num(stretch, nan=1.0, posinf=1.0), 0, 1)

    # hsv rainbow with hue from 2/3 (blue) down to 0 (red)
    h = (1 - stretch) * 4
    colors = numpy.empty((len(stretch), 4), dtype=numpy.float32)
    colors[:, 0] = numpy.clip(numpy.abs(h - 3) - 1, 0, 1)
    colors[:, 1] = numpy.clip(2 - numpy.abs(h - 2), 0, 1)
    colors[:, 2] = numpy.clip(2 - numpy.abs(h - 4), 0, 1)
    colors[:, 3] = 0.9
    return numpy.repeat(colors, 3, axis=0)


def set_snapshot(uv_co, uv_indices):
    """Build the fill and wire batches for the latest engine snapshot."""
    global _wire_batch, _fill_batch, _wire_shader, _fill_shader
    if _wire_shader is None:
        _wire_shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        _fill_shader = gpu.shader.from_builtin("SMOOTH_COLOR")

    tris = numpy.asarray(uv_indices, dtype=numpy.int32)
    co = numpy.asarray(uv_co, dtype=numpy.float32)

    edges = numpy.concatenate((tris[:, :2], tris[:, 1:], tris[:, ::2]))
    edges.sort(axis=1)
    edges = numpy.unique(edges, axis=0)
    _wire_batch = batch_for_shader(
        _wire_shader, "LINES", {"pos": co[edges.reshape(-1)]}
    )

    _fill_batch = None
    if _face_area is not None and len(_face_area) == len(tris):
        uv_pts = co[tris]
        colors = _stretch_colors(uv_pts)
        if colors is not None:
            _fill_batch = batch_for_shader(
                _fill_shader, "TRIS", {"pos": uv_pts.reshape(-1, 2), "color": colors}
            )


def _draw():
    gpu.state.blend_set("ALPHA")
    if _fill_batch is not None:
        _fill_batch.draw(_fill_shader)
    if _wire_batch is not None:
        _wire_shader.bind()
        alpha = 0.4 if _fill_batch is not None else 0.8
        _wire_shader.uniform_float("color", (1.0, 1.0, 1.0, alpha))
        _wire_batch.draw(_wire_shader)
    gpu.state.blend_set("NONE")


def start_viewer_draw():
    global _handler
    if _handler is None:
        _handler = bpy.types.SpaceImageEditor.draw_handler_add(
            _draw, (), "WINDOW", "POST_VIEW"
        )


def stop_viewer_draw():
    global _handler, _wire_batch, _fill_batch, _face_area
    if _handler is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(_handler, "WINDOW")
        _handler = None
    _wire_batch = None
    _fill_batch = None
    _face_area = None


class UVGAMI_OT_view_unwrap(bpy.types.Operator):
    bl_idname = "uvgami.view_unwrap"
    bl_label = "View Unwrap"
    bl_description = (
        "Show the unwrap's uv map live in the UV editor, which will be"
        " opened automatically"
    )

    index: bpy.props.IntProperty()

    def execute(self, context):
        unwrap = manager.active[self.index]
        manager.is_viewer_active = True
        manager.exit_viewer = False

        global old_ui, old_mode
        old_ui = None
        if context.area.ui_type != "UV":
            old_ui = context.area.ui_type
            context.area.ui_type = "UV"
        bpy.ops.image.view_all(fit_view=True)

        # an edit mode mesh would draw its own uv map under the snapshot
        old_mode = None
        active = context.view_layer.objects.active
        if active is not None and active.mode != "OBJECT":
            old_mode = (active, active.mode)
            bpy.ops.object.mode_set(mode="OBJECT")

        load_input_mesh(unwrap.path)
        start_viewer_draw()
        manager.current_viewer = unwrap
        unwrap.viewing = True
        self.report({"INFO"}, "Click to exit viewer")
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type in {"LEFTMOUSE", "RIGHTMOUSE", "ESC"} or manager.exit_viewer:
            manager.is_viewer_active = False
            if manager.current_viewer is not None:
                manager.current_viewer.viewing = False
                manager.current_viewer = None
            stop_viewer_draw()

            if old_ui is not None:
                context.area.ui_type = old_ui
            if old_mode is not None:
                obj, mode = old_mode
                # a finished unwrap can have hidden or removed the source
                if check_exists(obj) and obj.visible_get() and obj.mode == "OBJECT":
                    context.view_layer.objects.active = obj
                    bpy.ops.object.mode_set(mode=mode)

            return {"FINISHED"}

        # let navigation through so the map can be panned and zoomed
        return {"PASS_THROUGH"}
