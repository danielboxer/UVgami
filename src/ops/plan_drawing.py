from collections import namedtuple

import bpy
import gpu
import numpy
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector

BORDER_COLOR = (0.05, 0.05, 0.05, 1.0)
HOVER_COLOR = (1.0, 0.9, 0.1, 1.0)
AXIS_COLORS = {
    "X": (1.0, 0.23, 0.33),
    "Y": (0.54, 0.83, 0.0),
    "Z": (0.16, 0.56, 0.9),
}

Shape = namedtuple(
    "Shape", ["kind", "positions", "indices", "color", "nudge", "line_width"]
)
_shader = None
_line_shader = None

FILL_NUDGE = 1e-3
LINE_NUDGE = 1e-2
LINE_WIDTH = 5.0
PLANE_TRIANGLES = ((0, 1, 2), (0, 2, 3))
PLANE_OUTLINE = ((0, 1), (1, 2), (2, 3), (3, 0))


def _toward_camera(nudge):
    region_data = bpy.context.region_data
    if region_data.is_perspective:
        # view space has the camera at its origin
        return Matrix.Scale(1 - nudge, 4)
    return Matrix.Translation((0, 0, nudge * region_data.view_distance))


# planes no bigger than the bounding box are hidden under a rounded surface
PLANE_MARGIN = 1.2


def plane_corners(symmetry, positions, matrix):
    half_extent = numpy.ptp(positions, axis=0) / 2
    floor = 0.1 * max(half_extent)
    half = Vector(PLANE_MARGIN * max(value, floor) for value in half_extent)
    if symmetry.axis == "X":
        u, v = Vector((0, half.y, 0)), Vector((0, 0, half.z))
    elif symmetry.axis == "Y":
        u, v = Vector((half.x, 0, 0)), Vector((0, 0, half.z))
    else:
        u, v = Vector((half.x, 0, 0)), Vector((0, half.y, 0))
    center = Vector(symmetry.center)
    corners = (center - u - v, center + u - v, center + u + v, center - u + v)
    return [matrix @ corner for corner in corners]


def plane_shapes(symmetry, positions, matrix):
    plane = plane_corners(symmetry, positions, matrix)
    color = AXIS_COLORS[symmetry.axis]
    return [
        Shape("TRIS", plane, PLANE_TRIANGLES, (*color, 0.25), 0.0, None),
        Shape("LINES", plane, PLANE_OUTLINE, (*color, 0.9), 0.0, LINE_WIDTH),
    ]


def _shaders():
    global _shader, _line_shader
    if _shader is None:
        _shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        # polyline sets width in the shader, past the driver's line limit of 1
        _line_shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    return _shader, _line_shader


def _shape_shader(shape):
    shader, line_shader = _shaders()
    return line_shader if shape.kind == "LINES" else shader


def shape_batch(shape):
    # float64 buffers go into the vertex buffer as raw float32 bytes
    positions = numpy.ascontiguousarray(shape.positions, dtype=numpy.float32)
    shader = _shape_shader(shape)
    if shape.indices is None:
        return batch_for_shader(shader, shape.kind, {"pos": positions})
    return batch_for_shader(
        shader, shape.kind, {"pos": positions}, indices=shape.indices
    )


def draw_batches(items):
    _, line_shader = _shaders()
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("LESS_EQUAL")
    line_shader.bind()
    line_shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
    line_shader.uniform_bool("lineSmooth", True)
    view = gpu.matrix.get_model_view_matrix()
    for shape, batch in items:
        shader = _shape_shader(shape)
        shader.bind()
        shader.uniform_float("color", shape.color)
        if shape.kind == "LINES":
            shader.uniform_float("lineWidth", shape.line_width)
        with gpu.matrix.push_pop():
            gpu.matrix.load_matrix(_toward_camera(shape.nudge) @ view)
            batch.draw(shader)
    gpu.state.depth_test_set("NONE")
    gpu.state.blend_set("NONE")
