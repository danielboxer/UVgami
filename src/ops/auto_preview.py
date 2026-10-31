from collections import namedtuple

import bpy
import gpu
import numpy
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector

from ..engines import active_engine
from ..job import unwrap_symmetry
from ..manager import SettingsCopy, manager
from ..no_bpy import SURFACE_MODE_CODES, EngineError
from ..no_bpy.mesh import face_starts
from ..no_bpy.preseed import SegmentRegions, classify_job
from ..no_bpy.worker import WorkerError, WorkerProcess
from ..part_overrides import KEEP_ATTRIBUTE, face_values, surface_modes
from ..proxy import part_triangles
from ..symmetry import mirror_segmentation, paints_seams
from ..utils.mesh import (
    applied_matrix,
    lifted_triangles,
    loop_vertices_and_face_sizes,
    mesh_arrays,
    vertex_position_array,
    world_positions,
)
from ..utils.paths import get_extension_dir_path
from ..utils.ui import count_label, tag_redraw
from ..utils.uvgami_tools import tools_command

ORGANIC_COLOR = (0.25, 0.75, 0.35, 0.4)
HARD_COLOR = (0.95, 0.5, 0.1, 0.4)
BORDER_COLOR = (0.05, 0.05, 0.05, 1.0)
AXIS_COLORS = {
    "X": (1.0, 0.23, 0.33),
    "Y": (0.54, 0.83, 0.0),
    "Z": (0.16, 0.56, 0.9),
}

Shape = namedtuple("Shape", ["kind", "positions", "indices", "color", "nudge"])
Preview = namedtuple("Preview", ["shapes", "lines"])
_preview = None
_batches = None
_draw_handler = None
_shader = None
_line_shader = None


class PreviewRun:
    def __init__(self, obj, engine, props):
        self.name = obj.name
        self.engine = engine
        self.props = props
        self.mesh = evaluated_mesh(obj)
        self.symmetry = unwrap_symmetry(
            engine, props, self.mesh, painted=paints_seams(obj, engine, props)
        )
        try:
            self.worker = self._start_worker(applied_matrix(obj))
        except EngineError:
            bpy.data.meshes.remove(self.mesh)
            raise
        self.plane = None
        self.local_positions = vertex_position_array(self.mesh)
        if self.symmetry is not None:
            self.plane = plane_corners(
                self.symmetry, self.local_positions, obj.matrix_world
            )
        self.mesh.transform(obj.matrix_world)
        self.cancelled = False

    def tick(self):
        global _preview
        if self.cancelled:
            return None
        if self.worker is not None and not self.worker.done():
            return 0.1
        try:
            _preview = build_preview(
                self.name,
                self.mesh,
                self.engine,
                self.props,
                self.symmetry,
                self.plane,
                self._segments(),
            )
        except (EngineError, WorkerError) as error:
            _preview = Preview([], [self.name, f"Failed, {error}"])
        finally:
            bpy.data.meshes.remove(self.mesh)
            end_run()
        return None

    def _start_worker(self, matrix):
        unset_mode = self.engine.unset_surface_mode(self.props)
        if self.engine.splits_regions(self.props):
            from ..segmentation import segment_work

            return segment_work(
                self.mesh,
                world_positions(self.mesh, matrix),
                tools_command(),
                unset_mode,
            )
        if not self.engine.supports_hard_surface:
            return None
        return WorkerProcess(
            classify_job,
            tools_command(),
            get_extension_dir_path() / "segment",
            *mesh_arrays(self.mesh),
            surface_modes(self.mesh, unset_mode),
        )

    def _segments(self):
        if self.worker is None:
            return None
        result = self.worker.result()
        if self.engine.splits_regions(self.props):
            _, segments = mirror_segmentation(
                self.local_positions,
                *loop_vertices_and_face_sizes(self.mesh),
                result,
                self.symmetry,
            )
            return segments
        return SegmentRegions(
            result, [], 0, numpy.zeros(len(result), dtype=numpy.int64)
        )

    def cancel(self):
        # bpy.app.timers unregisters a callback that returns None
        self.cancelled = True
        if self.worker is not None:
            self.worker.close()
        bpy.data.meshes.remove(self.mesh)


_run = None


def end_run():
    global _run
    _run = None
    tag_redraw()


def stop_run():
    if _run is None:
        return
    _run.cancel()
    end_run()


def evaluated_mesh(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph))


FILL_NUDGE = 1e-3
LINE_NUDGE = 1e-2
PLANE_TRIANGLES = ((0, 1, 2), (0, 2, 3))
PLANE_OUTLINE = ((0, 1), (1, 2), (2, 3), (3, 0))


def build_preview(name, mesh, engine, props, symmetry, plane, segments):
    positions = vertex_position_array(mesh)
    shapes = []
    lines = [name]
    part_counts = part_triangles(mesh)
    decimated = 0
    if engine.uses_proxy(props):
        decimated = int((part_counts.triangle_counts > props.proxy_faces).sum())
    kept = face_values(mesh, KEEP_ATTRIBUTE) != 0
    part_count = len(part_counts.part_labels)
    parts = count_label(part_count, "part")

    if engine.supports_hard_surface:
        hard = segments.surface_modes == SURFACE_MODE_CODES["HARD"]
        parts += (
            f": {mode_counts(mesh, part_counts.vertex_labels, part_count, kept, hard)}"
        )
        mesh.calc_loop_triangles()
        corners, triangle_faces = lifted_triangles(mesh, positions, 0.0)
        for faces, color in (
            (~kept & ~hard, ORGANIC_COLOR),
            (~kept & hard, HARD_COLOR),
        ):
            shown = faces[triangle_faces]
            if shown.any():
                shown_corners = corners[shown].reshape(-1, 3)
                shapes.append(Shape("TRIS", shown_corners, None, color, FILL_NUDGE))
    lines.append(parts)

    if segments is not None and segments.region_count:
        lines.append(f"{segments.region_count} regions")
        edges = numpy.array(segments.border_edges, dtype=numpy.int64).reshape(-1, 2)
        if len(edges):
            shapes.append(
                Shape("LINES", positions[edges.ravel()], None, BORDER_COLOR, LINE_NUDGE)
            )

    if symmetry is not None:
        color = AXIS_COLORS[symmetry.axis]
        shapes.append(Shape("TRIS", plane, PLANE_TRIANGLES, (*color, 0.25), 0.0))
        shapes.append(Shape("LINES", plane, PLANE_OUTLINE, (*color, 0.9), 0.0))
        lines.append(
            f"Mostly symmetrical on {symmetry.axis}"
            if symmetry.kind == "WHOLE"
            else f"Symmetrical on {symmetry.axis}"
        )
    elif engine.uses_symmetry(props) and len(positions) > 0:
        lines.append("Not symmetrical")

    if decimated:
        lines.append(f"{decimated} parts decimated to {props.proxy_faces} triangles")

    return Preview(shapes, lines)


def mode_counts(mesh, labels, part_count, kept, hard):
    corners, totals = loop_vertices_and_face_sizes(mesh)
    face_labels = labels[corners[face_starts(totals)]]
    hard_count = len(numpy.unique(face_labels[hard]))
    kept_count = len(numpy.unique(face_labels[kept]))
    organic_count = part_count - hard_count - kept_count
    counts = [f"{organic_count} organic", f"{hard_count} hard surface"]
    if kept_count:
        counts.append(f"{kept_count} keep UVs")
    return ", ".join(counts)


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


def _shape_shader(shape):
    return _line_shader if shape.kind == "LINES" else _shader


def _shape_batch(shape):
    # float64 buffers go into the vertex buffer as raw float32 bytes
    positions = numpy.ascontiguousarray(shape.positions, dtype=numpy.float32)
    shader = _shape_shader(shape)
    if shape.indices is None:
        return batch_for_shader(shader, shape.kind, {"pos": positions})
    return batch_for_shader(
        shader, shape.kind, {"pos": positions}, indices=shape.indices
    )


LINE_WIDTH = 5.0


def _draw():
    global _shader, _line_shader, _batches
    if _preview is None:
        return
    if _shader is None:
        _shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        # polyline sets width in the shader, past the driver's line limit of 1
        _line_shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("LESS_EQUAL")
    _line_shader.bind()
    _line_shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
    _line_shader.uniform_float("lineWidth", LINE_WIDTH)
    _line_shader.uniform_bool("lineSmooth", True)
    if _batches is None:
        _batches = [(shape, _shape_batch(shape)) for shape in _preview.shapes]
    view = gpu.matrix.get_model_view_matrix()
    for shape, batch in _batches:
        shader = _shape_shader(shape)
        shader.bind()
        shader.uniform_float("color", shape.color)
        with gpu.matrix.push_pop():
            gpu.matrix.load_matrix(_toward_camera(shape.nudge) @ view)
            batch.draw(shader)
    gpu.state.depth_test_set("NONE")
    gpu.state.blend_set("NONE")


def _ensure_draw_handler():
    global _draw_handler
    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw, (), "WINDOW", "POST_VIEW"
        )


# def show_preview(preview):
#     global _preview, _batches
#     _ensure_draw_handler()
#     _preview = preview
#     _batches = None
#     tag_redraw(("WINDOW",))


class UVGAMI_OT_preview_auto_settings(bpy.types.Operator):
    bl_idname = "uvgami.preview_auto_settings"
    bl_label = "Preview"
    bl_description = (
        "Show the settings the unwrap uses for the active mesh: the mode of each part,"
        " the region borders, the symmetry plane and the decimated parts"
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            _run is None
            and not manager.is_active
            and context.mode in ("OBJECT", "EDIT_MESH")
            and obj is not None
            and obj.type == "MESH"
        )

    def execute(self, context):
        global _run, _preview, _batches
        props = context.scene.uvgami
        engine = active_engine(props.engine)
        try:
            run = PreviewRun(context.active_object, engine, SettingsCopy(props))
        except EngineError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        _ensure_draw_handler()
        _preview = None
        _batches = None
        _run = run
        bpy.app.timers.register(_run.tick)
        tag_redraw()
        return {"FINISHED"}


class UVGAMI_OT_clear_auto_preview(bpy.types.Operator):
    bl_idname = "uvgami.clear_auto_preview"
    bl_label = "Exit Preview"
    bl_description = "Stop the preview and remove it from the viewport"

    def execute(self, context):
        stop_auto_preview()
        return {"FINISHED"}


def stop_auto_preview():
    global _draw_handler, _preview, _batches
    stop_run()
    _preview = None
    _batches = None
    if _draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, "WINDOW")
        _draw_handler = None
    tag_redraw()


def draw_auto_preview(layout):
    shown = _run is not None or _preview is not None
    box = layout.box() if shown else layout
    row = box.row()
    row.scale_y = 1.5
    if shown:
        row.operator("uvgami.clear_auto_preview", icon="PANEL_CLOSE")
    else:
        row.operator("uvgami.preview_auto_settings", icon="HIDE_OFF")
    if _run is not None:
        box.label(text=f"Previewing {_run.name}", icon="SORTTIME")
    if _preview is None:
        return
    column = box.column(align=True)
    column.active = False
    for line in _preview.lines:
        column.label(text=line)
