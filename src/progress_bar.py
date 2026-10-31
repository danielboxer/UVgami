import bpy
import gpu
from gpu_extras.batch import batch_for_shader

IDLE_PROGRESS = (0.0, 0.0, 1.0)
# gpu is unavailable at import time in background mode
SHADER = None


def _get_shader():
    global SHADER
    if SHADER is None:
        SHADER = gpu.shader.from_builtin("UNIFORM_COLOR")
    return SHADER


class ProgressBar:
    def __init__(self):
        self._batch = [None, None, None]
        self._handle = [None, None, None]
        self._space = None
        self.is_active = False

    def _draw(self, index):
        space = bpy.context.space_data
        # IMAGE_EDITOR is also the space type of the image viewer and the paint modes
        if space.type == "IMAGE_EDITOR" and space.mode != "UV":
            return
        shader = _get_shader()
        shader.bind()
        segment_colours = [
            (0.0, 0.0, 0.7355, 1),
            (0.6, 1.0, 0.6, 1),
            (1.0, 0.0, 0.0, 1),
        ]
        shader.uniform_float("color", segment_colours[index])
        self._batch[index].draw(shader)

    def start(self, *, uv_editor):
        # registering again leaks the old draw handlers
        if self.is_active:
            return
        self.is_active = True
        self._space = bpy.types.SpaceImageEditor if uv_editor else bpy.types.SpaceView3D
        self.update(IDLE_PROGRESS)
        for index in range(3):
            self._handle[index] = self._space.draw_handler_add(
                self._draw, (index,), "WINDOW", "POST_PIXEL"
            )

    def update(self, percentages):
        left = 25
        bottom = 25
        top = bottom + 5
        width = 150
        start = left
        for index in range(3):
            end = (width * percentages[index]) + start
            self._batch[index] = batch_for_shader(
                _get_shader(),
                "TRIS",
                {"pos": ((start, bottom), (end, bottom), (start, top), (end, top))},
                indices=((0, 1, 2), (2, 1, 3)),
            )
            start = end

    def remove(self):
        if self.is_active:
            self.is_active = False
            for index in range(3):
                self._space.draw_handler_remove(self._handle[index], "WINDOW")


progress_bar = ProgressBar()
