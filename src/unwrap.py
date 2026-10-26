import collections
import pathlib
import time

import mathutils

from .batch import EngineOutput, EngineProcess
from .logger import logger
from .manager import manager
from .ops.viewer import set_snapshot
from .utils.paths import engine_output_path
from .utils.ui import tag_redraw

# a healthy run's progress changes every few seconds
PROGRESS_STALL_SECONDS = 120


class Unwrap:
    def __init__(
        self,
        name: str,
        input_name: str,
        path: pathlib.Path,
        *,
        join_job=None,
        hide_job=None,
        symmetrize_job=None,
        transfer_uvs_job=None,
        segment_weld_job=None,
        preseeded: bool = False,
    ):
        self.name = name
        self.input_name = input_name
        self.preseeded = preseeded

        self.path = path
        self.output_path = engine_output_path(self.path)

        # the export sets this when Keep Quads triangulates the mesh
        self.preserve_job = None
        self.join_job = join_job
        if self.join_job is not None:
            self.join_job.members.append(self)
        self.hide_job = hide_job
        self.symmetrize_job = symmetrize_job
        self.transfer_uvs_job = transfer_uvs_job
        self.segment_weld_job = segment_weld_job

        # a duplicate skips the engine and takes copy_of's output moved by copy_matrix
        self.copy_of = None
        self.copy_matrix = None
        # a reordered copy's indices don't line up with that output
        self.copy_reordered = False
        self.twins = []

        # set once by manager.record_result
        self.result = None

        self.has_uvs = False
        # a part override skips the engine and keeps the part's uvs
        self.keeps_uvs = False
        # self.priority_override = None
        self.matrix = None

        self.is_exported = False
        self.guide_path = None
        self.origin = None
        self.materials = []
        self.added_edges = []
        self.vertex_count = 0
        self.material_indices = []
        self.vertex_groups = {}
        self.face_smooth = []
        self.shading_modifiers = []
        self.overrides = {}

        self.is_active = False
        self.progress = (0, 0, 1)
        self.has_reported = False
        # shared with the other unwraps when part of a batch process
        self.process = None
        self.batch_process = None
        self.viewing = False
        self.progress_data = collections.deque()
        self.uv_co = collections.deque()
        self.uv_indices = collections.deque()
        self.is_uv_data_ready = False
        self.is_stopped = False
        self.progress_changed_at = None
        self.started_at = None
        self.stop_requested_at = None
        # None for a batch member
        self._engine_process = None
        self._output = EngineOutput(self)

    # a transfer writes onto the input and deletes the output
    @property
    def keeps_output(self):
        return self.transfer_uvs_job is None

    # the defaults cover a fix export, which carries no mesh metadata
    def set_export_data(
        self,
        *,
        origin,
        vertex_count,
        guide_path=None,
        materials=(),
        added_edges=(),
        material_indices=(),
        vertex_groups=None,
        face_smooth=(),
        shading_modifiers=(),
        overrides=None,
    ):
        self.guide_path = guide_path
        self.origin = mathutils.Vector(origin)
        self.materials = materials
        self.added_edges = added_edges
        self.vertex_count = vertex_count
        self.material_indices = material_indices
        self.vertex_groups = vertex_groups or {}
        self.face_smooth = face_smooth
        self.shading_modifiers = shading_modifiers
        self.overrides = overrides or {}
        self.is_exported = True

    def start_unwrap(self):
        args = manager.engine.build_args(manager.engine_ctx, self.path, manager.props)

        self._engine_process = EngineProcess(
            args, manager.engine.build_env(manager.engine_ctx)
        )
        self.process = self._engine_process.process

        self.is_active = True
        self.started_at = time.monotonic()

    def join_batch(self, batch_process):
        self.batch_process = batch_process
        self.process = batch_process.process
        self.is_active = True

    def leave_batch(self):
        self.batch_process = None
        self.process = None
        self.is_active = False

    # None while running, 0 on success, or a failure code
    def poll_engine(self):
        if self.batch_process is None:
            return self.process.poll()
        # the engine reports when it reaches each mesh
        if self.started_at is None and self.path.stem in self.batch_process.started:
            self.started_at = time.monotonic()
        return self.batch_process.poll_result(self.path.stem)

    # for a batch member this kills the whole batch process
    def stop_process(self):
        if self.process is not None and self.process.poll() is None:
            manager.engine.stop(self.process, manager.engine_ctx)

    # the process stays alive for the rest of the queue
    def cancel_solve(self):
        if self.batch_process is not None and self.is_solving:
            manager.engine.request_cancel(self.process)

    # deleting the input file in cleanup is what makes the cli skip this mesh
    def release_engine(self):
        if self.batch_process is not None:
            return
        self.stop_process()
        if self.process is None:
            return
        # without this a model with many pieces runs out of open files
        try:
            self.process.stdin.close()
        except OSError:
            # the flush fails on a dead engine but stdin still closes
            pass
        self.process.wait()
        self._engine_process.close()

    def get_stderr_tail(self):
        if self.batch_process is not None:
            return self.batch_process.stderr_lines()
        return self._engine_process.stderr_tail()

    def read_output(self):
        if self.batch_process is not None:
            self.batch_process.read_output()
            return
        for line in self._engine_process.read_lines():
            self._output.feed(line)

    def update_progress(self):
        if len(self.progress_data) > 0:
            # only the newest queued line matters
            progress = self.progress_data.pop()
            self.progress_data.clear()
            try:
                parsed = tuple(float(num) for num in progress.split())
            except ValueError:
                return
            self.has_reported = True
            if parsed != self.progress:
                self.progress = parsed
                self.progress_changed_at = time.monotonic()

    # a batch process takes the whole queue when it spawns
    @property
    def is_running(self):
        if self.batch_process is None:
            return self.is_active
        return self.is_active and self.path.stem in self.batch_process.started

    # a stdin stop reaches this mesh, not the process's next one
    @property
    def is_solving(self):
        if not self.is_running:
            return False
        if self.batch_process is None:
            return self.process.poll() is None
        return self.batch_process.poll_result(self.path.stem) is None

    # False on engines that never report progress
    @property
    def is_stalled(self):
        return (
            self.is_active
            and self.progress_changed_at is not None
            and time.monotonic() - self.progress_changed_at > PROGRESS_STALL_SECONDS
        )

    # an engine that has reported has a uv map to stop with
    @property
    def is_stoppable(self):
        return self.is_running and self.has_reported

    # an all-high-distortion report is the 0 0 1 the progress numbers start at
    @property
    def is_viewable(self):
        return self.is_active and self.has_reported

    def update_viewer(self):
        manager.engine.request_snapshot(self.process)
        if self.is_uv_data_ready:
            uvs = list(self.uv_co)
            uv_idcs = list(self.uv_indices)
            self.is_uv_data_ready = False
            if uvs and uv_idcs:
                set_snapshot(uvs, uv_idcs)
                tag_redraw(("WINDOW",))

    def cleanup(self):
        try:
            if self.path.is_file():
                self.path.unlink()
            if self.guide_path is not None and self.guide_path.is_file():
                self.guide_path.unlink()
            # a later piece can reuse the stem and would read these
            for suffix in ("_plane", "_locked", "_fixed"):
                sidecar = self.path.parent / f"{self.path.stem}{suffix}"
                if sidecar.is_file():
                    sidecar.unlink()
        except PermissionError:
            logger.add_data("errors", "Error deleting file")
