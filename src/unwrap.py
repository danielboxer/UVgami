import collections
import pathlib
import time

from .batch import EngineOutput, EngineProcess
from .logger import logger
from .manager import manager
from .ops.viewer import set_snapshot
from .progress_bar import IDLE_PROGRESS
from .utils.paths import ENGINE_FILE_SUFFIXES, engine_output_path
from .utils.ui import tag_redraw

# progress on healthy runs changes every few seconds
PROGRESS_STALL_SECONDS = 120


ExportData = collections.namedtuple(
    "ExportData",
    [
        "origin",
        "vertex_count",
        "materials",
        "added_edges",
        "material_indices",
        "vertex_groups",
        "face_smooth",
        "shading_modifiers",
        "overrides",
    ],
    defaults=((), (), (), {}, (), (), {}),
)


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

        self.preserve_job = None
        self.join_job = join_job
        if self.join_job is not None:
            self.join_job.members.append(self)
        self.hide_job = hide_job
        self.symmetrize_job = symmetrize_job
        self.transfer_uvs_job = transfer_uvs_job
        self.segment_weld_job = segment_weld_job

        self.copy_of = None
        self.copy_matrix = None
        self.copy_reordered = False
        self.twins = []

        self.result = None

        self.has_uvs = False
        self.keeps_uvs = False
        # self.priority_override = None
        self.matrix = None

        self.export = None

        self.has_process = False
        self.progress = IDLE_PROGRESS
        self.has_reported = False
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
        self._engine_process = None
        self._output = EngineOutput(self)

    @property
    def keeps_output(self):
        return self.transfer_uvs_job is None

    @property
    def is_exported(self):
        return self.export is not None

    def start_unwrap(self):
        args = manager.engine.build_args(manager.engine_ctx, self.path, manager.props)

        self._engine_process = EngineProcess(
            args, manager.engine.build_env(manager.engine_ctx)
        )
        self.process = self._engine_process.process

        self.has_process = True
        self.started_at = time.monotonic()

    def join_batch(self, batch_process):
        self.batch_process = batch_process
        self.process = batch_process.process
        self.has_process = True

    def leave_batch(self):
        self.batch_process = None
        self.process = None
        self.has_process = False

    def poll_engine(self):
        if self.batch_process is None:
            return self.process.poll()
        if (
            self.started_at is None
            and self.path.stem in self.batch_process.started_stems()
        ):
            self.started_at = time.monotonic()
        return self.batch_process.poll_result(self.path.stem)

    def stop_process(self):
        if self.process is not None and self.process.poll() is None:
            manager.engine.stop(self.process, manager.engine_ctx)

    def cancel_solve(self):
        if self.batch_process is not None and self.is_solving:
            manager.engine.request_cancel(self.process)

    def release_engine(self):
        if self.batch_process is not None:
            # batch processes do not unwrap a mesh whose input file cleanup deleted
            return
        self.stop_process()
        if self.process is None:
            return
        # without this a model with many pieces runs out of open files
        try:
            self.process.stdin.close()
        except OSError:
            # stdin.close() raises on an exited engine but still closes stdin
            pass
        self.process.wait()
        self._engine_process.close()

    def get_stderr_tail(self):
        if self.batch_process is not None:
            return self.batch_process.engine_process.stderr_tail()
        return self._engine_process.stderr_tail()

    def read_output(self):
        if self.batch_process is not None:
            self.batch_process.read_output()
            return
        for line in self._engine_process.read_lines():
            self._output.feed(line)

    def update_progress(self):
        if len(self.progress_data) > 0:
            progress = self.progress_data.pop()
            self.progress_data.clear()
            try:
                parsed = tuple(float(number) for number in progress.split())
            except ValueError:
                return
            self.has_reported = True
            if parsed != self.progress:
                self.progress = parsed
                self.progress_changed_at = time.monotonic()

    @property
    def is_running(self):
        if self.batch_process is None:
            return self.has_process
        return self.has_process and self.path.stem in self.batch_process.started_stems()

    @property
    def is_solving(self):
        if not self.is_running:
            return False
        if self.batch_process is None:
            return self.process.poll() is None
        # stdin stops apply only to the mesh in progress
        return self.batch_process.poll_result(self.path.stem) is None

    @property
    def is_stalled(self):
        return (
            self.has_process
            and self.progress_changed_at is not None
            and time.monotonic() - self.progress_changed_at > PROGRESS_STALL_SECONDS
        )

    @property
    def is_stoppable(self):
        return self.is_running and self.has_reported

    @property
    def is_viewable(self):
        # all-high-distortion reports are 0 0 1, the same as the starting progress
        return self.has_process and self.has_reported

    def update_viewer(self):
        manager.engine.request_snapshot(self.process)
        if self.is_uv_data_ready:
            uvs = list(self.uv_co)
            triangles = list(self.uv_indices)
            self.is_uv_data_ready = False
            if uvs and triangles:
                set_snapshot(uvs, triangles)
                tag_redraw(("WINDOW",))

    def cleanup(self):
        try:
            # later pieces can reuse the stem and read these
            for suffix in ENGINE_FILE_SUFFIXES:
                (self.path.parent / f"{self.path.stem}{suffix}").unlink(missing_ok=True)
        except PermissionError:
            logger.add_error("Error deleting file")
