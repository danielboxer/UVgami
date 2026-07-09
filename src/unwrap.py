# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import collections
import pathlib
import subprocess
import threading
import time

import bmesh
import bpy
import mathutils

from .batch import EngineOutput
from .logger import logger
from .manager import manager
from .utils.io import print_stdin
from .utils.mesh import check_exists
from .utils.paths import get_extension_dir_path

# grid layout for incoming partuv charts in the viewer
CHART_COLS = 4
CHART_CELL = 0.25
CHART_MARGIN = 0.02


class Unwrap:
    def __init__(
        self,
        name: str,
        input_name: str,
        path: pathlib.Path,
        guide_path: pathlib.Path,
        edge_path: pathlib.Path,
        jobs: tuple,
        origin: mathutils.Vector,
        materials: list,
        added_edges: list,
        vertex_count: int,
        material_indices: list,
        vertex_groups: dict,
        shade_smooth: bool,
        auto_smooth: int,
        merge_cuts: bool,
    ):
        # unwrap name
        self.name = name
        self.input_name = input_name

        # paths
        self.path = path
        self.output_path = get_extension_dir_path() / "output" / f"{self.path.stem}.obj"
        # seam restrictions
        self.guide_path = guide_path
        # for untriangulation (added edges)
        self.edge_path = edge_path

        # jobs
        self.jobs = [j for j in jobs if j is not None]
        self.preserve_job = jobs[0]
        self.join_job = jobs[1]
        self.cleanup_job = jobs[2]
        self.symmetrize_job = jobs[3]
        self.transfer_uvs_job = jobs[4]

        # object info
        self.origin = mathutils.Vector(origin)
        self.materials = materials
        self.added_edges = added_edges
        self.vertex_count = vertex_count
        self.material_indices = material_indices
        self.vertex_groups = vertex_groups
        self.shade_smooth = shade_smooth
        self.auto_smooth = auto_smooth

        # other
        self.merge_cuts = merge_cuts

        # unwrap state
        self.is_active = False
        self.progress = (0, 0, 1)
        # unwrap process, shared with other unwraps when part of a batch process
        self.process = None
        self.batch_process = None
        # copy of input obj used for viewing
        self.viewer_obj = None
        self.viewing = False
        self.view_update_count = 0
        self.progress_data = collections.deque()
        self.uv_co = collections.deque()
        self.uv_indices = collections.deque()
        self.is_uv_data_ready = False
        # finished partuv charts as raw chart_begin block strings; consumed
        # into applied_charts once shown in the viewer
        self.charts = collections.deque()
        self.applied_charts = []
        self.is_stopped = False
        self.stop_requested_at = None

    def start_unwrap(self):
        props = bpy.context.scene.uvgami
        args = manager.engine.build_args(manager.engine_path, self.path, props)

        self.process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            universal_newlines=True,
            env=manager.engine.build_env(manager.engine_path),
        )

        # start reading thread
        thread = threading.Thread(target=self.get_output)
        thread.start()

        self.is_active = True
        self.started_at = time.monotonic()

    def join_batch(self, batch_process):
        """Run inside a shared batch process instead of spawning our own."""
        self.batch_process = batch_process
        self.process = batch_process.process
        self.is_active = True

    def poll_engine(self):
        """None while running, 0 on success, or a failure code."""
        if self.batch_process is None:
            return self.process.poll()
        # the engine reports when it reaches each mesh, the timeout clock
        # starts then
        if not hasattr(self, "started_at") and self.path.stem in self.batch_process.started:
            self.started_at = time.monotonic()
        return self.batch_process.poll_result(self.path.stem)

    def stop_process(self):
        """Hard stop: for a batch member this kills the whole batch process."""
        if self.process is not None and self.process.poll() is None:
            manager.engine.stop(self.process, manager.engine_path)

    def release_engine(self):
        """This unwrap no longer needs the engine. A batch process is left
        running for the other meshes; deleting our input file in cleanup()
        is what makes the cli skip this mesh."""
        if self.batch_process is None:
            self.stop_process()

    def get_output(self):
        parser = EngineOutput(self)
        # get lines until there are no more left
        for line in iter(self.process.stdout.readline, ""):
            parser.feed(line)
        # process has ended, thread will exit here

    def update_progress(self):
        """Read progress from the stdout reader thread."""
        if len(self.progress_data) > 0:
            # only the newest queued line matters
            progress = self.progress_data.pop()
            self.progress_data.clear()
            try:
                self.progress = tuple(float(num) for num in progress.split())
            except ValueError:
                # invalid progress string
                return

    def update_viewer(self):
        if manager.engine.viewer_push:
            self._apply_charts()
            return
        print_stdin(self.process, "snapshot")
        if self.is_uv_data_ready:
            uvs = list(self.uv_co)
            uv_idcs = list(self.uv_indices)
            self.is_uv_data_ready = False

            # need to use from_edit_mesh here so mesh is updated in edit mode
            bm = bmesh.from_edit_mesh(self.viewer_obj.data)
            uv_map = bm.loops.layers.uv.verify()

            for face in bm.faces:
                # set uvs
                uv_idx_triple = uv_idcs[face.index]
                face.loops[0][uv_map].uv = uvs[uv_idx_triple[0]]
                face.loops[1][uv_map].uv = uvs[uv_idx_triple[1]]
                face.loops[2][uv_map].uv = uvs[uv_idx_triple[2]]

            # need to use update_edit_mesh, don't call bm.free(), it will crash
            bmesh.update_edit_mesh(self.viewer_obj.data)

    def _apply_charts(self):
        """Add newly finished partuv charts to the viewer mesh."""
        if not self.charts:
            return
        bm = bmesh.from_edit_mesh(self.viewer_obj.data)
        uv_map = bm.loops.layers.uv.verify()
        while self.charts:
            chart = self.charts.popleft()
            self._add_chart(bm, uv_map, chart, len(self.applied_charts))
            self.applied_charts.append(chart)
        bmesh.update_edit_mesh(self.viewer_obj.data)

    def _add_chart(self, bm, uv_map, chart, index):
        verts, uvs, tris = [], [], []
        for line in chart.splitlines():
            values = line.split()
            if values[0] == "v":
                verts.append((float(values[1]), float(values[2]), float(values[3])))
            elif values[0] == "vt":
                uvs.append((float(values[1]), float(values[2])))
            elif values[0] == "f":
                tris.append((int(values[1]), int(values[2]), int(values[3])))
        if not tris or len(uvs) != len(verts):
            return
        # fit the chart into its own grid cell so charts don't stack
        min_u = min(u for u, _ in uvs)
        min_v = min(v for _, v in uvs)
        size = max(max(u for u, _ in uvs) - min_u, max(v for _, v in uvs) - min_v)
        scale = (CHART_CELL - 2 * CHART_MARGIN) / max(size, 1e-12)
        cell_u = (index % CHART_COLS) * CHART_CELL + CHART_MARGIN
        cell_v = (index // CHART_COLS) * CHART_CELL + CHART_MARGIN

        bm_verts = [bm.verts.new(co) for co in verts]
        for tri in tris:
            face = bm.faces.new([bm_verts[i] for i in tri])
            for loop, vert_idx in zip(face.loops, tri):
                u, v = uvs[vert_idx]
                loop[uv_map].uv = (
                    cell_u + (u - min_u) * scale,
                    cell_v + (v - min_v) * scale,
                )

    def cleanup(self):
        """Clean up files and viewer objects."""
        try:
            if self.path.is_file():
                self.path.unlink()
            if self.guide_path is not None and self.guide_path.is_file():
                self.guide_path.unlink()
        except PermissionError:
            logger.add_data("errors", "Error deleting file")
        if self.viewer_obj is not None and check_exists(self.viewer_obj):
            bpy.data.objects.remove(self.viewer_obj, do_unlink=True)
