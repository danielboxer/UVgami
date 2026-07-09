import importlib.util
import sys
import time
from collections import deque
from pathlib import Path

# loaded from file so importing doesn't touch the blender addon package
spec = importlib.util.spec_from_file_location(
    "addon_batch", Path(__file__).parents[1] / "src" / "batch.py"
)
addon_batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addon_batch)


def start(script, sinks=None):
    return addon_batch.BatchProcess([sys.executable, "-c", script], sinks=sinks)


class Sink:
    """Bare stand-in for an Unwrap as an EngineOutput target."""

    def __init__(self):
        self.progress_data = deque()
        self.uv_co = deque()
        self.uv_indices = deque()
        self.is_uv_data_ready = False
        self.charts = deque()


def feed(lines, sink=None):
    sink = sink or Sink()
    parser = addon_batch.EngineOutput(sink)
    for line in lines:
        parser.feed(line + "\n")
    return sink


def wait_result(batch_process, stem, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code = batch_process.poll_result(stem)
        if code is not None:
            return code
        time.sleep(0.05)
    raise AssertionError(f"no result for {stem}")


def test_markers_reported():
    batch_process = start(
        "print('start: a');print('done: a');print('failed: b -2')"
    )
    assert wait_result(batch_process, "a") == 0
    assert wait_result(batch_process, "b") == -2
    assert "a" in batch_process.started


def test_unknown_lines_ignored():
    batch_process = start(
        "print('loading model');print('start: a');print('done: a')"
    )
    assert wait_result(batch_process, "a") == 0


def test_missing_marker_fails_after_exit():
    batch_process = start("print('done: a')")
    assert wait_result(batch_process, "a") == 0
    assert wait_result(batch_process, "b") != 0


def test_pending_while_running():
    batch_process = start(
        "import time;print('start: a',flush=True);time.sleep(30)"
    )
    try:
        deadline = time.monotonic() + 10
        while "a" not in batch_process.started:
            assert time.monotonic() < deadline, "start marker never arrived"
            time.sleep(0.05)
        assert batch_process.poll_result("a") is None
    finally:
        batch_process.process.kill()


def test_parser_uvgami_snapshot():
    sink = feed(
        [
            "progress: 0.5 0.3 0.2",
            "visual_begin:",
            "vt 0.1 0.2",
            "vt 0.3 0.4",
            "f 0 1 1",
            "visual_end:",
        ]
    )
    assert sink.progress_data.popleft() == "0.5 0.3 0.2\n"
    assert list(sink.uv_co) == [(0.1, 0.2), (0.3, 0.4)]
    assert list(sink.uv_indices) == [(0, 1, 1)]
    assert sink.is_uv_data_ready


def test_parser_new_snapshot_replaces_old():
    sink = feed(["visual_begin:", "vt 0.1 0.2", "f 0 0 0", "visual_end:"])
    feed(["visual_begin:", "vt 0.5 0.6", "f 0 0 0", "visual_end:"], sink)
    assert list(sink.uv_co) == [(0.5, 0.6)]


def test_parser_charts():
    chart = ["v 0 0 0", "v 1 0 0", "v 0 1 0", "vt 0 0", "vt 1 0", "vt 0 1", "f 0 1 2"]
    sink = feed(
        [
            "chart_begin:",
            *chart,
            "PART: 0 created with charts: 1",
            "chart_end:",
            "progress: 0.4 0 0",
        ]
    )
    # stray engine prints inside the block are dropped, geometry is kept raw
    assert list(sink.charts) == ["".join(line + "\n" for line in chart)]
    assert sink.progress_data.popleft() == "0.4 0 0\n"


def test_parser_ignores_loose_geometry_lines():
    # partuv logs outside chart blocks must not be parsed as uv data
    sink = feed(["f 0 1 2", "vt 0.5 0.5", "chart_end:", "found 3 parts"])
    assert not sink.uv_co
    assert not sink.uv_indices
    assert not sink.charts


def test_batch_routes_output_to_sinks():
    sinks = {"a": Sink(), "b": Sink()}
    batch_process = start(
        "print('start: a');"
        "print('chart_begin:');print('v 0 0 0');print('chart_end:');"
        "print('progress: 0.5 0 0');"
        "print('done: a');"
        "print('start: b');"
        "print('chart_begin:');print('v 1 1 1');print('chart_end:');"
        "print('done: b')",
        sinks,
    )
    assert wait_result(batch_process, "a") == 0
    assert wait_result(batch_process, "b") == 0
    assert list(sinks["a"].charts) == ["v 0 0 0\n"]
    assert sinks["a"].progress_data.popleft() == "0.5 0 0\n"
    assert list(sinks["b"].charts) == ["v 1 1 1\n"]
    assert not sinks["b"].progress_data
