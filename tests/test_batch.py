import importlib.util
import sys
import time
from pathlib import Path

# loaded from file so importing doesn't touch the blender addon package
spec = importlib.util.spec_from_file_location(
    "addon_batch", Path(__file__).parents[1] / "src" / "batch.py"
)
addon_batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addon_batch)


def start(script):
    return addon_batch.BatchProcess([sys.executable, "-c", script])


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
