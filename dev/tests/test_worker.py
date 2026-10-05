import importlib.util
import sys
from pathlib import Path

import pytest

# loaded from file, the addon package imports bpy
PKG = Path(__file__).parents[2] / "src" / "seams"
spec = importlib.util.spec_from_file_location(
    "seams", PKG / "__init__.py", submodule_search_locations=[str(PKG)]
)
sys.modules["seams"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sys.modules["seams"])
from seams import (  # noqa: E402
    Cancelled,
    FlattenError,
    check_cancelled,
    check_manifold,
    face_edges,
)
from seams.worker import WorkerProcess, last_progress  # noqa: E402

# the venv interpreter stands in for blender's bundled one
PYTHON = (sys.executable, ())

TWO_TRIANGLES = [(0, 1, 2), (1, 3, 2)]
THREE_FACES_ON_ONE_EDGE = [(0, 1, 2), (0, 1, 3), (0, 1, 4)]


def test_result_comes_back():
    worker = WorkerProcess(PYTHON, face_edges, TWO_TRIANGLES)
    assert worker.result() == face_edges(TWO_TRIANGLES)
    assert worker.done()


def test_error_is_raised_as_the_class_the_function_raised():
    worker = WorkerProcess(PYTHON, check_manifold, THREE_FACES_ON_ONE_EDGE)
    with pytest.raises(FlattenError, match="Non Manifold"):
        worker.result()


def test_the_function_sees_the_cancel():
    worker = WorkerProcess(PYTHON, check_cancelled)
    worker.cancel()
    with pytest.raises(Cancelled):
        worker.result()


def test_result_removes_the_worker_folder():
    worker = WorkerProcess(PYTHON, face_edges, TWO_TRIANGLES)
    folder = worker._folder
    assert folder.is_dir()
    worker.result()
    assert not folder.exists()


def test_last_progress_skips_a_half_written_line(tmp_path):
    stdout = tmp_path / "stdout"
    stdout.write_text("progress: 0.25\nprogress: 0.5\nprogress: 0.7")
    assert last_progress(stdout) == 0.5
