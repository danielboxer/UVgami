import pytest

from seams import (
    Cancelled,
    EngineError,
    check_cancelled,
    face_edges,
)
from seams.preseed import _read_hardsurface
from seams.worker import WorkerProcess, last_progress

TWO_TRIANGLES = [(0, 1, 2), (1, 3, 2)]


def test_result_comes_back():
    worker = WorkerProcess(face_edges, TWO_TRIANGLES)
    assert worker.result() == face_edges(TWO_TRIANGLES)
    assert worker.done()


def test_error_is_raised_as_the_class_the_function_raised(tmp_path):
    (tmp_path / "seams.txt").write_text("")
    uvs = tmp_path / "uvs.txt"
    uvs.write_text("5 0 0 1 0 0 1\n")
    worker = WorkerProcess(_read_hardsurface, uvs, 2)
    with pytest.raises(EngineError, match="flattened face 5"):
        worker.result()


def test_the_function_sees_the_cancel():
    worker = WorkerProcess(check_cancelled)
    worker.cancel()
    with pytest.raises(Cancelled):
        worker.result()


def test_result_removes_the_worker_folder():
    worker = WorkerProcess(face_edges, TWO_TRIANGLES)
    folder = worker._folder
    assert folder.is_dir()
    worker.result()
    assert not folder.exists()


def test_last_progress_skips_a_half_written_line(tmp_path):
    stdout = tmp_path / "stdout"
    stdout.write_text("progress: 0.25\nprogress: 0.5\nprogress: 0.7")
    assert last_progress(stdout) == 0.5
