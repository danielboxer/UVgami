import subprocess
from pathlib import Path

import pytest

from uvgami_cli import optcuts
from uvgami_cli.common import UnwrapError

BUNDLED = optcuts.REPO_ROOT / "engines" / "windows" / "uvgami.exe"


class FakeProcess:
    """Stands in for the OptCuts subprocess and writes the expected output OBJ."""

    def __init__(
        self, argv, returncode=0, output_text="v 0 0 0\nvt 0 0\nf 1/1 1/1 1/1\n"
    ):
        self.argv = argv
        self.returncode = returncode
        self.output_text = output_text
        self.stdout = iter(["progress: 0 0 1\n"])

    def wait(self):
        input_path = Path(self.argv[self.argv.index("-i") + 1])
        # captured here because the workdir is deleted after run() returns
        self.sidecar_existed = (
            input_path.parent / f"{input_path.stem}_weights"
        ).is_file()
        if self.returncode == 0:
            out_dir = Path(self.argv[self.argv.index("-o") + 1])
            (out_dir / f"{input_path.stem}.obj").write_text(self.output_text)
        return self.returncode


@pytest.fixture
def fake_engine(tmp_path):
    path = tmp_path / "uvgami.exe"
    path.write_text("fake")
    return path


def popen_recorder(monkeypatch, **process_kwargs):
    calls = []

    def fake_popen(argv, **kwargs):
        process = FakeProcess(argv, **process_kwargs)
        calls.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return calls


def test_build_args_quality_and_weight_mapping(fake_engine):
    args = optcuts.build_args(
        fake_engine, Path("in.obj"), Path("out"), "high", 5, False
    )
    assert args[args.index("-u") + 1] == "4.05"
    assert args[args.index("-s") + 1] == "200"
    assert "-g" in args

    args = optcuts.build_args(fake_engine, Path("in.obj"), Path("out"), "low", 1, True)
    assert args[args.index("-u") + 1] == "4.2"
    assert args[args.index("-s") + 1] == "25"
    assert "-g" not in args


def test_output_dir_arg_has_trailing_separator(fake_engine):
    args = optcuts.build_args(
        fake_engine, Path("in.obj"), Path("out"), "medium", 3, False
    )
    out_arg = args[args.index("-o") + 1]
    assert out_arg.endswith(("/", "\\"))


def test_run_success(triangle, tmp_path, fake_engine, monkeypatch):
    popen_recorder(monkeypatch)
    output = tmp_path / "result.obj"
    optcuts.run(triangle, output, "medium", False, None, 3, fake_engine)
    assert output.is_file()
    assert "vt 0 0" in output.read_text()


def test_run_copies_weights_sidecar(triangle, tmp_path, fake_engine, monkeypatch):
    calls = popen_recorder(monkeypatch)
    weights = tmp_path / "weights.txt"
    weights.write_text("0,1.0")
    optcuts.run(
        triangle, tmp_path / "out.obj", "medium", False, weights, 3, fake_engine
    )
    argv = calls[0].argv
    assert Path(argv[argv.index("-i") + 1]).name == "triangle.obj"
    assert calls[0].sidecar_existed


def test_run_engine_failure(triangle, tmp_path, fake_engine, monkeypatch):
    popen_recorder(monkeypatch, returncode=7)
    with pytest.raises(UnwrapError) as error:
        optcuts.run(
            triangle, tmp_path / "out.obj", "medium", False, None, 3, fake_engine
        )
    assert error.value.exit_code == 4


def test_run_output_missing_uvs(triangle, tmp_path, fake_engine, monkeypatch):
    popen_recorder(monkeypatch, output_text="v 0 0 0\nf 1 1 1\n")
    with pytest.raises(UnwrapError) as error:
        optcuts.run(
            triangle, tmp_path / "out.obj", "medium", False, None, 3, fake_engine
        )
    assert error.value.exit_code == 5


def test_find_engine_explicit_path_missing(tmp_path):
    with pytest.raises(UnwrapError) as error:
        optcuts.find_engine(tmp_path / "nope.exe")
    assert error.value.exit_code == 3


@pytest.mark.smoke
@pytest.mark.skipif(not BUNDLED.is_file(), reason="bundled OptCuts binary not present")
def test_optcuts_smoke(cube, tmp_path):
    output = tmp_path / "cube_uv.obj"
    optcuts.run(cube, output, "low", False, None, 3, None)
    text = output.read_text()
    assert "vt " in text
    assert "f " in text
