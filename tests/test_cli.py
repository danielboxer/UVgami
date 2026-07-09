import json
import subprocess
import sys

import pytest

from uvgami_cli import cli, optcuts
from uvgami_cli.common import UnwrapError


@pytest.fixture
def fake_optcuts(monkeypatch):
    calls = []

    def fake_run(*args):
        calls.append(args)
        output_path = args[1]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("v 0 0 0\nvt 0 0\nf 1/1 1/1 1/1\n")

    monkeypatch.setattr(optcuts, "run", fake_run)
    return calls


def test_default_output_name(triangle, fake_optcuts):
    assert cli.main(["unwrap", str(triangle), "--engine", "optcuts"]) == 0
    assert (triangle.parent / "triangle_uv.obj").is_file()


def test_explicit_output(triangle, tmp_path, fake_optcuts):
    out = tmp_path / "result" / "custom.obj"
    assert cli.main(["unwrap", str(triangle), "--engine", "optcuts", "-o", str(out)]) == 0
    assert fake_optcuts[0][1] == out


def test_optcuts_defaults(triangle, fake_optcuts):
    cli.main(["unwrap", str(triangle), "--engine", "optcuts"])
    _, _, quality, import_uvs, seam_weights, seam_weight, engine_path = fake_optcuts[0]
    assert quality == "medium"
    assert import_uvs is False
    assert seam_weights is None
    assert seam_weight == 3
    assert engine_path is None


def test_missing_input(tmp_path, fake_optcuts):
    code = cli.main(["unwrap", str(tmp_path / "nope.obj"), "--engine", "optcuts"])
    assert code == 2


def test_non_obj_input(tmp_path, fake_optcuts):
    bad = tmp_path / "mesh.stl"
    bad.write_text("solid")
    assert cli.main(["unwrap", str(bad), "--engine", "optcuts"]) == 2


def test_overwrite_protection(triangle, fake_optcuts):
    existing = triangle.parent / "triangle_uv.obj"
    existing.write_text("v 0 0 0\n")
    assert cli.main(["unwrap", str(triangle), "--engine", "optcuts"]) == 2
    assert cli.main(["unwrap", str(triangle), "--engine", "optcuts", "--overwrite"]) == 0


def test_partuv_flag_rejected_for_optcuts(triangle, fake_optcuts):
    code = cli.main(
        ["unwrap", str(triangle), "--engine", "optcuts", "--threshold", "1.5"]
    )
    assert code == 2
    assert not fake_optcuts


def test_optcuts_flag_rejected_for_partuv(triangle):
    code = cli.main(
        ["unwrap", str(triangle), "--engine", "partuv", "--quality", "high"]
    )
    assert code == 2


def test_partuv_requires_checkpoint(triangle, tmp_path, monkeypatch):
    from uvgami_cli import partuv

    monkeypatch.delenv("UVGAMI_PARTUV_CHECKPOINT", raising=False)
    monkeypatch.setattr(partuv, "DEFAULT_CHECKPOINT", tmp_path / "missing.ckpt")
    assert cli.main(["unwrap", str(triangle), "--engine", "partuv"]) == 2


def test_partuv_geometric_needs_no_checkpoint(triangle, tmp_path, monkeypatch):
    from uvgami_cli import partuv

    monkeypatch.delenv("UVGAMI_PARTUV_CHECKPOINT", raising=False)
    monkeypatch.setattr(partuv, "DEFAULT_CHECKPOINT", tmp_path / "missing.ckpt")
    calls = []
    monkeypatch.setattr(partuv, "run", lambda *args: calls.append(args) or 0)
    code = cli.main(
        ["unwrap", str(triangle), "--engine", "partuv", "--segmentation", "geometric"]
    )
    assert code == 0
    pairs, checkpoint, _, _, segmentation, visual = calls[0]
    assert pairs == [(triangle, triangle.parent / "triangle_uv.obj")]
    assert checkpoint is None
    assert segmentation == "geometric"
    assert visual is False


def test_visual_flag_forwarded(triangle, monkeypatch):
    from uvgami_cli import partuv

    calls = []
    monkeypatch.setattr(partuv, "run", lambda *args: calls.append(args) or 0)
    code = cli.main(
        [
            "unwrap",
            str(triangle),
            "--engine",
            "partuv",
            "--segmentation",
            "geometric",
            "--visual",
        ]
    )
    assert code == 0
    assert calls[0][-1] is True


def test_visual_rejected_for_optcuts(triangle, fake_optcuts):
    code = cli.main(["unwrap", str(triangle), "--engine", "optcuts", "--visual"])
    assert code == 2
    assert not fake_optcuts


def test_checkpoint_rejected_for_geometric(triangle, tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("ckpt")
    code = cli.main(
        [
            "unwrap",
            str(triangle),
            "--engine",
            "partuv",
            "--segmentation",
            "geometric",
            "--checkpoint",
            str(checkpoint),
        ]
    )
    assert code == 2


def test_missing_seam_weights_file(triangle, tmp_path, fake_optcuts):
    code = cli.main(
        [
            "unwrap",
            str(triangle),
            "--engine",
            "optcuts",
            "--seam-weights",
            str(tmp_path / "nope_weights"),
        ]
    )
    assert code == 2


def test_json_success(triangle, fake_optcuts, capsys):
    assert cli.main(["unwrap", str(triangle), "--engine", "optcuts", "--json"]) == 0
    out = capsys.readouterr().out
    result = json.loads(out)
    assert result["status"] == "ok"
    assert result["engine"] == "optcuts"
    assert result["output"].endswith("triangle_uv.obj")


def test_json_error(triangle, monkeypatch, capsys):
    def fail(*args):
        raise UnwrapError(4, "engine blew up")

    monkeypatch.setattr(optcuts, "run", fail)
    assert cli.main(["unwrap", str(triangle), "--engine", "optcuts", "--json"]) == 4
    result = json.loads(capsys.readouterr().out)
    assert result == {"status": "error", "exit_code": 4, "message": "engine blew up"}


def test_stdout_empty_without_json(triangle, fake_optcuts, capsys):
    cli.main(["unwrap", str(triangle), "--engine", "optcuts"])
    assert capsys.readouterr().out == ""


# the addon's installed mode runs the cli as python -m uvgami_cli
def test_module_entry_point(triangle):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "uvgami_cli",
            "unwrap",
            str(triangle),
            "--engine",
            "partuv",
            "--segmentation",
            "geometric",
            "--threshold",
            "not-a-number",
        ],
        capture_output=True,
        text=True,
    )
    # argparse rejects the threshold before any engine import runs
    assert result.returncode == 2
    assert "invalid float value" in result.stderr


def test_engine_error_code_passthrough(triangle, monkeypatch):
    def fail(*args):
        raise UnwrapError(5, "no output")

    monkeypatch.setattr(optcuts, "run", fail)
    assert cli.main(["unwrap", str(triangle), "--engine", "optcuts"]) == 5


def test_batch_markers_and_output_dir(triangle, cube, tmp_path, fake_optcuts, capsys):
    out_dir = tmp_path / "out"
    code = cli.main(
        ["unwrap", str(triangle), str(cube), "--engine", "optcuts", "--output-dir", str(out_dir)]
    )
    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["start: triangle", "done: triangle", "start: cube", "done: cube"]
    assert fake_optcuts[0][1] == out_dir / "triangle.obj"
    assert fake_optcuts[1][1] == out_dir / "cube.obj"


def test_batch_isolates_failures(triangle, cube, monkeypatch, capsys):
    def run(input_path, output_path, *rest):
        if input_path.stem == "triangle":
            raise UnwrapError(4, "boom")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("v 0 0 0\nvt 0 0\nf 1/1 1/1 1/1\n")

    monkeypatch.setattr(optcuts, "run", run)
    code = cli.main(["unwrap", str(triangle), str(cube), "--engine", "optcuts"])
    assert code == 4
    lines = capsys.readouterr().out.splitlines()
    assert "failed: triangle 4" in lines
    assert "done: cube" in lines
    assert (cube.parent / "cube_uv.obj").is_file()


def test_batch_isolates_unexpected_exceptions(triangle, cube, monkeypatch, capsys):
    def run(input_path, output_path, *rest):
        if input_path.stem == "cube":
            raise RuntimeError("segfault-ish")
        output_path.write_text("v 0 0 0\nvt 0 0\nf 1/1 1/1 1/1\n")

    monkeypatch.setattr(optcuts, "run", run)
    code = cli.main(["unwrap", str(triangle), str(cube), "--engine", "optcuts"])
    assert code == 4
    lines = capsys.readouterr().out.splitlines()
    assert "done: triangle" in lines
    assert "failed: cube 4" in lines


def test_single_input_emits_no_markers(triangle, tmp_path, fake_optcuts, capsys):
    out_dir = tmp_path / "out"
    code = cli.main(
        ["unwrap", str(triangle), "--engine", "optcuts", "--output-dir", str(out_dir)]
    )
    assert code == 0
    assert capsys.readouterr().out == ""
    assert fake_optcuts[0][1] == out_dir / "triangle.obj"


def test_output_count_mismatch(triangle, cube, tmp_path, fake_optcuts):
    code = cli.main(
        ["unwrap", str(triangle), str(cube), "--engine", "optcuts", "-o", str(tmp_path / "x.obj")]
    )
    assert code == 2
    assert not fake_optcuts


def test_output_and_output_dir_conflict(triangle, tmp_path, fake_optcuts):
    code = cli.main(
        [
            "unwrap",
            str(triangle),
            "--engine",
            "optcuts",
            "-o",
            str(tmp_path / "x.obj"),
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert code == 2


def test_json_rejected_for_multiple_inputs(triangle, cube, fake_optcuts):
    code = cli.main(["unwrap", str(triangle), str(cube), "--engine", "optcuts", "--json"])
    assert code == 2


def test_batch_missing_input_fails_per_mesh(triangle, tmp_path, fake_optcuts, capsys):
    missing = tmp_path / "gone.obj"
    code = cli.main(["unwrap", str(triangle), str(missing), "--engine", "optcuts"])
    assert code == 2
    lines = capsys.readouterr().out.splitlines()
    assert "done: triangle" in lines
    assert "failed: gone 2" in lines


def test_input_deleted_mid_batch_is_skipped(triangle, cube, tmp_path, capsys):
    """Cancelling a mesh in the add-on deletes its input file while the batch
    runs; the mesh must fail fast without aborting the rest."""
    from uvgami_cli.common import unwrap_all

    unwrapped = []

    def unwrap_one(input_path, output_path):
        # cancel the cube while the triangle is being unwrapped
        cube.unlink(missing_ok=True)
        unwrapped.append(input_path.stem)

    code = unwrap_all(
        [(triangle, tmp_path / "a.obj"), (cube, tmp_path / "b.obj")], unwrap_one
    )
    assert code == 2
    assert unwrapped == ["triangle"]
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "start: triangle",
        "done: triangle",
        "start: cube",
        "failed: cube 2",
    ]


def test_colliding_outputs_rejected(triangle, tmp_path, fake_optcuts):
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other = other_dir / "triangle.obj"
    other.write_bytes(triangle.read_bytes())
    code = cli.main(
        [
            "unwrap",
            str(triangle),
            str(other),
            "--engine",
            "optcuts",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 2
