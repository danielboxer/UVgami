import json

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
    monkeypatch.setattr(partuv, "run", lambda *args: calls.append(args))
    code = cli.main(
        ["unwrap", str(triangle), "--engine", "partuv", "--segmentation", "geometric"]
    )
    assert code == 0
    assert calls[0][2] is None
    assert calls[0][5] == "geometric"


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


def test_engine_error_code_passthrough(triangle, monkeypatch):
    def fail(*args):
        raise UnwrapError(5, "no output")

    monkeypatch.setattr(optcuts, "run", fail)
    assert cli.main(["unwrap", str(triangle), "--engine", "optcuts"]) == 5
