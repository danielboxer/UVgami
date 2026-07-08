import sys
import types

import pytest

from uvgami_cli import partuv
from uvgami_cli.common import UnwrapError


# windows now bridges to wsl (see test_wsl.py), other systems still error
def test_requires_linux(triangle, tmp_path, monkeypatch):
    monkeypatch.setattr(partuv.platform, "system", lambda: "Darwin")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("ckpt")
    with pytest.raises(UnwrapError) as error:
        partuv.run([(triangle, tmp_path / "out.obj")], checkpoint, None, 1.25)
    assert error.value.exit_code == 3


def test_resolve_checkpoint_precedence(tmp_path, monkeypatch):
    flagged = tmp_path / "flag.ckpt"
    monkeypatch.setenv("UVGAMI_PARTUV_CHECKPOINT", "/root/env.ckpt")
    assert partuv.resolve_checkpoint(flagged) == str(flagged)
    assert partuv.resolve_checkpoint(None) == "/root/env.ckpt"

    monkeypatch.delenv("UVGAMI_PARTUV_CHECKPOINT")
    default = tmp_path / "model_objaverse.ckpt"
    monkeypatch.setattr(partuv, "DEFAULT_CHECKPOINT", default)
    with pytest.raises(UnwrapError) as error:
        partuv.resolve_checkpoint(None)
    assert error.value.exit_code == 2
    default.write_text("ckpt")
    assert partuv.resolve_checkpoint(None) == str(default)


def test_missing_checkpoint(triangle, tmp_path, monkeypatch):
    monkeypatch.setattr(partuv.platform, "system", lambda: "Linux")
    with pytest.raises(UnwrapError) as error:
        partuv.run([(triangle, tmp_path / "out.obj")], tmp_path / "nope.ckpt", None, 1.25)
    assert error.value.exit_code == 3


def test_missing_config(triangle, tmp_path, monkeypatch):
    monkeypatch.setattr(partuv.platform, "system", lambda: "Linux")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("ckpt")
    with pytest.raises(UnwrapError) as error:
        partuv.run([(triangle, tmp_path / "out.obj")], checkpoint, tmp_path / "nope.yaml", 1.25
        )
    assert error.value.exit_code == 3


class FakeMesh:
    vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    faces = [[0, 1, 2]]


@pytest.fixture
def fake_partuv_runtime(monkeypatch, tmp_path):
    """Install fake torch and partuv modules so run() orchestration is exercised."""
    monkeypatch.setattr(partuv.platform, "system", lambda: "Linux")

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", torch)

    calls = {}

    fake = types.ModuleType("partuv")
    preprocess_module = types.ModuleType("partuv.preprocess")
    output_module = types.ModuleType("partuv.output")
    geometric_module = types.ModuleType("partuv.geometric")

    class FakeModel:
        def __init__(self, checkpoint_path, device):
            calls["model"] = (checkpoint_path, device)
            calls["model_loads"] = calls.get("model_loads", 0) + 1

    def fake_preprocess(mesh_path, pf_model=None, output_path=None):
        calls["preprocess"] = mesh_path
        return FakeMesh(), None, {"tree": 1}, {}

    def fake_preprocess_geometric(mesh_path):
        calls["geometric"] = mesh_path
        return FakeMesh(), {"tree": 2}

    def fake_pipeline_numpy(V, F, tree_dict, config_path, threshold):
        calls["pipeline"] = (tree_dict, config_path, threshold)
        return "final", ["part0"]

    def fake_save_results(output_dir, final_part, individual_parts):
        calls["save"] = (final_part, individual_parts)
        result = output_dir / "final_components.obj"
        result.write_text("v 0 0 0\nvt 0 0\nf 1/1 1/1 1/1\n")

    fake.pipeline_numpy = fake_pipeline_numpy
    preprocess_module.preprocess = fake_preprocess
    preprocess_module.PFInferenceModel = FakeModel
    output_module.save_results = fake_save_results
    geometric_module.preprocess_geometric = fake_preprocess_geometric
    monkeypatch.setitem(sys.modules, "partuv", fake)
    monkeypatch.setitem(sys.modules, "partuv.preprocess", preprocess_module)
    monkeypatch.setitem(sys.modules, "partuv.output", output_module)
    monkeypatch.setitem(sys.modules, "partuv.geometric", geometric_module)
    return calls


def test_run_orchestration(triangle, tmp_path, fake_partuv_runtime):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("ckpt")
    config = tmp_path / "config.yaml"
    config.write_text("pipeline: {}")
    output = tmp_path / "out.obj"

    partuv.run([(triangle, output)], checkpoint, config, 1.5)

    calls = fake_partuv_runtime
    assert calls["model"] == (str(checkpoint), "cuda")
    assert calls["preprocess"] == str(triangle)
    assert calls["pipeline"] == ({"tree": 1}, str(config), 1.5)
    assert calls["save"] == ("final", ["part0"])
    assert output.is_file()
    assert "vt 0 0" in output.read_text()


def test_run_geometric_skips_torch_and_checkpoint(triangle, tmp_path, fake_partuv_runtime):
    config = tmp_path / "config.yaml"
    config.write_text("pipeline: {}")
    output = tmp_path / "out.obj"

    partuv.run([(triangle, output)], None, config, 1.25, "geometric")

    calls = fake_partuv_runtime
    assert calls["geometric"] == str(triangle)
    assert "model" not in calls
    assert calls["pipeline"] == ({"tree": 2}, str(config), 1.25)
    assert output.is_file()


def test_batch_loads_model_once(triangle, cube, tmp_path, fake_partuv_runtime, capsys):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("ckpt")
    config = tmp_path / "config.yaml"
    config.write_text("pipeline: {}")
    pairs = [(triangle, tmp_path / "a.obj"), (cube, tmp_path / "b.obj")]

    code = partuv.run(pairs, checkpoint, config, 1.5)

    assert code == 0
    assert fake_partuv_runtime["model_loads"] == 1
    assert (tmp_path / "a.obj").is_file()
    assert (tmp_path / "b.obj").is_file()
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["start: triangle", "done: triangle", "start: cube", "done: cube"]


def test_windows_runs_native_when_partuv_installed(
    triangle, tmp_path, fake_partuv_runtime, monkeypatch
):
    monkeypatch.setattr(partuv.platform, "system", lambda: "Windows")
    monkeypatch.delenv("UVGAMI_PARTUV_WSL", raising=False)
    config = tmp_path / "config.yaml"
    config.write_text("pipeline: {}")
    output = tmp_path / "out.obj"

    partuv.run([(triangle, output)], None, config, 1.25, "geometric")

    assert fake_partuv_runtime["geometric"] == str(triangle)
    assert output.is_file()


def test_windows_env_var_forces_wsl(triangle, tmp_path, fake_partuv_runtime, monkeypatch):
    monkeypatch.setattr(partuv.platform, "system", lambda: "Windows")
    monkeypatch.setenv("UVGAMI_PARTUV_WSL", "1")
    calls = {}
    monkeypatch.setattr(
        "uvgami_cli.wsl.run", lambda *args: calls.setdefault("wsl", args)
    )

    partuv.run([(triangle, tmp_path / "out.obj")], None, None, 1.25, "geometric")

    assert "wsl" in calls
    assert fake_partuv_runtime == {}
