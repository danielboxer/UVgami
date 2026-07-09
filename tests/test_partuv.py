import sys
import types
from pathlib import Path

import pytest

from partuv import cli
from partuv.common import UnwrapError


# windows now bridges to wsl (see test_wsl.py), other systems still error
def test_requires_linux(triangle, tmp_path, monkeypatch):
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("ckpt")
    with pytest.raises(UnwrapError) as error:
        cli.run([(triangle, tmp_path / "out.obj")], checkpoint, None, 1.25)
    assert error.value.exit_code == 3


def test_resolve_checkpoint_precedence(tmp_path, monkeypatch):
    flagged = tmp_path / "flag.ckpt"
    monkeypatch.setenv("UVGAMI_PARTUV_CHECKPOINT", "/root/env.ckpt")
    assert cli.resolve_checkpoint(flagged) == str(flagged)
    assert cli.resolve_checkpoint(None) == "/root/env.ckpt"

    monkeypatch.delenv("UVGAMI_PARTUV_CHECKPOINT")
    default = tmp_path / "model_objaverse.ckpt"
    monkeypatch.setattr(cli, "DEFAULT_CHECKPOINT", default)
    with pytest.raises(UnwrapError) as error:
        cli.resolve_checkpoint(None)
    assert error.value.exit_code == 2
    default.write_text("ckpt")
    assert cli.resolve_checkpoint(None) == str(default)


def test_missing_checkpoint(triangle, tmp_path, monkeypatch, fake_partuv_runtime):
    config = tmp_path / "config.yaml"
    config.write_text("pipeline: {}")
    with pytest.raises(UnwrapError) as error:
        cli.run([(triangle, tmp_path / "out.obj")], tmp_path / "nope.ckpt", config, 1.25)
    assert error.value.exit_code == 3


def test_missing_config(triangle, tmp_path, monkeypatch):
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("ckpt")
    with pytest.raises(UnwrapError) as error:
        cli.run([(triangle, tmp_path / "out.obj")], checkpoint, tmp_path / "nope.yaml", 1.25
        )
    assert error.value.exit_code == 3


class FakeMesh:
    vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    faces = [[0, 1, 2]]


@pytest.fixture
def fake_partuv_runtime(monkeypatch, tmp_path):
    """Install fake torch and partuv submodules so run() orchestration is exercised."""
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    # default to a machine with a gpu so geometric tests keep the config as-is;
    # cpu-fallback tests override this
    monkeypatch.setattr(cli, "_cuda_available", lambda: True)

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", torch)

    calls = {}

    core_module = types.ModuleType("partuv._core")
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

    def fake_pipeline_numpy(V, F, tree_dict, config_path, threshold, visual=False):
        calls["pipeline"] = (tree_dict, config_path, threshold)
        # read now: a cpu-fallback temp config is deleted when run() returns
        calls["config_text"] = Path(config_path).read_text()
        calls["visual"] = visual
        return "final", ["part0"]

    def fake_save_results(output_dir, final_part, individual_parts):
        calls["save"] = (final_part, individual_parts)
        result = output_dir / "final_components.obj"
        result.write_text("v 0 0 0\nvt 0 0\nf 1/1 1/1 1/1\n")

    core_module.pipeline_numpy = fake_pipeline_numpy
    preprocess_module.preprocess = fake_preprocess
    preprocess_module.PFInferenceModel = FakeModel
    output_module.save_results = fake_save_results
    geometric_module.preprocess_geometric = fake_preprocess_geometric
    monkeypatch.setitem(sys.modules, "partuv._core", core_module)
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

    cli.run([(triangle, output)], checkpoint, config, 1.5)

    calls = fake_partuv_runtime
    assert calls["model"] == (str(checkpoint), "cuda")
    assert calls["preprocess"] == str(triangle)
    assert calls["pipeline"] == ({"tree": 1}, str(config), 1.5)
    assert calls["visual"] is False
    assert calls["save"] == ("final", ["part0"])
    assert output.is_file()
    assert "vt 0 0" in output.read_text()


def test_run_geometric_skips_torch_and_checkpoint(triangle, tmp_path, fake_partuv_runtime):
    config = tmp_path / "config.yaml"
    config.write_text("pipeline: {}")
    output = tmp_path / "out.obj"

    cli.run([(triangle, output)], None, config, 1.25, "geometric")

    calls = fake_partuv_runtime
    assert calls["geometric"] == str(triangle)
    assert "model" not in calls
    assert calls["pipeline"] == ({"tree": 2}, str(config), 1.25)
    assert output.is_file()


def test_geometric_cpu_fallback_disables_pamo(
    triangle, tmp_path, fake_partuv_runtime, monkeypatch
):
    monkeypatch.setattr(cli, "_cuda_available", lambda: False)
    config = tmp_path / "config.yaml"
    config.write_text("unwrap:\n  pamo: true\n  usePamoFaceThreshold: 1000\n")

    cli.run([(triangle, tmp_path / "out.obj")], None, config, 1.25, "geometric")

    calls = fake_partuv_runtime
    # pipeline sees a rewritten temp config, not the original
    assert calls["pipeline"][1] != str(config)
    assert "pamo: false" in calls["config_text"]
    assert "pamo: true" not in calls["config_text"]


def test_geometric_gpu_keeps_config(
    triangle, tmp_path, fake_partuv_runtime, monkeypatch
):
    monkeypatch.setattr(cli, "_cuda_available", lambda: True)
    config = tmp_path / "config.yaml"
    config.write_text("unwrap:\n  pamo: true\n")

    cli.run([(triangle, tmp_path / "out.obj")], None, config, 1.25, "geometric")

    calls = fake_partuv_runtime
    assert calls["pipeline"][1] == str(config)
    assert "pamo: true" in calls["config_text"]


def test_ai_never_probes_cuda(triangle, tmp_path, fake_partuv_runtime, monkeypatch):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("ckpt")
    config = tmp_path / "config.yaml"
    config.write_text("unwrap:\n  pamo: true\n")
    probed = []
    monkeypatch.setattr(cli, "_cuda_available", lambda: probed.append(True) or True)

    cli.run([(triangle, tmp_path / "out.obj")], checkpoint, config, 1.5)

    calls = fake_partuv_runtime
    # torch.cuda already gates the ai path, so the driver probe stays untouched
    assert probed == []
    assert calls["pipeline"][1] == str(config)
    assert "pamo: true" in calls["config_text"]


def test_visual_reaches_pipeline(triangle, tmp_path, fake_partuv_runtime):
    config = tmp_path / "config.yaml"
    config.write_text("pipeline: {}")

    cli.run([(triangle, tmp_path / "out.obj")], None, config, 1.25, "geometric", True)

    assert fake_partuv_runtime["visual"] is True


def test_batch_loads_model_once(triangle, cube, tmp_path, fake_partuv_runtime, capsys):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("ckpt")
    config = tmp_path / "config.yaml"
    config.write_text("pipeline: {}")
    pairs = [(triangle, tmp_path / "a.obj"), (cube, tmp_path / "b.obj")]

    code = cli.run(pairs, checkpoint, config, 1.5)

    assert code == 0
    assert fake_partuv_runtime["model_loads"] == 1
    assert (tmp_path / "a.obj").is_file()
    assert (tmp_path / "b.obj").is_file()
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["start: triangle", "done: triangle", "start: cube", "done: cube"]


def test_windows_runs_native_when_partuv_installed(
    triangle, tmp_path, fake_partuv_runtime, monkeypatch
):
    monkeypatch.setattr(cli.platform, "system", lambda: "Windows")
    monkeypatch.delenv("UVGAMI_PARTUV_WSL", raising=False)
    # a loadable core is the native-vs-wsl routing knob
    monkeypatch.setattr("partuv._CORE_ERROR", None)
    config = tmp_path / "config.yaml"
    config.write_text("pipeline: {}")
    output = tmp_path / "out.obj"

    cli.run([(triangle, output)], None, config, 1.25, "geometric")

    assert fake_partuv_runtime["geometric"] == str(triangle)
    assert output.is_file()


def test_windows_env_var_forces_wsl(triangle, tmp_path, fake_partuv_runtime, monkeypatch):
    monkeypatch.setattr(cli.platform, "system", lambda: "Windows")
    monkeypatch.setenv("UVGAMI_PARTUV_WSL", "1")
    calls = {}
    monkeypatch.setattr(
        "partuv.wsl.run", lambda *args: calls.setdefault("wsl", args)
    )
    # the dev override must not gate on is_usable
    def boom():
        raise AssertionError("is_usable consulted for the dev override")

    monkeypatch.setattr("partuv.wsl.is_usable", boom)

    cli.run([(triangle, tmp_path / "out.obj")], None, None, 1.25, "geometric")

    assert "wsl" in calls
    assert fake_partuv_runtime == {}


def test_windows_wsl_fallback_when_usable(triangle, tmp_path, monkeypatch):
    monkeypatch.setattr(cli.platform, "system", lambda: "Windows")
    monkeypatch.delenv("UVGAMI_PARTUV_WSL", raising=False)
    # a non-None core error makes _native_available() report the native miss
    monkeypatch.setattr("partuv._CORE_ERROR", ImportError("no _core.pyd"))
    monkeypatch.setattr("partuv.wsl.is_usable", lambda: True)
    calls = {}
    monkeypatch.setattr("partuv.wsl.run", lambda *args: calls.setdefault("wsl", args))

    cli.run([(triangle, tmp_path / "out.obj")], None, None, 1.25, "geometric")

    assert "wsl" in calls


def test_windows_reinstall_hint_when_wsl_unusable(triangle, tmp_path, monkeypatch):
    monkeypatch.setattr(cli.platform, "system", lambda: "Windows")
    monkeypatch.delenv("UVGAMI_PARTUV_WSL", raising=False)
    monkeypatch.setattr("partuv._CORE_ERROR", ImportError("msvcp140 missing"))
    monkeypatch.setattr("partuv.wsl.is_usable", lambda: False)

    def fail(*args):
        raise AssertionError("wsl.run should not be called")

    monkeypatch.setattr("partuv.wsl.run", fail)

    with pytest.raises(UnwrapError) as error:
        cli.run([(triangle, tmp_path / "out.obj")], None, None, 1.25, "geometric")

    assert error.value.exit_code == 3
    message = str(error.value)
    assert "reinstall PartUV" in message
    assert "msvcp140 missing" in message
