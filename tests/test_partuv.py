import sys
import types

import pytest

from uvgami_cli import partuv
from uvgami_cli.common import UnwrapError


def test_requires_linux(triangle, tmp_path, monkeypatch):
    monkeypatch.setattr(partuv.platform, "system", lambda: "Windows")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("ckpt")
    with pytest.raises(UnwrapError) as error:
        partuv.run(triangle, tmp_path / "out.obj", checkpoint, None, 1.25)
    assert error.value.exit_code == 3


def test_missing_checkpoint(triangle, tmp_path, monkeypatch):
    monkeypatch.setattr(partuv.platform, "system", lambda: "Linux")
    with pytest.raises(UnwrapError) as error:
        partuv.run(triangle, tmp_path / "out.obj", tmp_path / "nope.ckpt", None, 1.25)
    assert error.value.exit_code == 3


def test_missing_config(triangle, tmp_path, monkeypatch):
    monkeypatch.setattr(partuv.platform, "system", lambda: "Linux")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("ckpt")
    with pytest.raises(UnwrapError) as error:
        partuv.run(
            triangle, tmp_path / "out.obj", checkpoint, tmp_path / "nope.yaml", 1.25
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

    class FakeModel:
        def __init__(self, checkpoint_path, device):
            calls["model"] = (checkpoint_path, device)

    def fake_preprocess(mesh_path, pf_model=None, output_path=None):
        calls["preprocess"] = mesh_path
        return FakeMesh(), None, {"tree": 1}, {}

    def fake_pipeline_numpy(V, F, tree_dict, config_path, threshold):
        calls["pipeline"] = (tree_dict, config_path, threshold)
        return "final", ["part0"]

    def fake_save_results(output_dir, final_part, individual_parts):
        calls["save"] = (final_part, individual_parts)
        result = output_dir / "final_components.obj"
        result.write_text("v 0 0 0\nvt 0 0\nf 1/1 1/1 1/1\n")

    fake.preprocess = fake_preprocess
    fake.pipeline_numpy = fake_pipeline_numpy
    preprocess_module.PFInferenceModel = FakeModel
    preprocess_module.save_results = fake_save_results
    monkeypatch.setitem(sys.modules, "partuv", fake)
    monkeypatch.setitem(sys.modules, "partuv.preprocess", preprocess_module)
    return calls


def test_run_orchestration(triangle, tmp_path, fake_partuv_runtime):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("ckpt")
    config = tmp_path / "config.yaml"
    config.write_text("pipeline: {}")
    output = tmp_path / "out.obj"

    partuv.run(triangle, output, checkpoint, config, 1.5)

    calls = fake_partuv_runtime
    assert calls["model"] == (str(checkpoint), "cuda")
    assert calls["preprocess"] == str(triangle)
    assert calls["pipeline"] == ({"tree": 1}, str(config), 1.5)
    assert calls["save"] == ("final", ["part0"])
    assert output.is_file()
    assert "vt 0 0" in output.read_text()
