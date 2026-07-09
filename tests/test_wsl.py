import subprocess

import pytest

from uvgami_cli import partuv, wsl
from uvgami_cli.common import UnwrapError


def utf16(names):
    return ("\r\n".join(names) + "\r\n").encode("utf-16-le")


class FakeWsl:
    """Answers the three wsl.exe call shapes the bridge makes."""

    def __init__(self, unwrap_returncode=0):
        self.calls = []
        self.unwrap_returncode = unwrap_returncode

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if cmd[1] == "--list":
            return subprocess.CompletedProcess(
                cmd, 0, stdout=utf16(["docker-desktop", "Ubuntu-24.04"]), stderr=b""
            )
        if cmd[5] == "-c":  # wslpath translation batch
            paths = cmd[8:]
            out = "".join(
                "/mnt/c/" + p.replace("\\", "/").replace(":", "") + "\n" for p in paths
            )
            return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
        return subprocess.CompletedProcess(cmd, self.unwrap_returncode)


@pytest.fixture
def fake_wsl(monkeypatch):
    monkeypatch.delenv("UVGAMI_WSL_DISTRO", raising=False)
    monkeypatch.delenv("UVGAMI_WSL_VENV", raising=False)
    monkeypatch.setattr(partuv.platform, "system", lambda: "Windows")
    # force the bridge even when a native partuv is installed in the venv
    monkeypatch.setattr(partuv, "_native_available", lambda: False)
    fake = FakeWsl()
    monkeypatch.setattr(wsl.subprocess, "run", fake)
    return fake


def test_windows_bridges_to_wsl(triangle, tmp_path, fake_wsl):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("ckpt")

    partuv.run([(triangle, tmp_path / "out.obj")], str(checkpoint), None, 1.5)

    unwrap_call = fake_wsl.calls[-1]
    assert unwrap_call[:6] == ["wsl.exe", "-d", "Ubuntu-24.04", "-e", "bash", "-lc"]
    command = unwrap_call[6]
    assert "uv run --no-sync uvgami unwrap" in command
    assert "--engine partuv" in command
    assert "--threshold 1.5" in command
    assert 'UV_PROJECT_ENVIRONMENT="$HOME"/uvgami-venv' in command
    assert "/mnt/c/" in command


def test_env_overrides(triangle, tmp_path, fake_wsl, monkeypatch):
    monkeypatch.setenv("UVGAMI_WSL_DISTRO", "Ubuntu-22.04")
    monkeypatch.setenv("UVGAMI_WSL_VENV", "/root/uvgami-venv")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("ckpt")

    partuv.run([(triangle, tmp_path / "out.obj")], str(checkpoint), None, 1.25)

    unwrap_call = fake_wsl.calls[-1]
    assert unwrap_call[1:3] == ["-d", "Ubuntu-22.04"]
    assert "UV_PROJECT_ENVIRONMENT=/root/uvgami-venv" in unwrap_call[6]


def test_geometric_bridge_omits_checkpoint(triangle, tmp_path, fake_wsl):
    partuv.run([(triangle, tmp_path / "out.obj")], None, None, 1.25, "geometric")
    command = fake_wsl.calls[-1][6]
    assert "--segmentation geometric" in command
    assert "--checkpoint" not in command
    assert "--visual" not in command


def test_visual_forwarded_to_wsl(triangle, tmp_path, fake_wsl):
    partuv.run([(triangle, tmp_path / "out.obj")], None, None, 1.25, "geometric", True)
    assert "--visual" in fake_wsl.calls[-1][6]


def test_wsl_side_checkpoint_passes_through(triangle, tmp_path, fake_wsl):
    partuv.run([(triangle, tmp_path / "out.obj")], "/root/model.ckpt", None, 1.25)
    assert "--checkpoint /root/model.ckpt" in fake_wsl.calls[-1][6]


def test_missing_checkpoint_errors(triangle, tmp_path, fake_wsl):
    with pytest.raises(UnwrapError) as error:
        partuv.run([(triangle, tmp_path / "out.obj")], "C:\\nope\\model.ckpt", None, 1.25)
    assert error.value.exit_code == 3


def test_engine_exit_code_passthrough(triangle, tmp_path, fake_wsl):
    fake_wsl.unwrap_returncode = 4
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("ckpt")
    with pytest.raises(UnwrapError) as error:
        partuv.run([(triangle, tmp_path / "out.obj")], str(checkpoint), None, 1.25)
    assert error.value.exit_code == 4


def test_unknown_exit_code_maps_to_engine_failure(triangle, tmp_path, fake_wsl):
    fake_wsl.unwrap_returncode = 1
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("ckpt")
    with pytest.raises(UnwrapError) as error:
        partuv.run([(triangle, tmp_path / "out.obj")], str(checkpoint), None, 1.25)
    assert error.value.exit_code == 4


def test_batch_bridges_every_pair(triangle, cube, tmp_path, fake_wsl):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("ckpt")

    pairs = [(triangle, tmp_path / "a.obj"), (cube, tmp_path / "b.obj")]
    partuv.run(pairs, str(checkpoint), None, 1.25)

    command = fake_wsl.calls[-1][6]
    assert "triangle.obj" in command
    assert "cube.obj" in command
    assert command.count(" -o ") == 2


def test_wsl_missing_errors(triangle, tmp_path, monkeypatch):
    monkeypatch.delenv("UVGAMI_WSL_DISTRO", raising=False)
    monkeypatch.setattr(partuv.platform, "system", lambda: "Windows")
    monkeypatch.setattr(partuv, "_native_available", lambda: False)

    def no_wsl(cmd, **kwargs):
        raise FileNotFoundError("wsl.exe")

    monkeypatch.setattr(wsl.subprocess, "run", no_wsl)
    with pytest.raises(UnwrapError) as error:
        partuv.run([(triangle, tmp_path / "out.obj")], "model.ckpt", None, 1.25)
    assert error.value.exit_code == 3
