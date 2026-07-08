"""Runs the partuv engine inside WSL when the CLI is invoked on Windows."""

import os
import shlex
import subprocess
from pathlib import Path

from .common import (
    EXIT_ENGINE_FAILURE,
    EXIT_MISSING_RUNTIME,
    REPO_ROOT,
    UnwrapError,
    log,
)

EXIT_CODES = {2, 3, 4, 5}
NON_LINUX_DISTROS = {"docker-desktop", "docker-desktop-data"}


def pick_distro():
    named = os.environ.get("UVGAMI_WSL_DISTRO")
    if named:
        return named
    try:
        result = subprocess.run(
            ["wsl.exe", "--list", "--quiet"], capture_output=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise UnwrapError(
            EXIT_MISSING_RUNTIME,
            "WSL is not available (PartUV needs Linux with CUDA)",
        ) from error
    # wsl.exe prints UTF-16LE
    names = result.stdout.decode("utf-16-le", errors="ignore").split()
    distros = [name for name in names if name not in NON_LINUX_DISTROS]
    if not distros:
        raise UnwrapError(
            EXIT_MISSING_RUNTIME,
            "no WSL distro found, install Ubuntu or set UVGAMI_WSL_DISTRO",
        )
    return distros[0]


def to_wsl_paths(distro, paths):
    script = 'for p in "$@"; do wslpath -a "$p"; done'
    result = subprocess.run(
        ["wsl.exe", "-d", distro, "-e", "bash", "-c", script, "_"]
        + [str(path) for path in paths],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise UnwrapError(
            EXIT_MISSING_RUNTIME, f"path translation failed: {result.stderr.strip()}"
        )
    translated = result.stdout.split("\n")
    return [line.strip() for line in translated[: len(paths)]]


def wsl_checkpoint(distro, checkpoint):
    """Windows paths are translated; paths that only exist inside WSL pass through."""
    if Path(checkpoint).is_file():
        return to_wsl_paths(distro, [checkpoint])[0]
    if str(checkpoint).replace("\\", "/").startswith("/"):
        return str(checkpoint).replace("\\", "/")
    raise UnwrapError(EXIT_MISSING_RUNTIME, f"checkpoint not found: {checkpoint}")


def run(input_path, output_path, checkpoint, config, threshold, segmentation="ai"):
    distro = pick_distro()
    log(f"running PartUV in WSL distro {distro}")

    to_translate = [REPO_ROOT, input_path.resolve(), output_path.resolve()]
    if config is not None:
        to_translate.append(Path(config).resolve())
    translated = to_wsl_paths(distro, to_translate)
    repo, input_wsl, output_wsl = translated[:3]

    # --overwrite because the Windows side already validated the output path
    unwrap = [
        "uv",
        "run",
        "--no-sync",
        "uvgami",
        "unwrap",
        input_wsl,
        "-o",
        output_wsl,
        "--overwrite",
        "--engine",
        "partuv",
        "--segmentation",
        segmentation,
        "--threshold",
        str(threshold),
    ]
    if segmentation == "ai":
        unwrap += ["--checkpoint", wsl_checkpoint(distro, checkpoint)]
    if config is not None:
        unwrap += ["--config", translated[3]]

    # default venv lives on ext4: torch imports from /mnt/c are far too slow
    venv = os.environ.get("UVGAMI_WSL_VENV")
    venv_arg = shlex.quote(venv) if venv else '"$HOME"/uvgami-venv'
    command = (
        f"cd {shlex.quote(repo)} && "
        f"UV_PROJECT_ENVIRONMENT={venv_arg} {shlex.join(unwrap)}"
    )
    # login shell so ~/.local/bin (uv) is on PATH
    result = subprocess.run(["wsl.exe", "-d", distro, "-e", "bash", "-lc", command])
    if result.returncode != 0:
        code = result.returncode if result.returncode in EXIT_CODES else EXIT_ENGINE_FAILURE
        raise UnwrapError(code, "PartUV failed in WSL (see log above)")
