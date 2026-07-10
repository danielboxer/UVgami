import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .common import (
    EXIT_ENGINE_FAILURE,
    EXIT_MISSING_RUNTIME,
    REPO_ROOT,
    UnwrapError,
    deliver,
    log,
)

QUALITY_UPPER_BOUND = {"high": "4.05", "medium": "4.1", "low": "4.2"}
SEAM_WEIGHT_LEVELS = {1: "25", 2: "50", 3: "100", 4: "150", 5: "200"}


def find_engine(explicit_path):
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.is_file():
            raise UnwrapError(EXIT_MISSING_RUNTIME, f"OptCuts binary not found: {path}")
        return path

    system = platform.system()
    if system == "Windows":
        subdir, name = "windows", "uvgami.exe"
    elif system == "Linux":
        subdir, name = "linux", "uvgami"
    elif system == "Darwin":
        machine = platform.machine().lower()
        subdir = "macos-arm64" if machine == "arm64" else "macos-x64"
        name = "uvgami"
    else:
        raise UnwrapError(EXIT_MISSING_RUNTIME, f"unsupported platform: {system}")

    path = REPO_ROOT / "engines" / subdir / name
    if not path.is_file():
        raise UnwrapError(
            EXIT_MISSING_RUNTIME,
            f"bundled OptCuts binary not found: {path} (pass --optcuts-path)",
        )
    return path


def build_args(engine_path, input_path, output_dir, quality, seam_weight, import_uvs):
    args = [
        str(engine_path),
        "-i",
        str(input_path),
        # optcuts appends the mesh name directly, so the separator is required
        "-o",
        str(output_dir) + os.sep,
        "-u",
        QUALITY_UPPER_BOUND[quality],
        "-s",
        SEAM_WEIGHT_LEVELS[seam_weight],
    ]
    if not import_uvs:
        args.append("-g")
    return args


def run(
    input_path, output_path, quality, import_uvs, seam_weights, seam_weight, engine_path
):
    engine = find_engine(engine_path)
    with tempfile.TemporaryDirectory(prefix="uvgami-") as tmp:
        in_dir = Path(tmp) / "in"
        out_dir = Path(tmp) / "out"
        in_dir.mkdir()
        out_dir.mkdir()

        work_input = in_dir / input_path.name
        shutil.copyfile(input_path, work_input)
        if seam_weights is not None:
            # optcuts reads the "<stem>_weights" sidecar next to the input
            shutil.copyfile(seam_weights, in_dir / f"{input_path.stem}_weights")

        args = build_args(engine, work_input, out_dir, quality, seam_weight, import_uvs)
        log(f"running: {' '.join(args)}", style="step")
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
        )
        for line in process.stdout:
            sys.stderr.write(line)
        returncode = process.wait()
        if returncode != 0:
            raise UnwrapError(
                EXIT_ENGINE_FAILURE, f"OptCuts exited with code {returncode}"
            )

        deliver(out_dir / f"{input_path.stem}.obj", output_path)
