import subprocess
import sys
import tempfile
from pathlib import Path

from .common import (
    EXIT_BAD_OUTPUT,
    EXIT_ENGINE_FAILURE,
    EXIT_INVALID_INPUT,
    EXIT_MISSING_RUNTIME,
    UnwrapError,
    deliver,
    find_engine,
    log,
)

PARTUV_EXIT_CODES = {
    EXIT_INVALID_INPUT,
    EXIT_MISSING_RUNTIME,
    EXIT_ENGINE_FAILURE,
    EXIT_BAD_OUTPUT,
}


def build_command(args, partuv_path, tools_path, input_path, output_path):
    shared = [
        str(input_path),
        "-o",
        str(output_path),
        "--threshold",
        str(args.threshold),
    ]
    if args.segmentation == "ai":
        return [
            sys.executable,
            "-m",
            "partuv_partfield",
            *shared,
            "--checkpoint",
            str(args.checkpoint),
            "--partuv-path",
            str(partuv_path),
        ]
    command = [str(partuv_path), *shared, "--segmentation", args.segmentation]
    if args.config is not None:
        command += ["--config", str(args.config)]
    if tools_path is not None:
        command += ["--uvgami-tools-path", str(tools_path)]
    return command


def run(input_path, output_path, args):
    partuv_path = find_engine("partuv", "PartUV", args.partuv_path)
    tools_path = None
    if args.segmentation == "tree":
        tools_path = find_engine("uvgami-tools", "uvgami-tools", None)
    # a rejected result must never land on output_path
    with tempfile.TemporaryDirectory(prefix="uvgami-") as tmp:
        result = Path(tmp) / f"{input_path.stem}.obj"
        command = build_command(args, partuv_path, tools_path, input_path, result)
        log(f"running: {' '.join(command)}", style="step")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
        )
        for line in process.stdout:
            sys.stderr.write(line)
        returncode = process.wait()
        if returncode != 0:
            code = (
                returncode if returncode in PARTUV_EXIT_CODES else EXIT_ENGINE_FAILURE
            )
            raise UnwrapError(code, f"partuv exited with code {returncode}")

        deliver(result, output_path)
