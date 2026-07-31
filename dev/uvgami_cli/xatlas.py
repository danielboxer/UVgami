import subprocess
import sys
import tempfile
from pathlib import Path

from .common import EXIT_ENGINE_FAILURE, UnwrapError, deliver, find_engine, log


def build_args(engine_path, input_path, output_path, max_cost=None):
    args = [str(engine_path), "-i", str(input_path), "-o", str(output_path)]
    # left off when unset so the engine's own default is the only one
    if max_cost is not None:
        args += ["--max-cost", f"{max_cost:.4f}"]
    return args


def run(input_path, output_path, engine_path, max_cost=None):
    engine = find_engine("xatlas", "xatlas", engine_path)
    # write to a temp file first so a rejected result never lands on output_path
    with tempfile.TemporaryDirectory(prefix="uvgami-") as tmp:
        result = Path(tmp) / f"{input_path.stem}.obj"
        args = build_args(engine, input_path, result, max_cost)
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
                EXIT_ENGINE_FAILURE, f"xatlas exited with code {returncode}"
            )

        deliver(result, output_path)
