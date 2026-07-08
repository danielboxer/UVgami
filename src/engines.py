# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import pathlib
import platform
import shutil
import subprocess

from .utils.io import print_stdin
from .utils.paths import (
    get_bundled_engine_path,
    get_dir_path,
    get_extension_dir_path,
    get_linux_path,
)


class Engine:
    id = ""
    label = ""
    # feature flags drive UI gating and post-processing compatibility
    supports_quality = False
    supports_guided = False
    supports_viewer = False
    supports_early_stop = False
    supports_preserve = False
    uses_threshold = False
    uses_segmentation = False

    def validate(self, prefs):
        """Return (engine_path, None) if usable, else (None, error_message)."""
        raise NotImplementedError

    def build_args(self, engine_path, input_path, props):
        """Return the subprocess argv that unwraps input_path."""
        raise NotImplementedError

    def stop(self, process, engine_path):
        """Stop a running unwrap process."""
        process.kill()


class UvgamiEngine(Engine):
    id = "UVGAMI"
    label = "UVgami"
    supports_quality = True
    supports_guided = True
    supports_viewer = True
    supports_early_stop = True
    supports_preserve = True

    def validate(self, prefs):
        raw = pathlib.Path(prefs.engine_path)
        if str(raw) == ".":
            # try bundled engine as fallback
            bundled = get_bundled_engine_path()
            if bundled is None:
                return (
                    None,
                    "Engine path is not set. Set the path in the add-on preferences",
                )
            path = bundled
        else:
            if not raw.is_file():
                return None, "Engine path doesn't exist"
            if raw.stem != "uvgami":
                return None, "Engine path is incorrect"
            path = raw

        wsl_error = self._setup_wsl(path, prefs)
        if wsl_error is not None:
            return None, wsl_error
        return path, None

    def _setup_wsl(self, path, prefs):
        if platform.system() != "Windows" or path.suffix != "" or prefs.is_wsl_setup:
            return None

        if shutil.which("wsl") is None:
            return "WSL is not installed. Either install WSL or use UVgami for Windows"

        r = subprocess.run(["bash", "-c", "test -e ~/uvgami"]).returncode
        if r == 1:
            # copy uvgami to wsl
            subprocess.run(["bash", "-c", f"cp {get_linux_path(path)} ~/"])
            prefs.is_wsl_setup = True
        elif r == 0:
            prefs.is_wsl_setup = True
        else:
            return "Unknown error configuring engine in WSL"
        return None

    def build_args(self, engine_path, input_path, props):
        u = {"HIGH": "4.05", "MEDIUM": "4.1"}.get(props.quality, "4.2")
        s = {5: "200", 4: "150", 3: "100", 2: "50", 1: "25"}.get(props.weight_value, "")
        shared_args = f"-u {u} -s {s}"

        if platform.system() == "Windows" and engine_path.suffix == "":
            input_arg = get_linux_path(input_path)
            output_arg = get_linux_path(get_extension_dir_path() / "output")
            return [
                "bash",
                "-c",
                f"~/uvgami -i {input_arg} -o {output_arg}/ {shared_args}",
            ]
        return [str(engine_path), "-i", str(input_path)] + shared_args.split()

    def stop(self, process, engine_path):
        if platform.system() == "Windows" and engine_path.suffix == "":
            # wsl
            print_stdin(process, "cancel")
        else:
            # windows
            process.kill()


class PartuvEngine(Engine):
    id = "PARTUV"
    label = "PartUV"
    uses_threshold = True
    uses_segmentation = True

    def validate(self, prefs):
        # runs through the developer cli, so it needs the repo checkout and uv
        repo = get_dir_path()
        if not (repo / "uvgami_cli").is_dir():
            return None, "PartUV needs the UVgami repo (run the add-on from source)"
        if shutil.which("uv") is None:
            return None, "PartUV needs uv installed (docs.astral.sh/uv)"
        if not (repo / ".venv").is_dir():
            return None, "PartUV needs the CLI set up (run 'uv sync' in the repo)"
        return repo, None

    def build_args(self, engine_path, input_path, props):
        output_path = get_extension_dir_path() / "output" / f"{input_path.stem}.obj"
        return [
            "uv",
            "run",
            "--project",
            str(engine_path),
            "--no-sync",
            "uvgami",
            "unwrap",
            str(input_path),
            "-o",
            str(output_path),
            "--overwrite",
            "--engine",
            "partuv",
            "--segmentation",
            props.partuv_segmentation.lower(),
            "--threshold",
            f"{props.partuv_threshold:.3f}",
        ]

    def stop(self, process, engine_path):
        if platform.system() == "Windows":
            # kill the whole tree, uv spawns the engine as a child
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
            )
        else:
            process.kill()


ENGINES = {engine.id: engine for engine in (UvgamiEngine(), PartuvEngine())}


def get_engine(engine_id):
    return ENGINES.get(engine_id, ENGINES["UVGAMI"])
