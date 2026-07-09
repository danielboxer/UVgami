# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import os
import pathlib
import platform
import shutil
import subprocess
import sys

from .utils.io import print_stdin
from .utils.paths import (
    get_bundled_engine_path,
    get_dir_path,
    get_extension_dir_path,
    get_linux_path,
    get_partuv_libs_path,
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
    # pushes finished charts on stdout instead of answering stdin snapshots
    viewer_push = False

    def validate(self, prefs):
        """Return (engine_path, None) if usable, else (None, error_message)."""
        raise NotImplementedError

    def allows_concurrent(self, props):
        """Whether multiple unwrap processes can run at once."""
        return True

    def wants_batch(self, props):
        """Whether queued meshes should share one engine process."""
        return False

    def build_args(self, engine_path, input_path, props):
        """Return the subprocess argv that unwraps input_path."""
        raise NotImplementedError

    def build_env(self, engine_path):
        """Return the subprocess env, or None to inherit."""
        return None

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


def find_partuv_dev_repo():
    """Return the repo path if the developer CLI is usable, else None."""
    repo = get_dir_path()
    if (
        (repo / "uvgami_cli").is_dir()
        and (repo / ".venv").is_dir()
        and shutil.which("uv") is not None
    ):
        return repo
    return None


def is_partuv_installed():
    return (get_partuv_libs_path() / "partuv").is_dir()


class PartuvEngine(Engine):
    id = "PARTUV"
    label = "PartUV"
    supports_viewer = True
    uses_threshold = True
    uses_segmentation = True
    viewer_push = True
    # set by validate: dev runs the repo cli through uv, installed runs the
    # wheel from the install operator with blender's python
    mode = "dev"

    def allows_concurrent(self, props):
        # ai loads torch and the partfield model per process, more than
        # one job oversubscribes vram and thrashes
        return props.partuv_segmentation != "AI"

    def wants_batch(self, props):
        # one process loads the model once for every queued mesh
        return props.partuv_segmentation == "AI"

    def validate(self, prefs):
        repo = find_partuv_dev_repo()
        if repo is not None:
            self.mode = "dev"
            return repo, None
        if is_partuv_installed():
            self.mode = "installed"
            return get_partuv_libs_path(), None
        return None, "PartUV is not installed. Install it in the add-on preferences"

    def build_args(self, engine_path, input_path, props):
        return self.build_batch_args(engine_path, [input_path], props)

    def build_batch_args(self, engine_path, input_paths, props):
        if self.mode == "dev":
            base = ["uv", "run", "--project", str(engine_path), "--no-sync", "uvgami"]
        else:
            base = [sys.executable, "-m", "uvgami_cli"]
        return base + [
            "unwrap",
            *[str(path) for path in input_paths],
            "--output-dir",
            str(get_extension_dir_path() / "output"),
            "--overwrite",
            "--engine",
            "partuv",
            "--segmentation",
            props.partuv_segmentation.lower(),
            "--threshold",
            f"{props.partuv_threshold:.3f}",
            # drives the progress bar and the live chart viewer
            "--visual",
        ]

    def build_env(self, engine_path):
        if self.mode == "dev":
            return None
        # uvgami_cli is resolved from the add-on dir, partuv from the libs dir
        env = os.environ.copy()
        paths = [str(engine_path), str(get_dir_path())]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        return env

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
