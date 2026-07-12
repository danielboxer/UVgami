# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import os
import pathlib
import platform
import shutil
import subprocess

from .utils.io import print_stdin
from .utils.paths import (
    get_bundled_engine_path,
    get_dir_path,
    get_extension_dir_path,
    get_partuv_checkpoint_path,
    get_partuv_venv_path,
    get_partuv_venv_python,
)


class Engine:
    id = ""
    label = ""
    # feature flags drive UI gating and post-processing compatibility
    # supports_* are optional capabilities the ui gates on, uses_* are
    # engine-specific params it configures
    supports_quality = False
    supports_guided = False
    supports_viewer = False
    supports_early_stop = False
    supports_preserve = False
    supports_import_uvs = False
    uses_threshold = False
    uses_segmentation = False
    # whether pack-after-unwrap starts enabled when this engine is selected
    pack_by_default = False

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

    def build_batch_args(self, engine_path, input_paths, props):
        """Return the argv unwrapping all input_paths in one process. Must be
        implemented when wants_batch can return True."""
        raise NotImplementedError

    def build_env(self, engine_path):
        """Return the subprocess env, or None to inherit."""
        return None

    def describe_failure(self, code):
        """Map an engine exit code to (message, move_to_invalid), or None if the
        engine does not recognize it (caller shows a generic unknown-error)."""
        # windows access violation (0xC0000005): the engine process crashed
        if code == -1073741819:
            return ("Engine crashed", True)
        return None

    def request_early_stop(self, process):
        """Ask a running process to stop and finish with its current result.
        Returns True if delivered; engines that cannot stop gracefully return False."""
        return False

    def request_snapshot(self, process):
        """Ask a running process to emit a uv snapshot for the live viewer. No-op
        for engines without a viewer."""
        pass

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
    supports_import_uvs = True

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

        return path, None

    def build_args(self, engine_path, input_path, props):
        u = {"HIGH": "4.05", "MEDIUM": "4.1"}.get(props.quality, "4.2")
        s = {5: "200", 4: "150", 3: "100", 2: "50", 1: "25"}.get(props.weight_value, "")
        shared_args = f"-u {u} -s {s}"

        return [str(engine_path), "-i", str(input_path)] + shared_args.split()

    def describe_failure(self, code):
        return {
            -1: ("Mesh needs cleanup", True),
            101: ("Non Manifold Edges", True),
            102: ("Non Manifold Vertices", True),
            105: ("Invalid Geometry", True),
            107: ("Invalid UV Input", True),
        }.get(code) or super().describe_failure(code)

    def request_early_stop(self, process):
        return print_stdin(process, "stop")

    def request_snapshot(self, process):
        print_stdin(process, "snapshot")


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
    return get_partuv_venv_python().is_file()


def is_partuv_ai_installed():
    return is_partuv_installed() and get_partuv_checkpoint_path().is_file()


class PartuvEngine(Engine):
    id = "PARTUV"
    label = "PartUV"
    uses_threshold = True
    uses_segmentation = True
    pack_by_default = True
    # set by validate: dev runs the workspace partuv through uv, installed runs
    # the wheel's python -m partuv from the install operator's venv
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
            return get_partuv_venv_path(), None
        return None, "PartUV is not installed. Install it in the add-on preferences"

    def build_args(self, engine_path, input_path, props):
        return self.build_batch_args(engine_path, [input_path], props)

    def build_batch_args(self, engine_path, input_paths, props):
        if self.mode == "dev":
            base = [
                "uv",
                "run",
                "--project",
                str(engine_path),
                "--no-sync",
                "python",
                "-m",
                "partuv",
            ]
        else:
            base = [str(get_partuv_venv_python()), "-m", "partuv"]
        # windows caps a command line near 32k chars, so a large batch of mesh
        # paths as argv overflows CreateProcess. pass them in a file instead.
        # named per invocation since solo mode spawns several over one session;
        # lives in the input dir so manager.finish cleans it up with the meshes.
        input_list = (
            get_extension_dir_path() / "input" / f"{input_paths[0].stem}_inputs.txt"
        )
        input_list.write_text(
            "\n".join(str(path) for path in input_paths) + "\n", encoding="utf-8"
        )
        return base + [
            "--input-list",
            str(input_list),
            "--output-dir",
            str(get_extension_dir_path() / "output"),
            "--overwrite",
            "--segmentation",
            props.partuv_segmentation.lower(),
            "--threshold",
            f"{props.partuv_threshold:.3f}",
            # drives the progress bar and the live chart viewer
            "--visual",
        ]

    def build_env(self, engine_path):
        env = os.environ.copy()
        # the checkpoint isn't shipped in the wheel, and the cli's source-tree
        # default resolves relative to the installed package, so point at the
        # repo copy in dev mode and the downloaded one in installed mode
        if self.mode == "dev":
            checkpoint = engine_path / "engine" / "partuv" / "model_objaverse.ckpt"
        else:
            checkpoint = get_partuv_checkpoint_path()
        env["UVGAMI_PARTUV_CHECKPOINT"] = str(checkpoint)
        return env

    def describe_failure(self, code):
        # partuv cli exit codes
        return {
            2: ("Invalid input mesh", True),
            3: ("PartUV runtime error, reinstall in preferences", False),
            4: ("PartUV failed on this mesh", True),
            5: ("PartUV produced invalid output", True),
        }.get(code) or super().describe_failure(code)

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
    # default to uvgami so a stale or removed engine id in an old file still loads
    return ENGINES.get(engine_id, ENGINES["UVGAMI"])
