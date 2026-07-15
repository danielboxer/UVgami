import os
import pathlib
import platform
import shutil
import subprocess
from dataclasses import dataclass

from .ops.install import PARTUV_PLATFORMS, install_state
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
    description = ""
    # feature flags drive UI gating and post-processing compatibility;
    # engine-specific parameters live in the engine's property group and are
    # drawn by draw_settings instead of being flagged here
    supports_guided = False
    supports_viewer = False
    supports_early_stop = False
    supports_preserve = False
    supports_import_uvs = False
    # whether pack-after-unwrap starts enabled when this engine is selected
    pack_by_default = False

    def is_available(self):
        """Whether this engine can run on the current platform."""
        return True

    def validate(self, prefs):
        """Return (ctx, None) if usable, else (None, error_message). ctx is an
        engine-defined run context passed back to the build_* and stop methods."""
        raise NotImplementedError

    def draw_settings(self, layout, props):
        """Draw this engine's settings rows in the main panel."""
        pass

    def draw_prefs(self, layout, prefs):
        """Draw this engine's section in the addon preferences."""
        pass

    def allows_concurrent(self, props):
        """Whether multiple unwrap processes can run at once."""
        return True

    def wants_batch(self, props):
        """Whether queued meshes should share one engine process."""
        return False

    def build_args(self, ctx, input_path, props):
        """Return the subprocess argv that unwraps input_path."""
        raise NotImplementedError

    def build_batch_args(self, ctx, input_paths, props):
        """Return the argv unwrapping all input_paths in one process. Must be
        implemented when wants_batch can return True."""
        raise NotImplementedError

    def build_env(self, ctx):
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

    def stop(self, process, ctx):
        """Stop a running unwrap process."""
        process.kill()


class OptcutsEngine(Engine):
    id = "OPTCUTS"
    label = "Optcuts"
    description = "The default Optcuts unwrapping engine"
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
            if raw.stem != "optcuts":
                return None, "Engine path is incorrect"
            path = raw

        return path, None

    def draw_settings(self, layout, props):
        row = layout.row()
        row.label(icon="SOLO_OFF", text="Quality")
        row.prop(props.optcuts, "quality", text="")

    def draw_prefs(self, layout, prefs):
        row = layout.row()
        row.scale_y = 1.5
        split = row.split(factor=0.2)
        split.scale_x = 1.5
        split.label(text="Engine Path")
        split.prop(prefs, "engine_path")

        engine_path = pathlib.Path(prefs.engine_path)
        if str(engine_path) == "." and get_bundled_engine_path() is not None:
            row = layout.row()
            row.label(text="Using bundled optcuts engine", icon="CHECKMARK")

    def build_args(self, ctx, input_path, props):
        u = {"HIGH": "4.05", "MEDIUM": "4.1"}.get(props.optcuts.quality, "4.2")
        s = {5: "200", 4: "150", 3: "100", 2: "50", 1: "25"}.get(props.weight_value, "")
        shared_args = f"-u {u} -s {s}"

        return [str(ctx), "-i", str(input_path)] + shared_args.split()

    def describe_failure(self, code):
        return {
            -1: ("Mesh needs cleanup", True),
            101: ("Non Manifold Edges", True),
            102: ("Non Manifold Vertices", True),
            105: ("Invalid Geometry", True),
            107: ("Invalid UV Input", True),
            108: ("Unsupported Mesh Topology", True),
            109: ("Initial Cut Failed", True),
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


@dataclass
class PartuvRun:
    # dev runs the workspace partuv through uv, installed runs the wheel's
    # python -m partuv from the install operator's venv
    mode: str
    path: pathlib.Path


class PartuvEngine(Engine):
    id = "PARTUV"
    label = "PartUV"
    description = "Part-based unwrapping engine, requires an NVIDIA GPU"
    pack_by_default = True

    def is_available(self):
        return platform.system() in PARTUV_PLATFORMS

    def allows_concurrent(self, props):
        # ai loads torch and the partfield model per process, more than
        # one job oversubscribes vram and thrashes
        return props.partuv.segmentation != "AI"

    def wants_batch(self, props):
        # one process loads the model once for every queued mesh
        return props.partuv.segmentation == "AI"

    def validate(self, prefs):
        repo = find_partuv_dev_repo()
        if repo is not None:
            return PartuvRun("dev", repo), None
        if is_partuv_installed():
            return PartuvRun("installed", get_partuv_venv_path()), None
        return None, "PartUV is not installed. Install it in the add-on preferences"

    def draw_settings(self, layout, props):
        row = layout.row()
        row.label(icon="MOD_EXPLODE", text="Segmentation")
        row.prop(props.partuv, "segmentation", text="")

        row = layout.row()
        row.label(icon="MOD_LENGTH", text="Threshold")
        row.prop(props.partuv, "threshold")

    def draw_prefs(self, layout, prefs):
        if not self.is_available():
            layout.row().label(
                text="PartUV needs Windows or Linux with an NVIDIA GPU", icon="ERROR"
            )
            return
        if find_partuv_dev_repo() is not None:
            layout.row().label(
                text="PartUV: dev mode (running from repo)", icon="CHECKMARK"
            )
        elif install_state["running"]:
            row = layout.row()
            phase = install_state["phase"] or "Installing PartUV"
            total = install_state["bytes_total"]
            if total:
                factor = install_state["bytes_done"] / total
                row.progress(
                    factor=factor, type="BAR", text=f"{phase}  {factor * 100:.0f}%"
                )
            else:
                row.label(text=phase, icon="SORTTIME")
        else:
            geometric_installed = is_partuv_installed()
            ai_installed = is_partuv_ai_installed()
            if geometric_installed:
                layout.row().label(
                    text="PartUV installed (Geometric)", icon="CHECKMARK"
                )
                if ai_installed:
                    layout.row().label(text="PartUV installed (AI)", icon="CHECKMARK")
            else:
                layout.row().label(text="PartUV not installed")

            row = layout.row()
            row.scale_y = 1.5
            geometric = row.operator(
                "uvgami.install_partuv",
                text="Reinstall Geometric" if geometric_installed else "Geometric",
                icon="IMPORT",
            )
            geometric.tier = "GEOMETRIC"
            ai = row.operator(
                "uvgami.install_partuv",
                text="Reinstall AI" if ai_installed else "AI (~5 GB)",
                icon="IMPORT",
            )
            ai.tier = "AI"
            if install_state["error"] is not None:
                row = layout.row()
                row.label(text=install_state["error"], icon="ERROR")

    def build_args(self, ctx, input_path, props):
        return self.build_batch_args(ctx, [input_path], props)

    def build_batch_args(self, ctx, input_paths, props):
        if ctx.mode == "dev":
            base = [
                "uv",
                "run",
                "--project",
                str(ctx.path),
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
            props.partuv.segmentation.lower(),
            "--threshold",
            f"{props.partuv.threshold:.3f}",
            # drives the progress bar and the live chart viewer
            "--visual",
        ]

    def build_env(self, ctx):
        env = os.environ.copy()
        # the checkpoint isn't shipped in the wheel, and the cli's source-tree
        # default resolves relative to the installed package, so point at the
        # repo copy in dev mode and the downloaded one in installed mode
        if ctx.mode == "dev":
            checkpoint = ctx.path / "engine" / "partuv" / "model_objaverse.ckpt"
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

    def stop(self, process, ctx):
        if platform.system() == "Windows":
            # kill the whole tree, uv spawns the engine as a child
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
            )
        else:
            process.kill()


ENGINES = {engine.id: engine for engine in (OptcutsEngine(), PartuvEngine())}


def get_engine(engine_id):
    # default to optcuts so a stale or removed engine id in an old file still
    # loads (files saved before the rename stored "UVGAMI" and land here too)
    return ENGINES.get(engine_id, ENGINES["OPTCUTS"])
