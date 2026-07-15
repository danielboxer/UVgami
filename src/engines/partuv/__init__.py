import os
import pathlib
import platform
import shutil
import subprocess
from dataclasses import dataclass

import bpy

from .. import Engine
from ...utils.paths import get_dir_path, get_extension_dir_path
from .install import PARTUV_PLATFORMS, UVGAMI_OT_install_partuv, install_state
from .paths import (
    get_partuv_checkpoint_path,
    get_partuv_venv_path,
    get_partuv_venv_python,
)


class UVGAMI_PG_partuv(bpy.types.PropertyGroup):
    segmentation: bpy.props.EnumProperty(
        name="Segmentation",
        description="How PartUV splits the mesh into parts",
        items=(
            (
                "GEOMETRIC",
                "Geometric",
                "Geometric clustering. No AI model needed",
            ),
            (
                "AI",
                "AI",
                "AI segmentation with the PartField model. Requires the AI install",
            ),
        ),
        default="AI",
    )
    threshold: bpy.props.FloatProperty(
        name="",
        description="Distortion threshold. Lower values cut the mesh into more UV islands",
        default=1.25,
        min=1.0,
        max=10.0,
    )


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
    property_group = UVGAMI_PG_partuv
    classes = (UVGAMI_PG_partuv, UVGAMI_OT_install_partuv)
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


ENGINE = PartuvEngine()
