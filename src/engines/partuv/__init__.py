import functools
import os
import pathlib
import platform
import shutil
import signal
import subprocess
from dataclasses import dataclass

import bpy

from ...no_bpy.engine_run import EngineError
from ...utils.paths import get_dir_path, get_io_dir_paths
from ...utils.ui import only_active
from .. import Engine, EngineFailure
from ..install_task import (
    ENGINE_TOO_OLD_ERROR,
    NOT_DOWNLOADED_ERROR,
    UPDATE_ICON,
    CachedUpdateNotice,
    draw_error,
    draw_install_status,
    draw_online_access,
    draw_progress,
    task_state,
)
from .install import (
    PARTUV_PLATFORMS,
    PARTUV_VERSION,
    UVGAMI_OT_install_partuv,
    UVGAMI_OT_uninstall_partuv,
    get_installed_partuv_version,
    partuv_too_old,
    partuv_update_pending,
)
from .paths import (
    CHECKPOINT_NAME,
    get_partfield_checkpoint_path,
    get_partuv_venv_python,
)

GEOMETRIC_SEGMENTATION = 0
AI_SEGMENTATION = 1

# blender requires a dynamic enum's item strings to stay referenced
_SEGMENTATION_ITEMS = (
    (
        "GEOMETRIC",
        "Geometric",
        "Geometric clustering. Faster but more seams",
        "",
        GEOMETRIC_SEGMENTATION,
    ),
    (
        "AI",
        "AI",
        "AI segmentation. Better results than geometric",
        "",
        AI_SEGMENTATION,
    ),
)


def _segmentation_items(self, context):
    if is_ai_segmentation_available():
        return list(_SEGMENTATION_ITEMS)
    return [_SEGMENTATION_ITEMS[GEOMETRIC_SEGMENTATION]]


def _segmentation_get(self):
    stored = self.get("segmentation", AI_SEGMENTATION)
    if stored == AI_SEGMENTATION and not is_ai_segmentation_available():
        return GEOMETRIC_SEGMENTATION
    return stored


def _segmentation_set(self, value):
    self["segmentation"] = value


# 1.0 is a chart per triangle
PRIORITY_VALUES = {"LESS_STRETCH": "1.1", "BALANCED": "1.25", "FEWER_SEAMS": "1.5"}


class UVGAMI_PG_partuv(bpy.types.PropertyGroup):
    segmentation: bpy.props.EnumProperty(
        name="Mode",
        description="How PartUV splits the mesh into parts",
        items=_segmentation_items,
        get=_segmentation_get,
        set=_segmentation_set,
    )


@functools.cache
def find_partuv_dev_repo():
    repo = get_dir_path()
    if (
        (repo / "dev" / "uvgami_cli").is_dir()
        and (repo / ".venv").is_dir()
        and shutil.which("uv") is not None
    ):
        return repo
    return None


def is_partuv_installed():
    # uv venv creates the python before anything is installed into it
    return get_installed_partuv_version() is not None


def is_partuv_ai_installed():
    return is_partuv_installed() and get_partfield_checkpoint_path().is_file()


def is_ai_segmentation_available():
    return find_partuv_dev_repo() is not None or is_partuv_ai_installed()


@dataclass
class PartuvRun:
    command: list
    checkpoint: pathlib.Path


class PartuvEngine(CachedUpdateNotice, Engine):
    id = "PARTUV"
    enum_value = 1
    label = "PartUV"
    description = "GPU engine. Fewer islands and can be faster on dense meshes"
    icon = "MOD_EXPLODE"
    supports_symmetry = True
    property_group = UVGAMI_PG_partuv
    has_install_controls = True
    install_owner = "partuv"
    install_phase = "Installing PartUV"
    classes = (UVGAMI_PG_partuv, UVGAMI_OT_install_partuv, UVGAMI_OT_uninstall_partuv)

    def is_available(self):
        return platform.system() in PARTUV_PLATFORMS

    def batches_queue(self, props):
        # more than one process runs out of vram
        return props.partuv.segmentation == "AI"

    def validate(self):
        repo = find_partuv_dev_repo()
        if repo is not None:
            return PartuvRun(
                [
                    "uv",
                    "run",
                    "--project",
                    str(repo),
                    "--no-sync",
                    "python",
                    "-m",
                    "partuv",
                ],
                repo / "engine-builds" / CHECKPOINT_NAME,
            )
        if partuv_too_old():
            raise EngineError(ENGINE_TOO_OLD_ERROR)
        if is_partuv_installed():
            return PartuvRun(
                [str(get_partuv_venv_python()), "-m", "partuv"],
                get_partfield_checkpoint_path(),
            )
        raise EngineError(NOT_DOWNLOADED_ERROR)

    def invalidate_caches(self):
        super().invalidate_caches()
        get_installed_partuv_version.cache_clear()

    def draw_settings(self, layout, props):
        row = layout.row()
        row.label(icon="MOD_EXPLODE", text="Mode")
        row.prop(props.partuv, "segmentation", text="")

    def active_settings(self, props):
        partuv = props.partuv
        return only_active(
            (
                (
                    "MOD_EXPLODE",
                    "Geometric Segmentation",
                    "partuv.segmentation",
                    partuv.segmentation == "GEOMETRIC"
                    and is_ai_segmentation_available(),
                ),
            )
        )

    def describe(self):
        if find_partuv_dev_repo() is not None:
            return f"{self.label} (local build)"
        version = get_installed_partuv_version()
        return f"{self.label} v{version}" if version else self.label

    def update_pending(self):
        return find_partuv_dev_repo() is None and partuv_update_pending()

    def install_too_old(self):
        return partuv_too_old()

    def draw_prefs(self, layout):
        if not self.is_available():
            layout.row().label(text="PartUV is not available on macOS", icon="ERROR")
            return
        if find_partuv_dev_repo() is not None:
            layout.row().label(text="Local build", icon="CHECKMARK")
            return

        if task_state["running"] and task_state["owner"] == self.install_owner:
            draw_progress(layout, self.install_phase)
            return

        installed = is_partuv_installed()
        ai_installed = is_partuv_ai_installed()
        update_pending = partuv_update_pending()
        draw_install_status(
            layout,
            get_installed_partuv_version(),
            PARTUV_VERSION,
            too_old=partuv_too_old(),
            update_pending=update_pending,
        )

        current_tier = "AI" if ai_installed else "GEOMETRIC"
        wants_download = update_pending or not ai_installed
        if not (wants_download and draw_online_access(layout)):
            columns = layout.split(factor=0.5)
            for header, tier, tier_installed in (
                ("Geometric Segmentation", "GEOMETRIC", installed),
                ("AI Segmentation", "AI", ai_installed),
            ):
                column = columns.column()
                title = column.row()
                title.alignment = "CENTER"
                title.label(text=header)
                button = column.row()
                button.scale_y = 1.5
                if update_pending and tier == current_tier:
                    button.operator(
                        "uvgami.install_partuv", text="Update Engine", icon=UPDATE_ICON
                    ).tier = tier
                elif tier_installed:
                    button.alignment = "CENTER"
                    button.label(text="Downloaded", icon="CHECKMARK")
                else:
                    button.operator(
                        "uvgami.install_partuv", text="Download Engine", icon="IMPORT"
                    ).tier = tier
        if installed:
            row = layout.row()
            row.scale_y = 1.5
            row.operator("uvgami.uninstall_partuv", text="Delete Engine", icon="TRASH")

        draw_error(layout, self.install_owner)

    def build_args(self, ctx, input_path, props):
        return self.build_batch_args(ctx, [input_path], props)

    def build_batch_args(self, ctx, input_paths, props):
        input_dir, output_dir = get_io_dir_paths()
        # windows caps a command line near 32k characters
        input_list = input_dir / f"{input_paths[0].stem}_inputs.txt"
        input_list.write_text(
            "\n".join(str(path) for path in input_paths) + "\n", encoding="utf-8"
        )
        return ctx.command + [
            "--input-list",
            str(input_list),
            "--output-dir",
            str(output_dir),
            "--overwrite",
            "--segmentation",
            props.partuv.segmentation.lower(),
            "--threshold",
            PRIORITY_VALUES[props.priority],
            "--visual",
        ]

    def build_env(self, ctx):
        env = os.environ.copy()
        # partuv's default checkpoint path is relative to its installed package
        env["UVGAMI_PARTUV_CHECKPOINT"] = str(ctx.checkpoint)
        return env

    def describe_failure(self, code):
        return {
            2: EngineFailure("Invalid input mesh", moves_to_invalid=True),
            # 3 is a missing module, config, checkpoint or cuda, with the cause in stderr
            3: EngineFailure("PartUV could not start", moves_to_invalid=False),
            4: EngineFailure("PartUV failed on this mesh", moves_to_invalid=True),
            5: EngineFailure("PartUV produced invalid output", moves_to_invalid=True),
        }.get(code) or super().describe_failure(code)

    def stop(self, process, ctx):
        # uv starts the engine as a child process, which outlives a kill of uv alone
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
            )
            return
        children = pathlib.Path(f"/proc/{process.pid}/task/{process.pid}/children")
        for child in children.read_text().split():
            try:
                os.kill(int(child), signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.kill()


ENGINE = PartuvEngine()
