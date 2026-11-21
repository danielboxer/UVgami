import importlib.util
import os
import pathlib
import platform
import signal
import subprocess
from dataclasses import dataclass

import bpy

from ...no_bpy.engine_run import EngineError
from ...utils.engine_binaries import get_dev_build_path, get_platform_tag
from ...utils.paths import get_io_dir_paths
from ...utils.ui import only_active
from ...utils.uvgami_tools import tools_command
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
    PARTUV,
    PARTUV_ARCHIVE_SHA256S,
    PARTUV_VERSION,
    UVGAMI_OT_install_partuv,
    UVGAMI_OT_uninstall_partuv,
)

# the superhive build ships without the partfield package
if importlib.util.find_spec(f"{__name__}.partfield") is not None:
    from . import partfield
else:
    partfield = None

GEOMETRIC_SEGMENTATION = 0
AI_SEGMENTATION = 1
TREE_SEGMENTATION = 2

# blender requires a dynamic enum's item strings to stay referenced
_GEOMETRIC_ITEM = (
    "GEOMETRIC",
    "Geometric",
    "Geometric clustering. Faster but more seams",
    "",
    GEOMETRIC_SEGMENTATION,
)
_TREE_ITEM = (
    "TREE",
    "Tree",
    "Splits the mesh into parts by shape. No AI model needed",
    "",
    TREE_SEGMENTATION,
)
_AI_ITEM = (
    "AI",
    "AI",
    "AI segmentation. Better results than geometric",
    "",
    AI_SEGMENTATION,
)


def _segmentation_items(self, context):
    if is_ai_segmentation_available():
        return [_GEOMETRIC_ITEM, _TREE_ITEM, _AI_ITEM]
    return [_GEOMETRIC_ITEM, _TREE_ITEM]


def _segmentation_get(self):
    stored = self.get("segmentation", AI_SEGMENTATION)
    if stored == AI_SEGMENTATION and not is_ai_segmentation_available():
        return TREE_SEGMENTATION
    return stored


def _segmentation_set(self, value):
    self["segmentation"] = value


# 1.0 is a chart per triangle
PRIORITY_VALUES = {"LESS_STRETCH": "1.1", "BALANCED": "1.25", "FEWER_SEAMS": "1.5"}


class UVGAMI_PG_partuv(bpy.types.PropertyGroup):
    segmentation: bpy.props.EnumProperty(
        name="Mode",
        description="How the mesh is split into parts",
        items=_segmentation_items,
        get=_segmentation_get,
        set=_segmentation_set,
    )


def is_ai_segmentation_available():
    return partfield is not None and partfield.is_ai_segmentation_available()


@dataclass
class PartuvRun:
    partuv_path: pathlib.Path
    tools_path: str


class PartuvEngine(CachedUpdateNotice, Engine):
    id = "PARTUV"
    enum_value = 1
    label = "PartUV"
    description = "Fewer islands and can be faster on dense meshes"
    icon = "MOD_EXPLODE"
    supports_regions = True
    supports_symmetry = True
    supports_proxy = True
    property_group = UVGAMI_PG_partuv
    has_install_controls = True
    install_owner = "partuv"
    install_phase = "Installing PartUV"
    classes = (
        UVGAMI_PG_partuv,
        UVGAMI_OT_install_partuv,
        UVGAMI_OT_uninstall_partuv,
        *(() if partfield is None else (partfield.UVGAMI_OT_install_partfield,)),
    )

    def is_available(self):
        return get_platform_tag() in PARTUV_ARCHIVE_SHA256S

    def batches_queue(self, props):
        # more than one process runs out of vram
        return props.partuv.segmentation == "AI"

    def validate(self):
        partuv_path = get_dev_build_path(PARTUV.name) or PARTUV.installed_path()
        if partuv_path is None:
            if PARTUV.install_too_old():
                raise EngineError(ENGINE_TOO_OLD_ERROR)
            raise EngineError(NOT_DOWNLOADED_ERROR)
        return PartuvRun(partuv_path, tools_command().args[0])

    def invalidate_caches(self):
        super().invalidate_caches()
        if partfield is not None:
            partfield.get_installed_partfield_version.cache_clear()

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
                (
                    "MOD_EXPLODE",
                    "Tree Segmentation",
                    "partuv.segmentation",
                    partuv.segmentation == "TREE",
                ),
            )
        )

    def describe(self):
        if get_dev_build_path(PARTUV.name) is not None:
            return f"{self.label} v{PARTUV_VERSION} (local build)"
        version = PARTUV.installed_version() or PARTUV_VERSION
        return f"{self.label} v{version}"

    def update_pending(self):
        engine_behind = (
            get_dev_build_path(PARTUV.name) is None and PARTUV.update_available()
        )
        return engine_behind or (
            partfield is not None and partfield.partfield_update_pending()
        )

    def install_too_old(self):
        return PARTUV.install_too_old()

    def draw_prefs(self, layout):
        if not self.is_available():
            layout.row().label(
                text="PartUV is only available on Windows, Linux and Apple Silicon Macs",
                icon="ERROR",
            )
            return
        if task_state["running"] and task_state["owner"] == self.install_owner:
            draw_progress(layout, self.install_phase)
            return

        dev_build = get_dev_build_path(PARTUV.name) is not None
        installed = PARTUV.is_downloaded()
        ai_installed = partfield is not None and partfield.is_partfield_installed()
        update_pending = self.update_pending()
        if dev_build:
            layout.row().label(text="Local build", icon="CHECKMARK")
        else:
            draw_install_status(
                layout,
                PARTUV.installed_version(),
                PARTUV_VERSION,
                too_old=PARTUV.install_too_old(),
                update_pending=update_pending,
            )

        tiers = [
            ("Geometric and Tree", "uvgami.install_partuv", installed or dev_build)
        ]
        # partfield runs on cuda, and macs have no nvidia gpu
        if partfield is not None and get_platform_tag() != "macos-arm64":
            tiers.append(("AI Segmentation", "uvgami.install_partfield", ai_installed))
        update_operator = (
            "uvgami.install_partfield" if ai_installed else "uvgami.install_partuv"
        )
        wants_download = update_pending or not all(
            tier_installed for _, _, tier_installed in tiers
        )
        if not (wants_download and draw_online_access(layout)):
            columns = layout.split(factor=1 / len(tiers))
            for header, install_operator, tier_installed in tiers:
                column = columns.column()
                title = column.row()
                title.alignment = "CENTER"
                title.label(text=header)
                button = column.row()
                button.scale_y = 1.5
                if update_pending and install_operator == update_operator:
                    button.operator(
                        install_operator, text="Update Engine", icon=UPDATE_ICON
                    )
                elif tier_installed:
                    button.alignment = "CENTER"
                    button.label(text="Downloaded", icon="CHECKMARK")
                else:
                    button.operator(
                        install_operator, text="Download Engine", icon="IMPORT"
                    )
        if installed or ai_installed:
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
        shared = [
            "--input-list",
            str(input_list),
            "--output-dir",
            str(output_dir),
            "--overwrite",
            "--threshold",
            PRIORITY_VALUES[props.priority],
            "--visual",
        ]
        segmentation = props.partuv.segmentation
        if segmentation == "AI":
            return partfield.build_partfield_args(shared, ctx.partuv_path)
        args = [str(ctx.partuv_path), *shared, "--segmentation", segmentation.lower()]
        if segmentation == "TREE":
            args += ["--uvgami-tools-path", ctx.tools_path]
        return args

    def describe_failure(self, code):
        return {
            2: EngineFailure("Invalid input mesh", moves_to_invalid=True),
            # 3 is a missing checkpoint, cuda or uvgami-tools, with the cause in stderr
            3: EngineFailure("PartUV could not start", moves_to_invalid=False),
            4: EngineFailure("PartUV failed on this mesh", moves_to_invalid=True),
            5: EngineFailure("PartUV produced invalid output", moves_to_invalid=True),
        }.get(code) or super().describe_failure(code)

    def stop(self, process, ctx):
        # ai mode runs partuv as a child of python, which outlives a kill of python alone
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
