import shutil

import bpy

from ...utils.engine_binaries import get_platform_tag
from ..binary_engine import EngineRelease, engine_install_root
from ..install_task import (
    DELETE_DESCRIPTION,
    DELETED_MESSAGE,
    DOWNLOADED_MESSAGE,
    InstallTask,
    offline_error,
    task_state,
)
from .paths import get_partfield_checkpoint_path, get_partuv_venv_path, get_uv_path

# must match the partuv engine VERSION and partuv-partfield's pyproject.toml
PARTUV_VERSION = "0.3.0"
PARTUV_MINIMUM_VERSION = "0.2.0"
PARTUV_ARCHIVE_SHA256S = {
    "windows": "21c612bb189c951512d0f0d530f8adb1a7697bac72b020a752f7b54c5852f001",
    "linux": "c2f6d9168894f4938067127ac9a1d5cd005fa8864809c35f0d3ad2c251ced13a",
    "macos-arm64": "313abe44358a5a7d857e96b955ba8910743d6f318625b490cec198c3b9601f0e",
}
PARTUV = EngineRelease(
    "partuv",
    "PartUV",
    PARTUV_VERSION,
    PARTUV_MINIMUM_VERSION,
    "2 MB",
    PARTUV_ARCHIVE_SHA256S,
)


def uninstall_partuv():
    task_state["phase"] = "Deleting engine"
    for folder in (
        engine_install_root(PARTUV.name),
        get_partuv_venv_path(),
        get_uv_path().parent,
    ):
        if folder.is_dir():
            shutil.rmtree(folder)
    get_partfield_checkpoint_path().unlink(missing_ok=True)


class PartuvTask(InstallTask):
    owner = "partuv"

    def precheck(self):
        if get_platform_tag() not in PARTUV_ARCHIVE_SHA256S:
            return "PartUV is only available on Windows, Linux and Apple Silicon Macs"
        return None


class UVGAMI_OT_install_partuv(PartuvTask, bpy.types.Operator):
    bl_idname = "uvgami.install_partuv"
    bl_label = "Download PartUV Engine"
    bl_description = "Download the engine with geometric and tree segmentation"
    done_message = DOWNLOADED_MESSAGE

    def precheck(self):
        return super().precheck() or offline_error()

    def invoke(self, context, event):
        action = "Update" if PARTUV.update_available() else "Download"
        return context.window_manager.invoke_confirm(
            self,
            event,
            title=f"{action} PartUV",
            message=f"{PARTUV.download_size} (UVgami Engine License)",
            confirm_text=action,
        )

    def build_task(self):
        return PARTUV.install


class UVGAMI_OT_uninstall_partuv(PartuvTask, bpy.types.Operator):
    bl_idname = "uvgami.uninstall_partuv"
    bl_label = "Delete PartUV Engine"
    bl_description = DELETE_DESCRIPTION
    done_message = DELETED_MESSAGE

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def build_task(self):
        return uninstall_partuv
