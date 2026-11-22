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
    "windows": "bbf31c3e3c8768ffb663463aa570d1c0178c6e0a5c6cf41cfb771684853febee",
    "linux": "2e64c60b98d06234cf320412b4f96c20fe96c3aeb6ef0da74440586a92e9d455",
    "macos-arm64": "7d65814f7cc3c0e9501ac3c4baafdf535af44b1383f664652907f1ff51581287",
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
