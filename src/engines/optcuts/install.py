import bpy

from ..binary_engine import (
    BinaryEngine,
    EngineRelease,
    InstallEngineTask,
    UVGAMI_OT_delete_engine,
)
from ..install_task import (
    UPDATE_ICON,
    draw_online_access,
    draw_progress,
    task_state,
)

# must match the optcuts engine VERSION
OPTCUTS_VERSION = "1.21.10"
OPTCUTS_MINIMUM_VERSION = "1.21.0"
OPTCUTS = EngineRelease(
    "optcuts", "Optcuts", OPTCUTS_VERSION, OPTCUTS_MINIMUM_VERSION, "2 MB"
)


class UVGAMI_OT_install_optcuts(InstallEngineTask, bpy.types.Operator):
    bl_idname = "uvgami.install_optcuts"
    bl_label = "Download Optcuts Engine"
    owner = "optcuts"
    release = OPTCUTS


# how optcuts is found, downloaded and started
class OptcutsInstall(BinaryEngine):
    release = OPTCUTS
    classes = (UVGAMI_OT_install_optcuts, UVGAMI_OT_delete_engine)

    def draw_not_installed(self, layout, waiting_for=None):
        box = layout.box()
        if task_state["running"] and waiting_for in (None, task_state["owner"]):
            draw_progress(box, "Downloading engine")
            return
        outdated = self.release.install_too_old()
        row = box.row()
        row.alignment = "CENTER"
        if outdated:
            row.label(text="Engine update required", icon="FILE_REFRESH")
        else:
            row.label(text="Engine not downloaded", icon="INFO")
        if draw_online_access(box):
            return
        row = box.row()
        row.scale_y = 1.5
        # skip the confirmation, this is the only way to get an engine
        row.operator_context = "EXEC_DEFAULT"
        row.operator(
            "uvgami.install_optcuts",
            text="Update Engine" if outdated else "Download Engine",
            icon=UPDATE_ICON if outdated else "IMPORT",
        )
