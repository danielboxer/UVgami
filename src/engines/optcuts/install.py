import bpy

from ..binary_engine import EngineRelease, InstallEngineTask

# must match the optcuts engine VERSION
OPTCUTS_VERSION = "1.21.9"
OPTCUTS_MINIMUM_VERSION = "1.21.0"
OPTCUTS = EngineRelease(
    "optcuts", "Optcuts", OPTCUTS_VERSION, OPTCUTS_MINIMUM_VERSION, "2 MB"
)


class UVGAMI_OT_install_optcuts(InstallEngineTask, bpy.types.Operator):
    bl_idname = "uvgami.install_optcuts"
    bl_label = "Download Optcuts Engine"
    owner = "optcuts"
    release = OPTCUTS
