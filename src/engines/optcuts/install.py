import bpy

from ..binary_engine import EngineRelease, InstallEngineTask

# must match engine/optcuts/VERSION (check-engine-versions.yml fails on drift)
OPTCUTS_VERSION = "1.16.0"
OPTCUTS = EngineRelease("optcuts", "Optcuts", OPTCUTS_VERSION, "2 MB")


class UVGAMI_OT_install_optcuts(InstallEngineTask, bpy.types.Operator):
    bl_idname = "uvgami.install_optcuts"
    bl_label = "Download Optcuts Engine"
    bl_description = "Download the Optcuts engine"
    done_message = "Optcuts engine downloaded"
    owner = "optcuts"
    release = OPTCUTS
