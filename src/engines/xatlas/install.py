import bpy

from ..binary_engine import (
    BinaryEngine,
    EngineRelease,
    InstallEngineTask,
    UVGAMI_OT_delete_engine,
)

# must match the xatlas engine VERSION
XATLAS_VERSION = "0.2.5"
XATLAS_MINIMUM_VERSION = "0.2.0"
XATLAS = EngineRelease(
    "xatlas", "xatlas", XATLAS_VERSION, XATLAS_MINIMUM_VERSION, "300 KB"
)


class UVGAMI_OT_install_xatlas(InstallEngineTask, bpy.types.Operator):
    bl_idname = "uvgami.install_xatlas"
    bl_label = "Download xatlas Engine"
    owner = "xatlas"
    release = XATLAS


# how xatlas is found, downloaded and started
class XatlasInstall(BinaryEngine):
    release = XATLAS
    classes = (UVGAMI_OT_install_xatlas, UVGAMI_OT_delete_engine)
