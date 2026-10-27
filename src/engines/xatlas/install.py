import bpy

from ..binary_engine import (
    BinaryEngine,
    EngineRelease,
    InstallEngineTask,
    UVGAMI_OT_delete_engine,
)

# must match the xatlas engine VERSION
XATLAS_VERSION = "0.2.6"
XATLAS_MINIMUM_VERSION = "0.2.0"
XATLAS_ARCHIVE_SHA256S = {
    "windows": "472852449fe83c7e0c88e69188518d31b1c7bcf0698f856221040a0d0716e7b7",
    "linux": "00dbba44c02606bd36cd97bf81bb62fa040810e78624943c06a72bf5d690fc99",
    "macos-x64": "ede9a441500f512406050fc50417fc04f61b188006c269ec0843e9585a679d5b",
    "macos-arm64": "578701fa34817f1825fbc19310cc5528b77aa9ff20bf92c54acc6feb648346a7",
}
XATLAS = EngineRelease(
    "xatlas",
    "xatlas",
    XATLAS_VERSION,
    XATLAS_MINIMUM_VERSION,
    "300 KB",
    XATLAS_ARCHIVE_SHA256S,
)


class UVGAMI_OT_install_xatlas(InstallEngineTask, bpy.types.Operator):
    bl_idname = "uvgami.install_xatlas"
    bl_label = "Download xatlas Engine"
    owner = "xatlas"
    release = XATLAS


# regions need the bundled segment binary
class XatlasInstall(BinaryEngine):
    release = XATLAS
    supports_regions = True
    classes = (UVGAMI_OT_install_xatlas, UVGAMI_OT_delete_engine)
