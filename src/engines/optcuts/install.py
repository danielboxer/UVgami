import bpy

from ..binary_engine import (
    BinaryEngine,
    EngineRelease,
    InstallEngineTask,
    UVGAMI_OT_delete_engine,
)

# must match the optcuts engine VERSION
OPTCUTS_VERSION = "1.23.1"
OPTCUTS_MINIMUM_VERSION = "1.23.0"
OPTCUTS_ARCHIVE_SHA256S = {
    "windows": "924cac80bcaa9f3fdf82fc7d75522110441efa7c43babe09e2488e5e4066a0cd",
    "linux": "428fcb1cf5253dff1a9ba944ce50818198319a961fb440bfd1da238aafff113f",
    "macos-x64": "53bcee016614896e58f3203065fec55d106630bdc86155ffda4e2d9ec5598b72",
    "macos-arm64": "bae86425275926504c9fc3411c88059f842b833e6938cae051d9bdec9dccc6ee",
}
OPTCUTS = EngineRelease(
    "optcuts",
    "Optcuts",
    OPTCUTS_VERSION,
    OPTCUTS_MINIMUM_VERSION,
    "3 MB",
    OPTCUTS_ARCHIVE_SHA256S,
)


class UVGAMI_OT_install_optcuts(InstallEngineTask, bpy.types.Operator):
    bl_idname = "uvgami.install_optcuts"
    bl_label = "Download Optcuts Engine"
    owner = "optcuts"
    release = OPTCUTS


# the flags here need this binary or the bundled uvgami-tools binary
class OptcutsInstall(BinaryEngine):
    release = OPTCUTS
    supports_keep_quads = True
    supports_hard_surface = True
    supports_plane_glue = True
    supports_regions = True
    supports_symmetry = True
    classes = (UVGAMI_OT_install_optcuts, UVGAMI_OT_delete_engine)
