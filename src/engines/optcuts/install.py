import bpy

from ..binary_engine import (
    BinaryEngine,
    EngineRelease,
    InstallEngineTask,
    UVGAMI_OT_delete_engine,
)

# must match the optcuts engine VERSION
OPTCUTS_VERSION = "1.23.0"
OPTCUTS_MINIMUM_VERSION = "1.23.0"
OPTCUTS_ARCHIVE_SHA256S = {
    "windows": "072dee9d46e3f7426b14efb81cd66b2264ca76f61dd711f871f08b4756f49d92",
    "linux": "6ceb5414da0943fbf85a6e5cd9a7bf90130ecb13b77e57c4f4ae9d228d75d9a7",
    "macos-x64": "77c9aacbefeb19638dcbfa8e99ef2bb133cd444aead7a6a7cc4e203ba0d03215",
    "macos-arm64": "4fc181b2e67141826e41f56492659e14da92a2dde1a7c0eef54c34ec6103c6d1",
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


# the flags here need this binary or the bundled segment and hardsurface binaries
class OptcutsInstall(BinaryEngine):
    release = OPTCUTS
    supports_keep_quads = True
    supports_hard_surface = True
    supports_plane_glue = True
    supports_regions = True
    classes = (UVGAMI_OT_install_optcuts, UVGAMI_OT_delete_engine)
