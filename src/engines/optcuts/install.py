import bpy

from ..binary_engine import (
    BinaryEngine,
    EngineRelease,
    InstallEngineTask,
    UVGAMI_OT_delete_engine,
)

PROXY_MODE_DEFAULT = "AUTO"
# must match the optcuts engine VERSION
OPTCUTS_VERSION = "1.24.0"
OPTCUTS_MINIMUM_VERSION = "1.23.0"
OPTCUTS_ARCHIVE_SHA256S = {
    "windows": "c898c8e6d779296e754467542d3e4d1b8b3dcd9e21faeea834dfa24a30da1e88",
    "linux": "08eb36a36b8c893f84551733399284084986b2a8b676b818b826f672c468deb6",
    "macos-arm64": "76f07f451a0a2d87ebbc61f9c59d601a9bb97f207746a77e55d8cecb4d85fa01",
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
    release = OPTCUTS


class OptcutsInstall(BinaryEngine):
    release = OPTCUTS
    supports_keep_quads = True
    supports_hard_surface = True
    supports_painted_islands = True
    keeps_plane_vertices_on_u_zero = True
    supports_regions = True
    supports_symmetry = True
    classes = (UVGAMI_OT_install_optcuts, UVGAMI_OT_delete_engine)
