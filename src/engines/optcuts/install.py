import bpy

from ..binary_engine import (
    BinaryEngine,
    EngineRelease,
    InstallEngineTask,
    UVGAMI_OT_delete_engine,
)

PROXY_MODE_DEFAULT = "AUTO"
# must match the optcuts engine VERSION
OPTCUTS_VERSION = "1.25.0"
OPTCUTS_MINIMUM_VERSION = "1.23.0"
OPTCUTS_ARCHIVE_SHA256S = {
    "windows": "a000c416d5ae0ce3a1e122a99b92a0ab879fc1fd641b91a646e98d045e6ef7c8",
    "linux": "0df5f23ca59ab0f3d00e4c0f83ba1cea49895533287153710c67a046f2a8198d",
    "macos-arm64": "f07a14b7818b88aae51f679ab252fc04be6385fa9a1602117ea9a962338fcaae",
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
    supports_material_seams = True
    classes = (UVGAMI_OT_install_optcuts, UVGAMI_OT_delete_engine)
