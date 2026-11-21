import bpy

from ..binary_engine import (
    BinaryEngine,
    EngineRelease,
    InstallEngineTask,
    UVGAMI_OT_delete_engine,
)

PROXY_MODE_DEFAULT = "AUTO"
# must match the optcuts engine VERSION
OPTCUTS_VERSION = "1.25.1"
OPTCUTS_MINIMUM_VERSION = "1.23.0"
OPTCUTS_ARCHIVE_SHA256S = {
    "windows": "94216a26842cfda4573c61d38f59f1a67a471df469d79b3a5694bf6c58da0389",
    "linux": "5809927677afd7dbbf9cae6faf80a3f03caa81b7826f3c99a157dab2c28d37e5",
    "macos-arm64": "28dcaa5d8f7d376b0d6d74fa6520e4513214443e28bdcc236bd2db315fb9cddc",
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
