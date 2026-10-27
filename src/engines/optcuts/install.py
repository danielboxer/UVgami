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
    "windows": "c545846fd2c711cd9cef90b3e9940d7acc60d3f34f258c4fb30cb434c975d4bd",
    "linux": "6ec848c160002968c8af7d76c0381f1331239822c5ca2d787f0d2912db4c5531",
    "macos-x64": "1ce3202978e1cec7396d3c8fe6c3876f23e001f784e3899569ce1ef6dea29448",
    "macos-arm64": "96c0983e6c0dc6e6d5ef88afbc556fa77819f9c96ad68ff9a5177409f6d966cd",
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
