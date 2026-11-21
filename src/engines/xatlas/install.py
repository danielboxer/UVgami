import bpy

from ..binary_engine import (
    BinaryEngine,
    EngineRelease,
    InstallEngineTask,
    UVGAMI_OT_delete_engine,
)

# must match the xatlas engine VERSION
XATLAS_VERSION = "0.2.8"
XATLAS_MINIMUM_VERSION = "0.2.0"
XATLAS_ARCHIVE_SHA256S = {
    "windows": "0836f686230f61800a8e3d57fbf8ee1225ec6b5751b005ffb41c696f3e4d630b",
    "linux": "781f8e280bef569577b63e6ff08d94b36ed9b4ff44de86b3fd970324bfc74bfa",
    "macos-arm64": "b26c0f50b7439ce2b841ebe33b57698fccf64dc481c4375e123be09b40be2a20",
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
    release = XATLAS


class XatlasInstall(BinaryEngine):
    release = XATLAS
    supports_regions = True
    supports_symmetry = True
    supports_material_seams = True
    supports_keep_quads = True
    supports_proxy = True
    supports_painted_islands = True
    classes = (UVGAMI_OT_install_xatlas, UVGAMI_OT_delete_engine)
