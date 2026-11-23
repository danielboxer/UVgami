import bpy

from ..binary_engine import (
    BinaryEngine,
    EngineRelease,
    InstallEngineTask,
    UVGAMI_OT_delete_engine,
)

# must match the xatlas engine VERSION
XATLAS_VERSION = "0.2.9"
XATLAS_MINIMUM_VERSION = "0.2.0"
XATLAS_ARCHIVE_SHA256S = {
    "windows": "449d58878775e2dcc2bc486a5c4f46fa1cf8ffe12ae814aac6ce2bdb29beca82",
    "linux": "3f84a83ed2bdd3670f4bd3f8399b9d591c6b35fd5c66b86e9219668830ca928e",
    "macos-arm64": "190ed8af4e7b6ad31e499f1793d8fd199fbe0f63babbdaa87ac1a1f7b46835a3",
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
