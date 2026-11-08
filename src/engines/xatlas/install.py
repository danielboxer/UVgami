import bpy

from ..binary_engine import (
    BinaryEngine,
    EngineRelease,
    InstallEngineTask,
    UVGAMI_OT_delete_engine,
)

# must match the xatlas engine VERSION
XATLAS_VERSION = "0.2.7"
XATLAS_MINIMUM_VERSION = "0.2.0"
XATLAS_ARCHIVE_SHA256S = {
    "windows": "7c100395c60182e1488aa36f3332ec9259e983be22ddf242b1afdd17b9c7c78d",
    "linux": "4b61dd2fd02ffdd450c1583adfcfc473f893e5dffab02a422f2a62bebe035c37",
    "macos-arm64": "fd80af5ab66006357de97fdfbaaec55c14a83a425197c7915a39be6a502ca243",
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
    classes = (UVGAMI_OT_install_xatlas, UVGAMI_OT_delete_engine)
