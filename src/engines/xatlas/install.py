import bpy

from ..binary_engine import (
    BinaryEngine,
    EngineRelease,
    InstallEngineTask,
    UVGAMI_OT_delete_engine,
)

# must match the xatlas engine VERSION
XATLAS_VERSION = "0.2.5"
XATLAS_MINIMUM_VERSION = "0.2.0"
XATLAS_ARCHIVE_SHA256S = {
    "windows": "f4545018283b6f1141213674a075ed7a3379e3556404e0db4f205df3d79aca23",
    "linux": "2ea27e2a56948c907976026cf7c077e65b2763d3616fbc4f52521ac7e7a5504f",
    "macos-x64": "c77de393c2f2442365faffa97357910cc05d82303d687b537702c329d8fc6dff",
    "macos-arm64": "dda8e59603be14dae48ddbca3b0d4a5db679f7ae383490265f9875143c6da391",
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


# how xatlas is found, downloaded and started
class XatlasInstall(BinaryEngine):
    release = XATLAS
    classes = (UVGAMI_OT_install_xatlas, UVGAMI_OT_delete_engine)
