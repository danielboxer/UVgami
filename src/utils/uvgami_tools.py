import os

from ..no_bpy.engine_run import EngineError, ToolsCommand
from .engine_binaries import (
    get_dev_build_path,
    get_engine_binary_name,
    get_platform_tag,
)
from .paths import get_dir_path

# must match the engine's VERSION
UVGAMI_TOOLS_VERSION = "0.5.0"

TOOLS_NAME = "uvgami-tools"


def tools_command():
    path = get_dev_build_path(TOOLS_NAME)
    if path is None:
        tag = get_platform_tag()
        if tag is None:
            raise EngineError("UVgami tools have no build for this platform")
        path = get_dir_path() / "bin" / tag / get_engine_binary_name(TOOLS_NAME)
    if not path.is_file():
        raise EngineError(f"UVgami tools binary missing at {path}, reinstall UVgami")
    # blender's zip install drops the executable bit
    if not os.access(path, os.X_OK):
        path.chmod(0o755)
    return ToolsCommand([str(path)], None)
