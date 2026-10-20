import os

from ..utils.paths import (
    get_dir_path,
    get_engine_binary_name,
    get_local_engine_path,
    get_platform_tag,
)

# each must match that engine's VERSION
SEGMENT_VERSION = "0.2.0"
HARDSURFACE_VERSION = "0.1.0"


# label names the engine in the error a missing build raises
def resolve_bundled_command(name, label):
    path = get_local_engine_path(name)
    if path is None:
        tag = get_platform_tag()
        if tag is None:
            raise RuntimeError(f"{label} has no build for this platform")
        path = get_dir_path() / "bin" / tag / get_engine_binary_name(name)
    if not path.is_file():
        raise RuntimeError(f"{label} binary missing at {path}, reinstall UVgami")
    # blender's zip install drops the executable bit
    if not os.access(path, os.X_OK):
        path.chmod(0o755)
    return [str(path)]
