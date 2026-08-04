import pathlib
import platform

import bpy


def get_dir_path():
    return pathlib.Path(__file__).parents[2]


def get_root_package():
    parts = __package__.split(".")
    return ".".join(parts[:3])


def get_addon_id():
    return get_root_package()


def get_preferences():
    return bpy.context.preferences.addons[get_addon_id()].preferences


def get_extension_dir_path():
    return pathlib.Path(bpy.utils.extension_path_user(get_root_package(), create=True))


def get_platform_tag():
    """Platform name used for both the local engine folders and the engine
    release asset names, or None on an unsupported platform."""
    system = platform.system()
    if system == "Windows":
        return "windows"
    if system == "Linux":
        return "linux"
    if system == "Darwin":
        if platform.machine().lower() == "arm64":
            return "macos-arm64"
        return "macos-x64"
    return None


def get_engine_binary_name(name):
    return f"{name}.exe" if platform.system() == "Windows" else name


def get_local_engine_path(name):
    """Path to an engine binary in engines/<platform>/, or None. No engines ship
    with the addon, so this only finds a build made in a dev checkout."""
    tag = get_platform_tag()
    if tag is None:
        return None
    engine_path = get_dir_path() / "engines" / tag / get_engine_binary_name(name)
    if engine_path.is_file():
        return engine_path
    return None
