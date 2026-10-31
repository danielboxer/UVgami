import os
import pathlib
import platform

ENGINE_DIR_ENV_VAR = "UVGAMI_ENGINE_DIR"


def get_platform_tag():
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


def local_engine_dirs():
    override = os.environ.get(ENGINE_DIR_ENV_VAR)
    if override:
        yield pathlib.Path(override)
    tag = get_platform_tag()
    if tag is not None:
        yield pathlib.Path(__file__).parents[2] / "engine-builds" / tag


def get_dev_build_path(name):
    binary_name = get_engine_binary_name(name)
    for directory in local_engine_dirs():
        engine_path = directory / binary_name
        if engine_path.is_file():
            return engine_path
    return None
