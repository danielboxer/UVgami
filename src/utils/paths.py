import functools
import pathlib
import tomllib

import bpy


def get_dir_path():
    return pathlib.Path(__file__).parents[2]


@functools.cache
def read_manifest():
    with (get_dir_path() / "blender_manifest.toml").open("rb") as file:
        return tomllib.load(file)


def get_addon_version():
    return read_manifest()["version"]


def get_addon_name():
    return read_manifest()["name"]


def get_root_package():
    # the bl_ext install prefix isn't always the same depth
    return __package__.rsplit(".", 2)[0]


def get_addon_id():
    return get_root_package()


def get_preferences():
    return bpy.context.preferences.addons[get_addon_id()].preferences


def get_extension_dir_path():
    return pathlib.Path(bpy.utils.extension_path_user(get_root_package(), create=True))


# engine binaries read argv in the ansi codepage
def engine_file_stem(name):
    return "".join(c if c.isascii() else "_" for c in bpy.path.clean_name(name))


# names can repeat across pieces, and the output file is keyed by stem
def unused_input_path(folder, name, claimed=frozenset()):
    path = folder / f"{engine_file_stem(name)}.obj"
    while path.is_file() or path in claimed:
        path = path.parent / f"{path.stem}1.obj"
    return path


ENGINE_FILE_SUFFIXES = (
    ".obj",
    "_weights",
    "_importance",
    "_edges",
    "_fixed",
    "_plane",
    "_locked",
    "_inputs.txt",
)


def get_io_dir_paths():
    input_path = get_extension_dir_path() / "input"
    output_path = get_extension_dir_path() / "output"
    input_path.mkdir(exist_ok=True)
    output_path.mkdir(exist_ok=True)
    return input_path, output_path


# the engine writes its result under the input's stem
def engine_output_path(input_path):
    return get_io_dir_paths()[1] / f"{input_path.stem}.obj"


def clear_io_dir(path):
    root = get_extension_dir_path()
    if not path.is_relative_to(root):
        raise ValueError(f"refusing to clear {path}, outside {root}")
    for file in path.iterdir():
        if file.is_file() and file.name.endswith(ENGINE_FILE_SUFFIXES):
            file.unlink()
