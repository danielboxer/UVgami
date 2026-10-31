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


def get_addon_id():
    # bl_ext install prefixes aren't always the same depth
    return __package__.rsplit(".", 2)[0]


def get_preferences():
    return bpy.context.preferences.addons[get_addon_id()].preferences


def get_extension_dir_path():
    return pathlib.Path(bpy.utils.extension_path_user(get_addon_id(), create=True))


def engine_file_stem(name):
    # windows passes argv to the engine binaries in the ansi codepage
    return "".join(
        character if character.isascii() else "_"
        for character in bpy.path.clean_name(name)
    )


def unused_input_path(folder, name, claimed):
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


def engine_output_path(input_path):
    _, output_path = get_io_dir_paths()
    return output_path / f"{input_path.stem}.obj"


def clear_io_dir(path):
    root = get_extension_dir_path()
    if not path.is_relative_to(root):
        raise ValueError(f"refusing to clear {path}, outside {root}")
    for file in path.iterdir():
        if file.is_file() and file.name.endswith(ENGINE_FILE_SUFFIXES):
            file.unlink()
