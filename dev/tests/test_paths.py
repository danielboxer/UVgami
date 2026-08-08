import importlib.util
import sys
import types
from pathlib import Path

# the stub must not outlive the load, other test dirs collect a real bpy
bpy_stub = types.ModuleType("bpy")
bpy_stub.path = types.SimpleNamespace(clean_name=lambda name: name)

spec = importlib.util.spec_from_file_location(
    "addon_paths", Path(__file__).parents[2] / "src" / "utils" / "paths.py"
)
paths = importlib.util.module_from_spec(spec)
real_bpy = sys.modules.get("bpy")
sys.modules["bpy"] = bpy_stub
try:
    spec.loader.exec_module(paths)
finally:
    if real_bpy is None:
        del sys.modules["bpy"]
    else:
        sys.modules["bpy"] = real_bpy


def test_ascii_names_are_unchanged():
    assert paths.engine_file_stem("robot_part_01") == "robot_part_01"


def test_non_ascii_is_replaced_so_the_engine_can_open_the_path():
    assert paths.engine_file_stem("テスト_モデル") == "_______"
    assert paths.engine_file_stem("café") == "caf_"
