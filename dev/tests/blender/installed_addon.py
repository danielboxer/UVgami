import importlib
import sys
from pathlib import Path

import bpy

sys.path.append(str(Path(__file__).parent))
import timer_pump  # noqa: E402

CUBE = Path(__file__).parents[1] / "fixtures" / "cube.obj"
ENGINE_NAMES = ("OPTCUTS", "XATLAS")
START_TIMEOUT_SECONDS = 60
UNWRAP_TIMEOUT_SECONDS = 300


def unwrap_cube(module, engine_name):
    manager = importlib.import_module(module + ".src.manager").manager
    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.wm.obj_import(filepath=str(CUBE))
    cube = bpy.data.objects[0]
    cube.select_set(True)
    bpy.context.view_layer.objects.active = cube
    bpy.context.scene.uvgami.engine = engine_name
    with timer_pump.pump_timers() as pump:
        bpy.ops.uvgami.start()
        pump.run_until(lambda: manager.is_active, START_TIMEOUT_SECONDS)
        pump.run_until(lambda: not manager.is_active, UNWRAP_TIMEOUT_SECONDS)
    unwrapped = [o for o in bpy.data.objects if o.name.endswith("_unwrapped")]
    if manager.error_messages or not unwrapped or not unwrapped[0].data.uv_layers:
        return f"{manager.summary} {manager.error_messages}"
    return None


def check_installed_addon(module, prepare_engines):
    bpy.ops.preferences.addon_enable(module=module)
    paths = importlib.import_module(module + ".src.utils.paths")
    engines = importlib.import_module(module + ".src.engines")
    prefs = paths.get_preferences()
    prefs.show_progress_bar = False
    prepare_engines(engines)
    engines.invalidate_engine_caches()

    failures = []
    for name in ENGINE_NAMES:
        error = engines.get_engine(name).validate(prefs)[1] or unwrap_cube(module, name)
        if error:
            failures.append(f"{name}: {error}")
        else:
            print(f"{name}: unwrapped")

    bpy.ops.preferences.addon_disable(module=module)
    bpy.ops.preferences.addon_enable(module=module)
    if failures:
        raise AssertionError("\n".join(failures))


# the same download the preferences button runs, minus its thread
def download_engines(engines):
    for name in ENGINE_NAMES:
        engines.get_engine(name).release.install()


if __name__ == "__main__":
    check_installed_addon(sys.argv[sys.argv.index("--") + 1], download_engines)
