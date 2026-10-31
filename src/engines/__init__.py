import importlib.util
from collections import namedtuple

from ..no_bpy.engine_run import EngineError

EngineFailure = namedtuple("EngineFailure", "message moves_to_invalid")


class Engine:
    id = ""
    label = ""
    description = ""
    icon = "TOOL_SETTINGS"
    # blend files store enum_value, never change or reuse one
    enum_value = 0
    property_group = None
    classes = ()
    supports_guided = False
    supports_viewer = False
    supports_early_stop = False
    supports_keep_quads = False
    supports_import_uvs = False
    supports_proxy = False
    supports_hard_surface = False
    keeps_plane_vertices_on_u_zero = False
    supports_regions = False
    supports_symmetry = False
    has_install_controls = False

    def describe(self):
        return self.label

    def is_available(self):
        return True

    def is_installed(self):
        try:
            self.validate()
        except EngineError:
            return False
        return True

    def validate(self):
        raise NotImplementedError

    def invalidate_caches(self):
        pass

    def draw_settings(self, layout, props):
        pass

    def active_settings(self, props):
        return []

    def import_uvs_ignored(self, props):
        return False

    def uses_import_uvs(self, props):
        return (
            props.import_uvs
            and self.supports_import_uvs
            and not self.import_uvs_ignored(props)
        )

    def uses_proxy(self, props):
        return props.use_proxy and self.supports_proxy

    def splits_regions(self, props):
        return (
            props.split_regions
            and self.supports_regions
            and not self.uses_import_uvs(props)
        )

    def uses_symmetry(self, props):
        return (
            props.symmetry
            and self.supports_symmetry
            and not self.uses_import_uvs(props)
        )

    def unset_surface_mode(self, props):
        return "AUTO"

    def preseed_work(self, obj, props, plane, surface_modes):
        return None

    def piece_uses_uvs(self, obj, props, has_uvs):
        return has_uvs

    def draw_prefs(self, layout):
        pass

    def draw_update_notice(self, layout):
        pass

    def draw_not_installed(self, layout, waiting_for=None):
        pass

    def batches_queue(self, props):
        return False

    def command_prefix(self, ctx):
        return [str(ctx)]

    def build_args(self, ctx, input_path, props):
        raise NotImplementedError

    def build_batch_args(self, ctx, input_paths, props):
        raise NotImplementedError

    def build_shared_args(self, ctx, input_path, props, threads):
        return None

    def build_env(self, ctx):
        pass

    def describe_failure(self, code):
        # -1073741819 is the windows access violation code 0xC0000005
        if code == -1073741819:
            return EngineFailure("Engine crashed", moves_to_invalid=True)
        return None

    def request_early_stop(self, process):
        raise NotImplementedError

    def request_snapshot(self, process):
        raise NotImplementedError

    def request_cancel(self, process):
        return False

    def stop(self, process, ctx):
        process.kill()


from . import optcuts  # noqa: E402

_engines = [optcuts.ENGINE]

# some builds ship without the xatlas and partuv folders
for optional in ("xatlas", "partuv"):
    if importlib.util.find_spec(f"{__name__}.{optional}") is not None:
        _engines.append(importlib.import_module(f"{__name__}.{optional}").ENGINE)

ENGINES = {engine.id: engine for engine in _engines}


_installed_cache = None


def invalidate_engine_caches():
    global _installed_cache
    _installed_cache = None
    for engine in ENGINES.values():
        engine.invalidate_caches()


def installed_engines():
    global _installed_cache
    if _installed_cache is None:
        _installed_cache = [
            engine
            for engine in ENGINES.values()
            if engine.is_available() and engine.is_installed()
        ]
    return _installed_cache


def active_engine(engine_id):
    engine = ENGINES.get(engine_id)
    if engine is not None and engine in installed_engines():
        return engine
    return None
