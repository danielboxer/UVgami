import importlib
import pkgutil


class Engine:
    id = ""
    label = ""
    description = ""
    # the engine's bpy PropertyGroup class, and every bpy class the engine needs
    # registered (its property group plus any operators)
    property_group = None
    classes = ()
    # feature flags drive UI gating and post-processing compatibility;
    # engine-specific parameters live in the engine's property group and are
    # drawn by draw_settings instead of being flagged here
    supports_guided = False
    supports_viewer = False
    supports_early_stop = False
    supports_preserve = False
    supports_import_uvs = False
    # whether pack-after-unwrap starts enabled when this engine is selected
    pack_by_default = False

    def is_available(self):
        """Whether this engine can run on the current platform."""
        return True

    def validate(self, prefs):
        """Return (ctx, None) if usable, else (None, error_message). ctx is an
        engine-defined run context passed back to the build_* and stop methods."""
        raise NotImplementedError

    def draw_settings(self, layout, props):
        """Draw this engine's settings rows in the main panel."""
        pass

    def draw_prefs(self, layout, prefs):
        """Draw this engine's section in the addon preferences."""
        pass

    def allows_concurrent(self, props):
        """Whether multiple unwrap processes can run at once."""
        return True

    def wants_batch(self, props):
        """Whether queued meshes should share one engine process."""
        return False

    def build_args(self, ctx, input_path, props):
        """Return the subprocess argv that unwraps input_path."""
        raise NotImplementedError

    def build_batch_args(self, ctx, input_paths, props):
        """Return the argv unwrapping all input_paths in one process. Must be
        implemented when wants_batch can return True."""
        raise NotImplementedError

    def build_env(self, ctx):
        """Return the subprocess env, or None to inherit."""
        return None

    def describe_failure(self, code):
        """Map an engine exit code to (message, move_to_invalid), or None if the
        engine does not recognize it (caller shows a generic unknown-error)."""
        # windows access violation (0xC0000005): the engine process crashed
        if code == -1073741819:
            return ("Engine crashed", True)
        return None

    def request_early_stop(self, process):
        """Ask a running process to stop and finish with its current result.
        Returns True if delivered; engines that cannot stop gracefully return False."""
        return False

    def request_snapshot(self, process):
        """Ask a running process to emit a uv snapshot for the live viewer. No-op
        for engines without a viewer."""
        pass

    def stop(self, process, ctx):
        """Stop a running unwrap process."""
        process.kill()


def _load_engines():
    # discover one engine per submodule; each defines a module-level ENGINE.
    # alphabetical module order sets the enum/ui order (optcuts first).
    # deleting an engine package from a shipped zip removes that engine with it.
    engines = {}
    for module_info in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{__name__}.{module_info.name}")
        engines[module.ENGINE.id] = module.ENGINE
    return engines


ENGINES = _load_engines()


def get_engine(engine_id):
    # default to optcuts so a stale or removed engine id in an old file still
    # loads (files saved before the rename stored "UVGAMI" and land here too)
    return ENGINES.get(engine_id, ENGINES["OPTCUTS"])
