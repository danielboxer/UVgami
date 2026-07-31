class Engine:
    id = ""
    label = ""
    description = ""
    icon = "TOOL_SETTINGS"
    # every bpy class the engine needs (its property group plus any operators)
    property_group = None
    classes = ()
    supports_guided = False
    supports_viewer = False
    supports_early_stop = False
    supports_preserve = False
    supports_import_uvs = False
    # whether the engine honours a _fixed sidecar of pinned uv verts
    supports_pinned = False
    # engines whose raw uvs overlap must be packed, so pack-after-unwrap is
    # forced on rather than left to the user
    requires_pack = False

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

    def prepare_uvs(self, obj, props):
        """Return whether to export obj's uv map, building one first if the
        engine wants seams of its own. obj is a temp copy, safe to edit."""
        return props.import_uvs and self.supports_import_uvs

    def piece_uses_uvs(self, obj, props, has_uvs):
        """Per separated piece, whether its uv map goes to the engine.
        has_uvs is what prepare_uvs returned for the whole object."""
        return has_uvs

    def draw_prefs(self, layout, prefs):
        """Draw this engine's section in the addon preferences."""
        pass

    def batches_queue(self, props):
        """Whether queued meshes share one engine process. Batching and running
        several processes at once are mutually exclusive."""
        return False

    def build_args(self, ctx, input_path, props):
        """Return the subprocess argv that unwraps input_path."""
        raise NotImplementedError

    def build_batch_args(self, ctx, input_paths, props):
        """Return the argv unwrapping all input_paths in one process. Must be
        implemented when batches_queue can return True."""
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
        Returns True if delivered. Only called when supports_early_stop is set."""
        raise NotImplementedError

    def request_snapshot(self, process):
        """Ask a running process to emit a uv snapshot for the live viewer. Only
        called when supports_viewer is set."""
        raise NotImplementedError

    def stop(self, process, ctx):
        """Stop a running unwrap process."""
        process.kill()


# imported after Engine because each module subclasses it
from . import optcuts, partuv, xatlas

# order sets the enum/ui order
ENGINES = {e.id: e for e in (optcuts.ENGINE, partuv.ENGINE, xatlas.ENGINE)}


def get_engine(engine_id):
    return ENGINES.get(engine_id, ENGINES["OPTCUTS"])
