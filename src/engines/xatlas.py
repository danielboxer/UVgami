import bpy

from . import Engine
from ..utils.paths import get_bundled_engine_path, get_extension_dir_path


class UVGAMI_PG_xatlas(bpy.types.PropertyGroup):
    # xatlas has no settings in v0.1, but the pointer registration needs a group
    pass


class XatlasEngine(Engine):
    id = "XATLAS"
    label = "xatlas"
    description = "Fast CPU engine for baking lightmaps and texture painting"
    icon = "MESH_GRID"
    property_group = UVGAMI_PG_xatlas
    classes = (UVGAMI_PG_xatlas,)
    # xatlas packs its own atlas, so it never needs forced packing

    def validate(self, prefs):
        # ignores prefs.engine_path, that setting is optcuts-only
        bundled = get_bundled_engine_path("xatlas")
        if bundled is None:
            return None, "Bundled xatlas engine is missing"
        return bundled, None

    def draw_prefs(self, layout, prefs):
        row = layout.row()
        _, error = self.validate(prefs)
        if error is not None:
            row.label(text=error, icon="ERROR")
        else:
            row.label(text="Using the bundled engine", icon="CHECKMARK")

    def build_args(self, ctx, input_path, props):
        output_path = get_extension_dir_path() / "output" / f"{input_path.stem}.obj"
        return [str(ctx), "-i", str(input_path), "-o", str(output_path)]

    def describe_failure(self, code):
        return {
            2: ("Invalid input mesh", True),
            3: ("Invalid geometry", True),
            4: ("Unwrap failed", True),
        }.get(code) or super().describe_failure(code)


ENGINE = XatlasEngine()
