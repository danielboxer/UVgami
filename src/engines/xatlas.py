import bpy

from . import Engine
from ..utils.paths import get_bundled_engine_path, get_extension_dir_path
from ..utils.ui import only_active


class UVGAMI_PG_xatlas(bpy.types.PropertyGroup):
    max_cost: bpy.props.FloatProperty(
        name="",
        description="Cost limit for growing a chart. Lower values cut the mesh into more UV islands",
        default=2.0,
        # above 10 the chart count stops dropping, xatlas' merge pass sets the floor
        min=0.1,
        max=10.0,
    )


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

    def draw_settings(self, layout, props):
        row = layout.row()
        row.label(icon="UV_ISLANDSEL", text="Chart Cost")
        row.prop(props.xatlas, "max_cost")

    def active_settings(self, props):
        max_cost = props.xatlas.max_cost
        return only_active(
            (
                (
                    "UV_ISLANDSEL",
                    f"Chart Cost {max_cost:.2f}",
                    "xatlas.max_cost",
                    max_cost != 2.0,
                ),
            )
        )

    def draw_prefs(self, layout, prefs):
        row = layout.row()
        _, error = self.validate(prefs)
        if error is not None:
            row.label(text=error, icon="ERROR")
        else:
            row.label(text="Using the bundled engine", icon="CHECKMARK")

    def build_args(self, ctx, input_path, props):
        output_path = get_extension_dir_path() / "output" / f"{input_path.stem}.obj"
        return [
            str(ctx),
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--max-cost",
            f"{props.xatlas.max_cost:.4f}",
        ]

    def describe_failure(self, code):
        return {
            2: ("Invalid input mesh", True),
            3: ("Invalid geometry", True),
            4: ("Unwrap failed", True),
        }.get(code) or super().describe_failure(code)


ENGINE = XatlasEngine()
