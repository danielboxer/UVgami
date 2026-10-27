from ...utils.paths import engine_output_path
from .install import XatlasInstall

# above 4.0 the output is identical
PRIORITY_VALUES = {"LESS_STRETCH": "0.1", "BALANCED": "2.0", "FEWER_SEAMS": "4.0"}


class XatlasEngine(XatlasInstall):
    id = "XATLAS"
    enum_value = 2
    label = "xatlas"
    description = "Fast CPU engine for baking lightmaps and texture painting"
    icon = "MESH_GRID"

    # xatlas has no hard surface route
    def unset_surface_mode(self, props):
        return "ORGANIC"

    def build_args(self, ctx, input_path, props):
        return [
            *self.command(ctx),
            "-i",
            str(input_path),
            "-o",
            str(engine_output_path(input_path)),
            "--max-cost",
            PRIORITY_VALUES[props.priority],
        ]

    def describe_failure(self, code):
        return {
            2: ("Invalid input mesh", True),
            3: ("Invalid geometry", True),
            4: ("Unwrap failed", True),
        }.get(code) or super().describe_failure(code)


ENGINE = XatlasEngine()
