from ...utils.paths import engine_output_path
from .. import EngineFailure
from .install import XatlasInstall

# above 4.0 the output is identical
PRIORITY_VALUES = {"LESS_STRETCH": "0.1", "BALANCED": "2.0", "FEWER_SEAMS": "4.0"}


class XatlasEngine(XatlasInstall):
    id = "XATLAS"
    enum_value = 2
    label = "xatlas"
    description = "Fast CPU engine for baking lightmaps and texture painting"
    icon = "MESH_GRID"
    # xatlas takes 196s on hard surface parts of 1.1M triangles at full resolution, 1.8M ran past 13 minutes
    auto_proxy_hard_surface_faces = 1_000_000

    def build_args(self, ctx, input_path, props):
        return [
            *self.command_prefix(ctx),
            "-i",
            str(input_path),
            "-o",
            str(engine_output_path(input_path)),
            "--max-cost",
            PRIORITY_VALUES[props.priority],
        ]

    def proxy_surface_mode(self, props):
        return "AUTO"

    def describe_failure(self, code):
        return {
            2: EngineFailure("Invalid input mesh", moves_to_invalid=True),
            3: EngineFailure("Invalid geometry", moves_to_invalid=True),
            4: EngineFailure("Unwrap failed", moves_to_invalid=True),
        }.get(code) or super().describe_failure(code)


ENGINE = XatlasEngine()
