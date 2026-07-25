import pathlib

import bpy

from . import Engine
from ..utils.io import print_stdin
from ..utils.paths import get_bundled_engine_path


class UVGAMI_PG_optcuts(bpy.types.PropertyGroup):
    quality: bpy.props.EnumProperty(
        name="Unwrap Quality",
        description=(
            "A higher quality unwrap will have less stretching, "
            "but it will take longer to finish"
        ),
        items=(
            ("HIGH", "High", ""),
            ("MEDIUM", "Medium", ""),
            ("LOW", "Low", ""),
        ),
        default="MEDIUM",
    )


class OptcutsEngine(Engine):
    id = "OPTCUTS"
    label = "Optcuts"
    description = "The default Optcuts unwrapping engine"
    property_group = UVGAMI_PG_optcuts
    classes = (UVGAMI_PG_optcuts,)
    supports_guided = True
    supports_viewer = True
    supports_early_stop = True
    supports_preserve = True
    supports_import_uvs = True

    def validate(self, prefs):
        raw = pathlib.Path(prefs.engine_path)
        if str(raw) == ".":
            # try bundled engine as fallback
            bundled = get_bundled_engine_path("optcuts")
            if bundled is None:
                return (
                    None,
                    "Engine path is not set. Set the path in the add-on preferences",
                )
            path = bundled
        else:
            if not raw.is_file():
                return None, "Engine path doesn't exist"
            if raw.stem != "optcuts":
                return None, "Engine path is incorrect"
            path = raw

        return path, None

    def draw_settings(self, layout, props):
        row = layout.row()
        row.label(icon="SOLO_OFF", text="Quality")
        row.prop(props.optcuts, "quality", text="")

    def draw_prefs(self, layout, prefs):
        row = layout.row()
        row.scale_y = 1.5
        split = row.split(factor=0.2)
        split.scale_x = 1.5
        split.label(text="Engine Path")
        split.prop(prefs, "engine_path")

        engine_path = pathlib.Path(prefs.engine_path)
        if str(engine_path) == "." and get_bundled_engine_path("optcuts") is not None:
            row = layout.row()
            row.label(text="Using bundled optcuts engine", icon="CHECKMARK")

    def build_args(self, ctx, input_path, props):
        u = {"HIGH": "4.05", "MEDIUM": "4.1"}.get(props.optcuts.quality, "4.2")
        s = {5: "200", 4: "150", 3: "100", 2: "50", 1: "25"}.get(props.weight_value, "")
        shared_args = f"-u {u} -s {s}"

        return [str(ctx), "-i", str(input_path)] + shared_args.split()

    def describe_failure(self, code):
        return {
            -1: ("Mesh needs cleanup", True),
            101: ("Non Manifold Edges", True),
            102: ("Non Manifold Vertices", True),
            105: ("Invalid Geometry", True),
            107: ("Invalid UV Input", True),
            108: ("Unsupported Mesh Topology", True),
            109: ("Initial Cut Failed", True),
        }.get(code) or super().describe_failure(code)

    def request_early_stop(self, process):
        return print_stdin(process, "stop")

    def request_snapshot(self, process):
        print_stdin(process, "snapshot")


ENGINE = OptcutsEngine()
