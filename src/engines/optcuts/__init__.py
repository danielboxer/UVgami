import math

import bpy

from ...hard_surface import (
    auto_hard_faces,
    build_seam_uvs,
    preseed_work,
    seam_restrictions,
)
from ...utils.io import print_stdin
from ...utils.ui import only_active
from ..binary_engine import BinaryEngine
from .install import OPTCUTS, UVGAMI_OT_install_optcuts


QUALITY_LABELS = {
    "LESS_STRETCH": "Less Stretch",
    "BALANCED": "Balanced",
    "FEWER_SEAMS": "Fewer Seams",
}


class UVGAMI_PG_optcuts(bpy.types.PropertyGroup):
    use_hard_surface: bpy.props.BoolProperty(
        name="",
        description="Cut seams on sharp features. Good for mechanical shapes",
    )
    hard_surface_auto: bpy.props.BoolProperty(
        name="Auto",
        description=(
            "Automatically unwrap sharp objects in hard surface mode and"
            " otherwise in normal mode"
        ),
    )
    hard_surface_angle: bpy.props.FloatProperty(
        name="Angle",
        description="What counts as a sharp feature. Lower keeps more seams",
        subtype="ANGLE",
        default=math.radians(66),
        min=math.radians(1),
        max=math.radians(180),
    )
    quality: bpy.props.EnumProperty(
        name="Priority",
        description="Whether the unwrap favors less stretching or fewer seams",
        items=(
            (
                "LESS_STRETCH",
                "Less Stretch",
                "Lowest stretching, longest seams. Slowest",
            ),
            ("BALANCED", "Balanced", "Low stretching with moderate seams"),
            ("FEWER_SEAMS", "Fewer Seams", "Shortest seams, allows visible stretching"),
        ),
        default="BALANCED",
    )

    @property
    def is_auto(self):
        return self.use_hard_surface and self.hard_surface_auto


class UVGAMI_PT_hard_surface(bpy.types.Panel):
    bl_label = "Hard Surface"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 0

    @classmethod
    def poll(cls, context):
        # imported here: engines imports this module back
        from .. import active_engine

        return active_engine(context.scene.uvgami.engine) is ENGINE

    def draw_header(self, context):
        self.layout.prop(context.scene.uvgami.optcuts, "use_hard_surface")

    def draw(self, context):
        optcuts = context.scene.uvgami.optcuts
        layout = self.layout
        layout.active = optcuts.use_hard_surface
        box = layout.box()

        row = box.row()
        row.alignment = "CENTER"
        row.label(text="Hard Surface", icon="MOD_BEVEL")

        box.prop(optcuts, "hard_surface_auto")

        row = box.row()
        row.label(icon="DRIVER_ROTATIONAL_DIFFERENCE", text="Angle")
        row.prop(optcuts, "hard_surface_angle", text="")


class OptcutsEngine(BinaryEngine):
    id = "OPTCUTS"
    enum_value = 0
    label = "Optcuts"
    description = (
        "Default CPU engine. Highest quality but can be slow."
        " Includes the UV island operators"
    )
    icon = "UV"
    property_group = UVGAMI_PG_optcuts
    classes = (
        UVGAMI_PG_optcuts,
        UVGAMI_OT_install_optcuts,
        UVGAMI_PT_hard_surface,
    )
    supports_guided = True
    supports_viewer = True
    supports_early_stop = True
    supports_preserve = True
    supports_import_uvs = True
    release = OPTCUTS
    uses_engine_path = True

    def draw_settings(self, layout, props):
        row = layout.row()
        row.label(icon="SOLO_OFF", text="Priority")
        row.prop(props.optcuts, "quality", text="")

    def active_settings(self, props):
        optcuts = props.optcuts
        hard_surface = "Hard Surface" + (" (Auto)" if optcuts.hard_surface_auto else "")
        return only_active(
            (
                (
                    "MOD_BEVEL",
                    hard_surface,
                    "optcuts.use_hard_surface",
                    optcuts.use_hard_surface,
                ),
                (
                    "SOLO_OFF",
                    QUALITY_LABELS[optcuts.quality],
                    "optcuts.quality",
                    optcuts.quality != "BALANCED",
                ),
            )
        )

    def prepare_uvs(self, obj, props):
        optcuts = props.optcuts
        if not optcuts.use_hard_surface:
            return props.import_uvs
        only = None
        if optcuts.is_auto:
            only = auto_hard_faces(obj)
            if not only:
                return props.import_uvs
            if len(only) == len(obj.data.polygons):
                only = None
        applied = build_seam_uvs(
            obj,
            math.degrees(optcuts.hard_surface_angle),
            weights=seam_restrictions(obj) if props.avoid_seams else None,
            only=only,
        )
        return applied or props.import_uvs

    def preseed_work(self, obj, props):
        optcuts = props.optcuts
        if not optcuts.use_hard_surface:
            return None
        return preseed_work(
            obj,
            math.degrees(optcuts.hard_surface_angle),
            weights=seam_restrictions(obj) if props.avoid_seams else None,
            auto=optcuts.is_auto,
        )

    def piece_uses_uvs(self, obj, props, has_uvs):
        # auto mode routes per loose part: a piece the preseed skipped has no
        # seams and goes to the engine bare, to be cut from scratch. With
        # import uvs on, organic pieces keep the user's map instead
        if not props.optcuts.is_auto or props.import_uvs:
            return has_uvs
        return has_uvs and any(e.use_seam for e in obj.data.edges)

    def build_args(self, ctx, input_path, props):
        bounds = {"LESS_STRETCH": "4.05", "BALANCED": "4.2", "FEWER_SEAMS": "5.0"}
        u = bounds[props.optcuts.quality]
        pinned = (input_path.parent / f"{input_path.stem}_fixed").is_file()
        if pinned and props.optcuts.quality == "FEWER_SEAMS":
            # a pinned repair under the loose bound is a no-op: the broken
            # patch already counts as feasible, so no cut gets added
            u = bounds["BALANCED"]
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
            110: ("Area UVs Too Broken To Pin", True),
            111: ("Island UVs Too Broken To Combine", True),
            113: ("Invalid Coordinates", True),
            114: ("Island UVs Too Broken To Relax", True),
            # 90 (the engine's terminate handler) stays unmapped on purpose:
            # the unknown-code path surfaces the fatal line from stderr
        }.get(code) or super().describe_failure(code)

    def request_early_stop(self, process):
        return print_stdin(process, "stop")

    def request_snapshot(self, process):
        print_stdin(process, "snapshot")


ENGINE = OptcutsEngine()
