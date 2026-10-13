import bpy

from ...hard_surface import (
    auto_hard_faces,
    build_seam_uvs,
    preseed_work,
    seam_flags,
    seam_restrictions,
)
from ...seams import uvs_collapsed
from ...utils.io import print_stdin
from ...utils.mesh import corner_uvs
from ...utils.ui import is_non_default, only_active
from .install import OptcutsInstall

# degrees, what counts as a sharp feature
HARD_SURFACE_ANGLE = 66

PRIORITY_VALUES = {"LESS_STRETCH": "4.05", "BALANCED": "4.2", "FEWER_SEAMS": "5.0"}


SURFACE_MODES = [
    ("ORGANIC", "Organic", "Let the engine place every seam"),
    ("HARD", "Hard Surface", "Cut seams on sharp features before the engine runs"),
    ("AUTO", "Auto", "Unwrap each loose part as hard surface or organic by its shape"),
]
SURFACE_MODE_LABELS = {mode: label for mode, label, _ in SURFACE_MODES}


class UVGAMI_PG_optcuts(bpy.types.PropertyGroup):
    surface_mode: bpy.props.EnumProperty(
        name="Mode",
        description="Whether seams follow sharp features or the engine places them",
        items=SURFACE_MODES,
        default="AUTO",
    )

    @property
    def is_auto(self):
        return self.surface_mode == "AUTO"

    @property
    def cuts_sharp_features(self):
        return self.surface_mode != "ORGANIC"


# class UVGAMI_OT_quick_unwrap(bpy.types.Operator):
#     bl_idname = "uvgami.quick_unwrap"
#     bl_label = "Quick Unwrap"
#     bl_description = "Mark the hard surface seams and flatten them"
#     bl_options = {"UNDO"}
#
#     def execute(self, context):
#         props = context.scene.uvgami
#         optcuts = props.optcuts
#         angle = HARD_SURFACE_ANGLE
#         guided = props.avoid_seams
#         selected = list(context.selected_objects)
#         active = context.view_layer.objects.active
#         mode = active.mode if active is not None else "OBJECT"
#         if mode != "OBJECT":
#             bpy.ops.object.mode_set(mode="OBJECT")
#
#         counts = []
#         flatten_error = None
#         for obj in selected:
#             if not validate_obj(self, obj):
#                 continue
#             only = None
#             if optcuts.is_auto:
#                 only = auto_hard_faces(obj)
#                 if not only:
#                     counts.append("organic")
#                     continue
#                 if len(only) == len(obj.data.polygons):
#                     only = None
#             weights = seam_restrictions(obj) if guided else None
#             try:
#                 applied = build_seam_uvs(obj, angle, weights=weights, only=only)
#             except FlattenError as error:
#                 flatten_error = str(error)
#                 break
#             if not applied:
#                 counts.append("no seams")
#                 continue
#             counts.append(str(int(seam_flags(obj.data).sum())))
#
#         deselect_all()
#         for obj in selected:
#             obj.select_set(True)
#         context.view_layer.objects.active = active
#         if active is not None and mode != "OBJECT":
#             bpy.ops.object.mode_set(mode=mode)
#
#         if flatten_error is not None:
#             self.report({"ERROR"}, flatten_error)
#             # FINISHED so the undo step covers meshes already written
#             return {"FINISHED"}
#         if not counts:
#             self.report({"ERROR"}, "Select a mesh with faces")
#             return {"CANCELLED"}
#         self.report({"INFO"}, f"Seams: {', '.join(counts)}")
#         return {"FINISHED"}


class OptcutsEngine(OptcutsInstall):
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
        *OptcutsInstall.classes,
        # UVGAMI_OT_quick_unwrap,
    )
    supports_guided = True
    supports_viewer = True
    supports_early_stop = True
    supports_preserve = True
    supports_import_uvs = True
    supports_proxy = True

    def draw_settings(self, layout, props):
        row = layout.row()
        row.label(icon="MOD_BEVEL", text="Mode")
        row.prop(props.optcuts, "surface_mode", text="")

    def active_settings(self, props):
        return only_active(
            (
                (
                    "MOD_BEVEL",
                    SURFACE_MODE_LABELS[props.optcuts.surface_mode],
                    "optcuts.surface_mode",
                    is_non_default(props, "optcuts.surface_mode"),
                ),
            )
        )

    # the auto preseed leaves the organic parts' uvs for the engine to import
    def import_uvs_ignored(self, props):
        return props.optcuts.surface_mode == "HARD"

    def prepare_uvs(self, obj, props):
        optcuts = props.optcuts
        if not optcuts.cuts_sharp_features:
            return props.import_uvs
        only = None
        if optcuts.is_auto:
            only = auto_hard_faces(obj)
            if not only:
                return False
            if len(only) == len(obj.data.polygons):
                only = None
        applied = build_seam_uvs(
            obj,
            HARD_SURFACE_ANGLE,
            weights=seam_restrictions(obj) if props.avoid_seams else None,
            only=only,
        )
        return applied

    def preseed_work(self, obj, props, mirrors=None):
        optcuts = props.optcuts
        if not optcuts.cuts_sharp_features:
            return None
        return preseed_work(
            obj,
            HARD_SURFACE_ANGLE,
            weights=seam_restrictions(obj) if props.avoid_seams else None,
            auto=optcuts.is_auto,
            mirrors=mirrors,
        )

    def piece_uses_uvs(self, obj, props, has_uvs):
        # a piece the preseed skipped has no seams and goes to the engine bare
        if props.optcuts.is_auto and not props.import_uvs:
            has_uvs = has_uvs and bool(seam_flags(obj.data).any())
        if not has_uvs or obj.data.uv_layers.active is None:
            return has_uvs
        # a collapsed flatten goes bare too, the engine can fail re-cutting it
        return not uvs_collapsed(corner_uvs(obj.data))

    # under the loose bound a broken patch already counts as feasible
    def _loose_bound_pinned(self, input_path, props):
        pinned = (input_path.parent / f"{input_path.stem}_fixed").is_file()
        return pinned and props.priority == "FEWER_SEAMS"

    def _bound_and_weight_args(self, props, priority):
        u = PRIORITY_VALUES[priority]
        s = {5: "200", 4: "150", 3: "100", 2: "50", 1: "25"}.get(props.weight_value, "")
        return f"-u {u} -s {s}".split()

    def build_args(self, ctx, input_path, props):
        priority = props.priority
        if self._loose_bound_pinned(input_path, props):
            priority = "BALANCED"
        return [
            *self.command(ctx),
            "-i",
            str(input_path),
            *self._bound_and_weight_args(props, priority),
        ]

    def build_shared_args(self, ctx, input_path, props, threads):
        # the bound is fixed per process
        if self._loose_bound_pinned(input_path, props):
            return None
        cap = ["-t", str(threads)] if threads else []
        return [
            *self.command(ctx),
            *self._bound_and_weight_args(props, props.priority),
            *cap,
        ]

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
            115: ("Inconsistent Face Orientation", True),
            116: ("Zero Area Faces", True),
            # 90 is the terminate handler, unmapped so stderr's fatal line surfaces
        }.get(code) or super().describe_failure(code)

    def request_early_stop(self, process):
        return print_stdin(process, "stop")

    def request_cancel(self, process):
        return print_stdin(process, "cancel")

    def request_snapshot(self, process):
        print_stdin(process, "snapshot")


ENGINE = OptcutsEngine()
