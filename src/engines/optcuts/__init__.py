import bpy

from ... import part_overrides
from ...hard_surface import preseed_work, seam_flags
from ...no_bpy import SURFACE_MODE_CODES, uvs_collapsed_to_points
from ...ops.guides import painted_weight_by_vertex
from ...utils.io import print_stdin
from ...utils.mesh import corner_uvs
from ...utils.ui import is_non_default, only_active
from .. import EngineFailure
from .flags import SPLIT_TARGETS, bound_and_weight_args
from .install import OptcutsInstall

SEAM_BORDER_TURN_DEGREES = {"LESS_STRETCH": 45, "BALANCED": 66, "FEWER_SEAMS": 75}


SURFACE_MODES = [
    ("AUTO", "Auto", "Pick Organic or Hard Surface for each loose part by its shape"),
    ("ORGANIC", "Organic", "Let the engine place every seam"),
    ("HARD", "Hard Surface", "Cut seams on sharp features before the engine runs"),
]
SURFACE_MODE_LABELS = {mode: label for mode, label, _ in SURFACE_MODES}


class UVGAMI_PG_optcuts(bpy.types.PropertyGroup):
    mode: bpy.props.EnumProperty(
        name="Mode",
        description=(
            "Whether seams are cut on sharp features or all placed by the engine."
            " Edit Parts overrides this for single parts"
        ),
        items=SURFACE_MODES,
        default="AUTO",
    )


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
    )
    supports_guided = True
    supports_viewer = True
    supports_early_stop = True
    supports_import_uvs = True
    supports_proxy = True

    def draw_settings(self, layout, props):
        if not self.supports_hard_surface:
            return
        row = layout.row()
        row.label(icon="MOD_BEVEL", text="Mode")
        row.prop(props.optcuts, "mode", expand=True)

    def active_settings(self, props):
        if not self.supports_hard_surface:
            return []
        return only_active(
            (
                (
                    "MOD_BEVEL",
                    SURFACE_MODE_LABELS[props.optcuts.mode],
                    "optcuts.mode",
                    is_non_default(props, "optcuts.mode"),
                ),
            )
        )

    def import_uvs_ignored(self, props):
        return self.supports_hard_surface and props.optcuts.mode == "HARD"

    def unset_surface_mode(self, props):
        if not self.supports_hard_surface:
            return "ORGANIC"
        return props.optcuts.mode

    def preseed_work(self, obj, props, plane, surface_modes):
        if not self.supports_hard_surface:
            return None
        if surface_modes is None:
            surface_modes = part_overrides.surface_modes(
                obj.data, self.unset_surface_mode(props)
            )
        if plane is None and (surface_modes == SURFACE_MODE_CODES["ORGANIC"]).all():
            return None
        restrictions = None
        if props.avoid_seams:
            # optcuts puts fewer seams on vertices with a higher weight in this group
            painted = painted_weight_by_vertex(obj).items()
            restrictions = {
                vertex: weight for vertex, weight in painted if weight
            } or None
        return preseed_work(
            obj,
            angle=SEAM_BORDER_TURN_DEGREES[props.priority],
            split_target=SPLIT_TARGETS[props.priority],
            engine_args=bound_and_weight_args(props.priority, props.weight_value),
            weights=restrictions,
            surface_modes=surface_modes,
            plane=plane,
        )

    def piece_uses_uvs(self, obj, props, has_uvs):
        if not self.uses_import_uvs(props):
            has_uvs = has_uvs and bool(seam_flags(obj.data).any())
        if not has_uvs or obj.data.uv_layers.active is None:
            return has_uvs
        # optcuts can fail re-cutting a flatten collapsed to points
        return not uvs_collapsed_to_points(corner_uvs(obj.data))

    def _loose_bound_pinned(self, input_path, props):
        pinned = (input_path.parent / f"{input_path.stem}_fixed").is_file()
        # under the loose bound a broken patch already counts as feasible
        return pinned and props.priority == "FEWER_SEAMS"

    def build_args(self, ctx, input_path, props):
        priority = props.priority
        if self._loose_bound_pinned(input_path, props):
            priority = "BALANCED"
        return [
            *self.command_prefix(ctx),
            "-i",
            str(input_path),
            *bound_and_weight_args(priority, props.weight_value),
        ]

    def build_shared_args(self, ctx, input_path, props, threads):
        # optcuts takes the bound only as a startup flag
        if self._loose_bound_pinned(input_path, props):
            return None
        cap = ["-t", str(threads)] if threads else []
        return [
            *self.command_prefix(ctx),
            *bound_and_weight_args(props.priority, props.weight_value),
            *cap,
        ]

    def describe_failure(self, code):
        return {
            -1: EngineFailure("Mesh needs cleanup", moves_to_invalid=True),
            101: EngineFailure("Non Manifold Edges", moves_to_invalid=True),
            102: EngineFailure("Non Manifold Vertices", moves_to_invalid=True),
            103: EngineFailure("Failed To Load Mesh", moves_to_invalid=True),
            104: EngineFailure("Unknown Mesh Format", moves_to_invalid=True),
            105: EngineFailure("Invalid Geometry", moves_to_invalid=True),
            106: EngineFailure("Mesh Not Triangulated", moves_to_invalid=True),
            107: EngineFailure("Invalid UV Input", moves_to_invalid=True),
            108: EngineFailure("Unsupported Mesh Topology", moves_to_invalid=True),
            109: EngineFailure("Initial Cut Failed", moves_to_invalid=True),
            110: EngineFailure("Area UVs Too Broken To Pin", moves_to_invalid=True),
            112: EngineFailure("Flatten Failed", moves_to_invalid=True),
            113: EngineFailure("Invalid Coordinates", moves_to_invalid=True),
            114: EngineFailure("Island UVs Too Broken To Relax", moves_to_invalid=True),
            115: EngineFailure("Inconsistent Face Orientation", moves_to_invalid=True),
            116: EngineFailure("Zero Area Faces", moves_to_invalid=True),
            117: EngineFailure("Symmetry Plane Needs UVs", moves_to_invalid=True),
            118: EngineFailure("Output Could Not Be Saved", moves_to_invalid=False),
            120: EngineFailure("Mesh needs cleanup", moves_to_invalid=True),
            121: EngineFailure("Invalid Quality Bound", moves_to_invalid=False),
            122: EngineFailure("Degenerate UV Boundary", moves_to_invalid=True),
            # optcuts exits 90 from its terminate handler with the cause in stderr
        }.get(code) or super().describe_failure(code)

    def request_early_stop(self, process):
        return print_stdin(process, "stop")

    def request_cancel(self, process):
        return print_stdin(process, "cancel")

    def request_snapshot(self, process):
        print_stdin(process, "snapshot")


ENGINE = OptcutsEngine()
