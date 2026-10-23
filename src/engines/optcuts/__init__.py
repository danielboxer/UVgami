import bpy

from ... import part_overrides
from ...hard_surface import preseed_work, seam_flags, seam_restrictions
from ...seams import SURFACE_MODE_CODES, uvs_collapsed
from ...utils.io import print_stdin
from ...utils.mesh import corner_uvs
from ...utils.ui import is_non_default, only_active
from .install import OptcutsInstall

PRIORITY_VALUES = {"LESS_STRETCH": "4.05", "BALANCED": "4.2", "FEWER_SEAMS": "5.0"}
# degrees a region border must turn to stay a seam, a duller border is merged away
HARD_SURFACE_ANGLES = {"LESS_STRETCH": 45, "BALANCED": 66, "FEWER_SEAMS": 75}
# the preseed's length cap target per priority, texture use a strip may drag down to
SPLIT_TARGETS = {"LESS_STRETCH": 0.6, "BALANCED": 0.5, "FEWER_SEAMS": 0.4}


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
    supports_keep_quads = True
    supports_import_uvs = True
    supports_proxy = True
    supports_hard_surface = True
    supports_plane_glue = True
    supports_regions = True
    supports_symmetry = True

    def draw_settings(self, layout, props):
        row = layout.row()
        row.label(icon="MOD_BEVEL", text="Mode")
        row.prop(props.optcuts, "mode", expand=True)

    def active_settings(self, props):
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

    # the preseed uvs replace the imported ones
    def import_uvs_ignored(self, props):
        return props.optcuts.mode == "HARD"

    def unset_surface_mode(self, props):
        return props.optcuts.mode

    # a plane mirrors the seams of every face, organic ones included
    def preseed_work(self, obj, props, plane, surface_modes):
        if surface_modes is None:
            surface_modes = part_overrides.surface_modes(
                obj.data, self.unset_surface_mode(props)
            )
        if plane is None and (surface_modes == SURFACE_MODE_CODES["ORGANIC"]).all():
            return None
        return preseed_work(
            obj,
            HARD_SURFACE_ANGLES[props.priority],
            SPLIT_TARGETS[props.priority],
            self._bound_and_weight_args(props, props.priority),
            seam_restrictions(obj) if props.avoid_seams else None,
            surface_modes,
            plane,
        )

    def piece_uses_uvs(self, obj, props, has_uvs):
        # a piece the preseed skipped has no seams and goes to the engine bare
        if not self.uses_import_uvs(props):
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
