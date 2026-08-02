import math
import pathlib

import bpy

from . import Engine
from ..hard_surface import (
    auto_hard_faces,
    build_seam_uvs,
    preseed_work,
    seam_restrictions,
)
from ..seams import FlattenError
from ..utils.io import print_stdin
from ..utils.mesh import deselect_all, validate_obj
from ..utils.paths import get_bundled_engine_path


class UVGAMI_PG_optcuts(bpy.types.PropertyGroup):
    hard_surface: bpy.props.EnumProperty(
        name="",
        description=(
            "Cut seams on sharp features first, then unwrap. "
            "Best for mechanical shapes, uses more seams"
        ),
        items=(
            ("OFF", "Off", "No feature seams, every part unwraps from scratch"),
            (
                "ON",
                "On",
                "Cut feature seams on every part before the unwrap",
            ),
            (
                "AUTO",
                "Auto",
                "Cut feature seams only on the loose parts that read as hard "
                "surface. Organic parts unwrap from scratch, so a mixed model "
                "gets both treatments",
            ),
        ),
        default="OFF",
    )
    hard_surface_marked: bpy.props.EnumProperty(
        name="",
        description=(
            "What to do with the seams already marked on the mesh. Run Seams "
            "Unwrap, edit the marks, then unwrap. Repair can still add cuts "
            "where an island would otherwise fail"
        ),
        items=(
            ("NONE", "Ignore", "Detect every seam, marked edges are replaced"),
            (
                "ADD",
                "Add",
                "Detect seams and cut the marked edges as well, keeping them "
                "whatever the shape says",
            ),
            (
                "ONLY",
                "Only",
                "Marked edges are the whole seam set, nothing is detected",
            ),
        ),
        default="NONE",
    )
    hard_surface_angle: bpy.props.FloatProperty(
        name="Feature Angle",
        description=(
            "What counts as a sharp feature. Boundaries that turn less than "
            "this merge away, so lower keeps more seams like an artist "
            "seaming every sharp edge"
        ),
        subtype="ANGLE",
        default=math.radians(66),
        min=math.radians(1),
        max=math.radians(180),
    )
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


class UVGAMI_OT_preview_seams(bpy.types.Operator):
    bl_idname = "uvgami.preview_seams"
    bl_label = "Seams Unwrap"
    bl_description = (
        "Unwrap the selected meshes with the hard surface seams and"
        " Blender's unwrap, no engine run. The result is exactly what Seams"
        " mode would send to the engine, so it doubles as a seam preview"
    )
    bl_options = {"UNDO"}

    def execute(self, context):
        props = context.scene.uvgami
        optcuts = props.optcuts
        angle = math.degrees(optcuts.hard_surface_angle)
        marked = optcuts.hard_surface_marked
        guided = props.use_guided_mode
        selected = list(context.selected_objects)
        active = context.view_layer.objects.active
        mode = active.mode if active is not None else "OBJECT"
        if mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        counts = []
        for obj in selected:
            if not validate_obj(self, obj):
                continue
            only = None
            if optcuts.hard_surface == "AUTO":
                only = auto_hard_faces(obj, marked)
                if not only:
                    counts.append("organic")
                    continue
                if len(only) == len(obj.data.polygons):
                    only = None
            weights = seam_restrictions(obj) if guided else None
            try:
                build_seam_uvs(obj, angle, marked, weights, only)
            except FlattenError as error:
                self.report({"ERROR"}, str(error))
                return {"CANCELLED"}
            counts.append(str(sum(1 for edge in obj.data.edges if edge.use_seam)))

        deselect_all()
        for obj in selected:
            obj.select_set(True)
        context.view_layer.objects.active = active
        if active is not None and mode != "OBJECT":
            bpy.ops.object.mode_set(mode=mode)

        if not counts:
            self.report({"ERROR"}, "Select a mesh with faces")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Seams: {', '.join(counts)}")
        return {"FINISHED"}


class UVGAMI_PT_hard_surface(bpy.types.Panel):
    bl_label = "Hard Surface"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "UVgami"
    bl_parent_id = "UVGAMI_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return context.scene.uvgami.engine == "OPTCUTS"

    def draw(self, context):
        optcuts = context.scene.uvgami.optcuts
        on = optcuts.hard_surface != "OFF"
        box = self.layout.box()

        row = box.row()
        row.alignment = "CENTER"
        row.label(text="Hard Surface", icon="MOD_BEVEL")

        split = box.split(factor=0.7)
        split.label(icon="OPTIONS", text="Mode")
        split.prop(optcuts, "hard_surface", text="")

        split = box.split(factor=0.7)
        split.active = on
        split.label(icon="EDGESEL", text="Marked Seams")
        split.prop(optcuts, "hard_surface_marked", text="")

        row = box.row()
        row.active = on and optcuts.hard_surface_marked != "ONLY"
        row.label(icon="DRIVER_ROTATIONAL_DIFFERENCE", text="Feature Angle")
        row.prop(optcuts, "hard_surface_angle", text="")

        row = box.row()
        row.active = on
        row.scale_y = 1.5
        row.operator("uvgami.preview_seams", icon="UV_EDGESEL")


class OptcutsEngine(Engine):
    id = "OPTCUTS"
    label = "Optcuts"
    description = "Default CPU engine. Least stretching and islands, but slow"
    icon = "UV"
    property_group = UVGAMI_PG_optcuts
    classes = (UVGAMI_PG_optcuts, UVGAMI_OT_preview_seams, UVGAMI_PT_hard_surface)
    supports_guided = True
    supports_viewer = True
    supports_early_stop = True
    supports_preserve = True
    supports_import_uvs = True
    supports_pinned = True
    supports_combine = True

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

    def prepare_uvs(self, obj, props):
        optcuts = props.optcuts
        if optcuts.hard_surface == "OFF":
            return props.import_uvs
        only = None
        if optcuts.hard_surface == "AUTO":
            only = auto_hard_faces(obj, optcuts.hard_surface_marked)
            if not only:
                return props.import_uvs
            if len(only) == len(obj.data.polygons):
                only = None
        build_seam_uvs(
            obj,
            math.degrees(optcuts.hard_surface_angle),
            optcuts.hard_surface_marked,
            seam_restrictions(obj) if props.use_guided_mode else None,
            only,
        )
        return True

    def preseed_work(self, obj, props):
        optcuts = props.optcuts
        if optcuts.hard_surface == "OFF":
            return None
        compute, apply = preseed_work(
            obj,
            math.degrees(optcuts.hard_surface_angle),
            optcuts.hard_surface_marked,
            seam_restrictions(obj) if props.use_guided_mode else None,
            auto=optcuts.hard_surface == "AUTO",
        )
        # an auto run that finds nothing hard falls back to prepare_uvs' answer
        return compute, lambda result: apply(result) or props.import_uvs

    def piece_uses_uvs(self, obj, props, has_uvs):
        # auto mode routes per loose part: a piece the preseed skipped has no
        # seams and goes to the engine bare, to be cut from scratch. With
        # import uvs on, organic pieces keep the user's map instead
        if props.optcuts.hard_surface != "AUTO" or props.import_uvs:
            return has_uvs
        return has_uvs and any(e.use_seam for e in obj.data.edges)

    def draw_prefs(self, layout, prefs):
        row = layout.row()
        _, error = self.validate(prefs)
        if error is not None:
            row.label(text=error, icon="ERROR")
        elif str(pathlib.Path(prefs.engine_path)) == ".":
            row.label(text="Using the bundled engine", icon="CHECKMARK")
        else:
            row.label(text="Using the engine path below", icon="CHECKMARK")

        row = layout.row()
        row.scale_y = 1.5
        split = row.split(factor=0.2)
        split.scale_x = 1.5
        split.label(text="Engine Path")
        split.prop(prefs, "engine_path")

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
            110: ("Area UVs Too Broken To Pin", True),
            111: ("Island UVs Too Broken To Combine", True),
            # 90 (the engine's terminate handler) stays unmapped on purpose:
            # the unknown-code path surfaces the fatal line from stderr
        }.get(code) or super().describe_failure(code)

    def request_early_stop(self, process):
        return print_stdin(process, "stop")

    def request_snapshot(self, process):
        print_stdin(process, "snapshot")


ENGINE = OptcutsEngine()
