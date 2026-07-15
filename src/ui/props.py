import multiprocessing

import bpy

from ..engines import ENGINES, get_engine
from ..utils.paths import get_addon_id


def update_engine(self, context):
    # reset pack-after-unwrap to the new engine's default when switching
    self.pack_after_unwrap = get_engine(self.engine).pack_by_default


class UVGAMI_PG_properties(bpy.types.PropertyGroup):
    engine: bpy.props.EnumProperty(
        name="Engine",
        description="The unwrapping engine to use",
        items=tuple(
            (e.id, e.label, e.description)
            for e in ENGINES.values()
            if e.is_available()
        ),
        default="OPTCUTS",
        update=update_engine,
    )
    import_uvs: bpy.props.BoolProperty(
        name="", description="Use the UV map on the mesh as input"
    )
    # preserve mesh
    untriangulate: bpy.props.BoolProperty(
        name="",
        description="Untriangulate mesh after unwrap. N-gons might not be preserved",
    )
    maintain_mode: bpy.props.EnumProperty(
        name="Preserve",
        description="How much of the mesh to untriangulate after unwrap",
        items=(
            (
                "FULL",
                "Full",
                (
                    "Fully untriangulate mesh and reroute seams."
                    " This might cause some areas to overlap slightly."
                    " There might also be a small amount of increased stretching."
                    " N-gons will remain triangulated."
                ),
            ),
            ("PARTIAL", "Partial", "Untriangulate all areas except for the seams"),
        ),
    )
    # speed
    concurrent: bpy.props.BoolProperty(
        name="",
        description=(
            "Unwrap multiple meshes at the same time."
            " This only has an effect if you are unwrapping multiple meshes, "
            "or if the mesh is made up of multiple joined meshes"
        ),
    )
    max_cores: bpy.props.IntProperty(
        name="",
        description="The maximum number of processor cores to use for concurrent mode",
        default=int(multiprocessing.cpu_count() / 2 - 1),
        max=multiprocessing.cpu_count(),
        min=1,
    )
    early_stop: bpy.props.IntProperty(
        name="",
        description=(
            "When to stop the unwrap."
            " This is based on the amount of stretching in the UV map"
        ),
        min=1,
        max=100,
        default=100,
        subtype="PERCENTAGE",
    )
    unwrap_timeout: bpy.props.IntProperty(
        name="",
        description=(
            "Maximum time in minutes for each unwrap."
            " Timed out meshes will be moved to the invalid collection."
            " Set to 0 to disable"
        ),
        min=0,
        max=120,
        default=0,
    )
    use_cuts: bpy.props.BoolProperty(
        name="",
        description=("Cut the input mesh into pieces. This will speed up the unwrap"),
    )
    cut_type: bpy.props.EnumProperty(
        name="Cut Type",
        description="Where the mesh will be cut",
        items=(
            ("EVEN", "Even", "Make even cuts on the chosen axes"),
            ("SEAMS", "Seams", "Make cuts on the seams"),
        ),
    )
    cuts: bpy.props.IntProperty(
        name="",
        description="The amount of cuts to make in the mesh",
        min=1,
        max=15,
        default=1,
    )
    cut_axes: bpy.props.EnumProperty(
        name="Axes",
        description=(
            "Limit cuts to specific axes."
            " Hold down Shift to select or deselect multiple axes"
        ),
        items=(
            ("X", "X", "X axis"),
            ("Y", "Y", "Y axis"),
            ("Z", "Z", "Z axis"),
        ),
        options={"ENUM_FLAG"},
    )
    # seam restrictions
    use_guided_mode: bpy.props.BoolProperty(
        name="", description="Avoid placing seams on parts of the mesh"
    )
    weight_value: bpy.props.IntProperty(
        name="",
        description=(
            "A higher weight will follow the seam restrictions more "
            "but will take longer to finish the unwrap"
        ),
        min=1,
        max=5,
        default=3,
    )
    # symmetry
    use_symmetry: bpy.props.BoolProperty(
        name="",
        description=(
            "Use this setting for symmetrical meshes only."
            " This will result in a quicker unwrap with a symmetrical UV map"
        ),
    )
    sym_axes: bpy.props.EnumProperty(
        name="Axes",
        description=(
            "The axis or axes of symmetry of the input mesh."
            " Hold down Shift to select or deselect multiple axes"
        ),
        items=(
            ("X", "X", "X axis"),
            ("Y", "Y", "Y axis"),
            ("Z", "Z", "Z axis"),
        ),
        # allows for selection of multiple items
        options={"ENUM_FLAG"},
    )
    sym_merge: bpy.props.BoolProperty(
        name="Merge",
        description=(
            "Overlap and combine symmetrical UVs. This will remove the seam on the axis"
        ),
        default=True,
    )
    # grid
    grid_type: bpy.props.EnumProperty(
        name="Grid Type",
        description="The type of grid material that will be added",
        items=(
            ("UV", "UV", "Normal UV grid"),
            ("COLOUR", "Colour", "Coloured UV grid"),
        ),
    )
    grid_res: bpy.props.IntProperty(
        name="Resolution",
        description="The resolution of the grid texture in pixels",
        default=1024,
        subtype="PIXEL",
        min=1,
        max=16384,
    )
    auto_grid: bpy.props.BoolProperty(
        name="Auto Grid", description="Automatically add a UV grid after unwrapping"
    )
    # pack
    margin: bpy.props.FloatProperty(
        name="", description="The space between UV islands", min=0, max=1
    )
    combine_uvs: bpy.props.BoolProperty(
        name="Combine UVs",
        description="Pack UVs of all selected objects into a single UV map",
    )
    fix_scale: bpy.props.BoolProperty(
        name="Average Islands Scale",
        description="Scale UV islands based on their actual size",
        default=True,
    )
    pack_after_unwrap: bpy.props.BoolProperty(
        name="Pack After Unwrap",
        description="Automatically pack UVs after each unwrap finishes",
    )
    transfer_uvs: bpy.props.BoolProperty(
        name="",
        description=(
            "Transfer the UV map from the output mesh to the original input mesh."
            " Works when the output has the same topology as the input, or a"
            " triangulated version of it. The original object must be unchanged"
            " since starting the unwrap"
        ),
    )
    preview_unwrap_sharp: bpy.props.BoolProperty(
        name="",
        description=(
            "Preview: Only mark sharp edges as seams. Use this for high poly meshes"
        ),
    )


# each engine contributes a pointer to its own settings group, keyed by engine id
for engine in ENGINES.values():
    UVGAMI_PG_properties.__annotations__[engine.id.lower()] = bpy.props.PointerProperty(
        type=engine.property_group
    )


class UVGAMI_AP_preferences(bpy.types.AddonPreferences):
    bl_idname = get_addon_id()

    autosave: bpy.props.BoolProperty(
        name="Autosave",
        description=(
            "Automatically save the Blender file before unwrapping "
            "to avoid losing work. This is recommended"
        ),
        default=True,
    )
    show_popup: bpy.props.BoolProperty(
        name="Show Popup",
        description=(
            "Show a popup when all meshes are finished unwrapping."
            " This might contain other information like if any objects were invalid or "
            "if there were any errors"
        ),
        default=True,
    )
    engine_path: bpy.props.StringProperty(
        name="",
        description="The path to the unwrapper application stored on your computer",
        subtype="FILE_PATH",
    )
    cleanup: bpy.props.EnumProperty(
        name="Input Cleanup",
        description="The action to perform on the original input mesh",
        items=(
            ("NONE", "None", "Leave the input mesh as it is"),
            ("HIDE", "Hide", "Hide the original input mesh"),
            ("DELETE", "Delete", "Delete the original input mesh"),
        ),
        default="HIDE",
    )
    invalid_collection: bpy.props.BoolProperty(
        name="Not Unwrapped Collection",
        description="Add meshes that failed to unwrap, were cancelled, or were"
        " stopped to a collection",
        default=True,
    )
    show_progress_bar: bpy.props.BoolProperty(
        name="Progress Bar",
        description="Display a progress bar in the 3D view during an unwrap",
        default=True,
    )
    stop_timeout: bpy.props.IntProperty(
        name="Stop Timeout",
        description=(
            "Time in minutes to wait after requesting a stop before force killing the engine."
            " Set to 0 to disable."
        ),
        min=0,
        max=60,
        default=10,
    )
    show_info: bpy.props.BoolProperty(
        name="Info",
        description="Show information about previous unwraps in the info panel",
        default=True,
    )
    viewer_workspace: bpy.props.StringProperty(
        name="Viewer Workspace",
        description=(
            "The name of the workspace that will be opened when viewing an unwrap."
            " If this is empty, the UV editor will be opened instead"
        ),
    )

    def draw(self, context):
        layout = self.layout

        for engine in ENGINES.values():
            engine.draw_prefs(layout.box(), self)

        box = layout.box()

        cf = box.column_flow(columns=3)

        row = cf.row()
        row.label(icon="FILE_TICK")
        row.prop(self, "autosave")

        row = cf.row()
        row.label(icon="WINDOW")
        row.prop(self, "show_popup")

        row = cf.row()
        row.label(icon="SORTTIME")
        row.prop(self, "show_progress_bar")

        row = cf.row()
        row.label(icon="TIME")
        row.prop(self, "stop_timeout")

        row = cf.row()
        row.label(icon="INFO")
        row.prop(self, "show_info")

        row = cf.row()
        row.label(
            icon="OUTLINER_COLLECTION" if bpy.app.version >= (2, 92, 0) else "GROUP"
        )
        row.prop(self, "invalid_collection")

        box.separator()

        row = box.row()
        row.label(icon="MOD_WIREFRAME")
        row.prop(self, "cleanup")

        if self.cleanup == "DELETE":
            row = box.row()
            row.label(
                text=(
                    "Warning: Use 'Input Cleanup: Delete' at your own risk, "
                    "losing work is possible"
                )
            )

        row = box.row()
        row.label(icon="WORKSPACE")
        row.prop(self, "viewer_workspace")
