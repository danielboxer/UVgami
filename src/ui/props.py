import multiprocessing

import bpy

from ..engines import ENGINES
from ..utils.paths import get_addon_id


class UVGAMI_PG_properties(bpy.types.PropertyGroup):
    engine: bpy.props.EnumProperty(
        name="Engine",
        description="The unwrapping engine to use",
        items=tuple(
            (e.id, e.label, e.description) for e in ENGINES.values() if e.is_available()
        ),
        default="OPTCUTS",
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
    area_expand: bpy.props.IntProperty(
        name="",
        description=(
            "Grow the selected area by this many face rings before fixing."
            " The border of the grown area is what stays in place, so"
            " expanding lets the selection itself reshape and gives cuts"
            " room to land"
        ),
        min=0,
        max=10,
        default=1,
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
    use_proxy: bpy.props.BoolProperty(
        name="",
        description=(
            "Unwrap a decimated copy of the mesh, then cut the original along"
            " its seams and unwrap it in Blender. Much faster on dense meshes."
            " The UV map lands on the original object, which must be unchanged"
            " since starting the unwrap"
        ),
    )
    proxy_faces: bpy.props.IntProperty(
        name="",
        description="How many triangles the decimated copy keeps",
        min=100,
        max=100000,
        default=2000,
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
            "Show a popup when all meshes are finished unwrapping. The same"
            " summary is always shown in the panel and the status bar"
        ),
        default=False,
    )
    engine_path: bpy.props.StringProperty(
        name="",
        description="The path to the unwrapper application stored on your computer",
        subtype="FILE_PATH",
    )
    invalid_collection: bpy.props.BoolProperty(
        name="Not Unwrapped Collection",
        description="Add meshes that failed to unwrap, were cancelled, or were"
        " stopped to a collection",
        default=True,
    )
    show_progress_bar: bpy.props.BoolProperty(
        name="Progress Bar",
        description="Display a progress bar in the 3D view and UV editor during an unwrap",
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

    def draw(self, context):
        layout = self.layout

        row = layout.row()
        row.label(text="Engines", icon="TOOL_SETTINGS")

        for engine in ENGINES.values():
            box = layout.box()
            row = box.row()
            row.label(text=engine.label, icon=engine.icon)
            row = box.row()
            row.active = False
            row.label(text=engine.description)
            engine.draw_prefs(box, self)

        box = layout.box()
        row = box.row()
        row.label(text="General", icon="PREFERENCES")

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
        row.label(
            icon="OUTLINER_COLLECTION" if bpy.app.version >= (2, 92, 0) else "GROUP"
        )
        row.prop(self, "invalid_collection")

        box.separator()

        box.operator(
            "uvgami.reset_settings", text="Reset Settings", icon="FILE_REFRESH"
        )
