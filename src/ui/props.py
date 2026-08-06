import multiprocessing

import bpy

from ..engines import ENGINES, installed_engines, invalidate_engine_caches
from ..utils.paths import get_addon_id

# built once so the item strings stay referenced, which blender requires for
# dynamic enum callbacks. explicit numbers keep saved files stable as the
# installed set changes
_ENGINE_ITEMS = {
    e.id: (e.id, e.label, e.description, e.enum_value) for e in ENGINES.values()
}


def _engine_items(self, context):
    return [_ENGINE_ITEMS[e.id] for e in installed_engines()]


# the getter clamps to an installed engine without touching the stored value,
# so the widget can't go blank when the selected engine is deleted, and
# reinstalling it restores the old selection
def _engine_get(self):
    stored = self.get("engine", -1)
    installed = installed_engines()
    if any(_ENGINE_ITEMS[e.id][3] == stored for e in installed):
        return stored
    return _ENGINE_ITEMS[installed[0].id][3] if installed else 0


def _engine_set(self, value):
    self["engine"] = value


class UVGAMI_PG_properties(bpy.types.PropertyGroup):
    engine: bpy.props.EnumProperty(
        name="Engine",
        description="The unwrapping engine to use",
        items=_engine_items,
        get=_engine_get,
        set=_engine_set,
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
            "Grow the selected area by this many face rings to affect a larger area"
        ),
        min=0,
        max=10,
        default=1,
    )
    stack_similar: bpy.props.BoolProperty(
        name="",
        description=(
            "Stack repeated mesh pieces by only unwrapping one and copying it."
        ),
    )
    use_proxy: bpy.props.BoolProperty(
        name="",
        description=(
            "Unwrap a low poly copy, then transfer the seams to the original."
            " Much faster for high poly meshes"
        ),
    )
    proxy_faces: bpy.props.IntProperty(
        name="",
        description="How many triangles the decimated copy keeps",
        min=100,
        max=100000,
        default=2000,
    )
    # weights
    use_weights: bpy.props.BoolProperty(
        name="", description="Use the painted weights to change the unwrap"
    )
    weight_mode: bpy.props.EnumProperty(
        name="Mode",
        description="What the painted weights do",
        items=(
            ("SEAMS", "Avoid Seams", "Keep seams off the painted faces"),
            (
                "STRETCH",
                "Reduce Stretching",
                "Prioritize the painted faces to have less stretching."
                " This also allows other faces to have more stretching.",
            ),
            ("BOTH", "Both", "Avoid seams on the painted faces and stretch them less"),
        ),
        default="SEAMS",
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
    sym_preview: bpy.props.BoolProperty(
        name="Preview",
        description="Show the symmetry planes of the selected meshes in the viewport",
        default=True,
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
        default={"X"},
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
        default=True,
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

    @property
    def avoid_seams(self):
        return self.use_weights and self.weight_mode in {"SEAMS", "BOTH"}

    @property
    def reduce_stretching(self):
        return self.use_weights and self.weight_mode in {"STRETCH", "BOTH"}


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
        update=lambda self, context: invalidate_engine_caches(),
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
    show_warnings: bpy.props.BoolProperty(
        name="Warnings",
        description="Show warning if modifier is applied before unwrap",
        default=True,
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

        grid = box.grid_flow(row_major=True, columns=3, even_columns=True)

        row = grid.row()
        row.label(icon="FILE_TICK")
        row.prop(self, "autosave")

        row = grid.row()
        row.label(icon="WINDOW")
        row.prop(self, "show_popup")

        row = grid.row()
        row.label(icon="SORTTIME")
        row.prop(self, "show_progress_bar")

        row = grid.row()
        row.label(icon="OUTLINER_COLLECTION")
        row.prop(self, "invalid_collection")

        row = grid.row()
        row.label(icon="INFO")
        row.prop(self, "show_warnings")

        box.separator()

        box.operator(
            "uvgami.reset_settings", text="Reset Settings", icon="FILE_REFRESH"
        )
