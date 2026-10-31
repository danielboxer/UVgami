import bpy

from ..grid import add_grid, grid_material, remove_grid
from ..utils.mesh import validate_obj
from ..utils.ui import switch_solid_color


class UVGAMI_OT_add_grid(bpy.types.Operator):
    bl_idname = "uvgami.add_grid"
    bl_label = "Add Grid"
    bl_description = "Add grid material to all selected meshes"
    bl_options = {"UNDO"}

    def execute(self, context):
        material = grid_material()

        valid_count = 0
        for obj in context.selected_objects:
            if validate_obj(self, obj):
                add_grid(obj, material)
                valid_count += 1

        if valid_count > 0:
            switch_solid_color("TEXTURE")

        self.report({"INFO"}, "Added UV grid")
        return {"FINISHED"}


class UVGAMI_OT_remove_grid(bpy.types.Operator):
    bl_idname = "uvgami.remove_grid"
    bl_label = "Remove Grid"
    bl_description = "Remove grid material from all selected meshes"
    bl_options = {"UNDO"}

    def execute(self, context):
        for obj in context.selected_objects:
            if validate_obj(self, obj):
                remove_grid(obj)

        switch_solid_color("MATERIAL")

        return {"FINISHED"}
