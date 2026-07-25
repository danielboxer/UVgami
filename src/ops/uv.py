import bpy

from ..utils.mesh import edit_restore, select_uvs, validate_obj


def pack():
    select_uvs()
    if bpy.context.scene.uvgami.fix_scale:
        bpy.ops.uv.average_islands_scale()
    bpy.ops.uv.pack_islands(margin=bpy.context.scene.uvgami.margin)
    bpy.ops.uv.select_all(action="DESELECT")


def show_seams():
    select_uvs()
    bpy.ops.uv.mark_seam(clear=True)
    bpy.ops.uv.seams_from_islands()
    bpy.ops.uv.select_all(action="DESELECT")
    bpy.ops.mesh.select_all(action="DESELECT")


class UVGAMI_OT_pack(bpy.types.Operator):
    bl_idname = "uvgami.pack"
    bl_label = "Pack"
    bl_description = "Pack UVs with Blender's packer"
    bl_options = {"UNDO"}

    def execute(self, context):
        combine_uvs = context.scene.uvgami.combine_uvs
        valid_objs = []

        for obj in context.selected_objects:
            if not validate_obj(self, obj, check_uvs=True):
                continue
            if combine_uvs:
                valid_objs.append(obj)
            else:
                edit_restore([obj], pack)

        if combine_uvs:
            edit_restore(valid_objs, pack)

        return {"FINISHED"}
