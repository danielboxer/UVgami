import bpy

# from ..hard_surface import apply_seams, marked_seams
# from ..proxy import bounds_frame, snap_cuts, vertex_map
from ..pack import pack_objects
from ..utils.mesh import validate_obj

# class UVGAMI_OT_transfer_seams(bpy.types.Operator):
#     bl_idname = "uvgami.transfer_seams"
#     bl_label = "Transfer Seams"
#     bl_description = (
#         "Copy the seams of the other selected object onto the active object"
#     )
#     bl_options = {"UNDO"}
#
#     @classmethod
#     def poll(cls, context):
#         return context.mode == "OBJECT"
#
#     def execute(self, context):
#         target = context.view_layer.objects.active
#         if target is None or not validate_obj(self, target, report=True):
#             return {"CANCELLED"}
#         others = [obj for obj in context.selected_objects if obj is not target]
#         if len(others) != 1:
#             self.report({"ERROR"}, "Select one other object to copy the seams from")
#             return {"CANCELLED"}
#         donor = others[0]
#         if not validate_obj(self, donor, report=True):
#             return {"CANCELLED"}
#         seams = marked_seams(donor.data)
#         if not seams:
#             self.report({"ERROR"}, f"{donor.name} has no seams")
#             return {"CANCELLED"}
#
#         mapped = vertex_map(target, donor, bounds_frame(target), bounds_frame(donor))
#         apply_seams(target.data, snap_cuts(target, mapped, seams))
#         self.report({"INFO"}, f"Transferred {len(seams)} seams from {donor.name}")
#         return {"FINISHED"}


class UVGAMI_OT_pack(bpy.types.Operator):
    bl_idname = "uvgami.pack"
    bl_label = "Pack"
    bl_description = "Pack UVs with Blender's packer"
    bl_options = {"UNDO"}

    def execute(self, context):
        props = context.scene.uvgami
        packable = []

        for obj in context.selected_objects:
            if not validate_obj(self, obj) or not obj.data.uv_layers:
                continue
            if props.combine_uvs:
                packable.append(obj)
            else:
                pack_objects([obj], props)

        if props.combine_uvs:
            pack_objects(packable, props)

        return {"FINISHED"}
