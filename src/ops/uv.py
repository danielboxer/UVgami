from collections import namedtuple

import bmesh
import bpy

# from ..hard_surface import apply_seams, marked_seams
# from ..proxy import bounds_frame, snap_cuts, vertex_map
# from ..part_overrides import UV_SCALE_ATTRIBUTE
from ..no_bpy import faces_by_edge, islands_from_uvs
from ..similar import find_stacks
from ..utils.mesh import (
    corner_uvs,
    edit_restore,
    face_vertices,
    in_object_mode,
    validate_obj,
)

Stack = namedtuple(
    "Stack",
    [
        "kept_loops",
        "anchor_loop",
        "reference_loop",
        "anchor",
        "reference",
        "duplicate_faces",
    ],
)


def island_stacks(obj, faces=None, edges=None):
    mesh = obj.data
    if mesh.uv_layers.active is None:
        return []
    if faces is None:
        faces = face_vertices(mesh)
    if edges is None:
        edges = faces_by_edge(faces)
    uvs = corner_uvs(mesh)

    stacks = []
    groups = islands_from_uvs(faces, uvs, edges)
    for kept, duplicates in find_stacks(groups, uvs):
        anchor_loop = (kept[0], 0)
        anchor = complex(*uvs[kept[0]][0])
        corners = (
            (face_index, corner_index)
            for face_index in kept
            for corner_index in range(len(uvs[face_index]))
        )
        reference_loop = max(
            corners,
            key=lambda corner: abs(complex(*uvs[corner[0]][corner[1]]) - anchor),
        )
        reference = complex(*uvs[reference_loop[0]][reference_loop[1]])
        if reference == anchor:
            continue
        kept_loops = {}
        for face_index in kept:
            for corner_index, uv in enumerate(uvs[face_index]):
                kept_loops.setdefault(uv, (face_index, corner_index))
        stacks.append(
            Stack(
                kept_loops=kept_loops,
                anchor_loop=anchor_loop,
                reference_loop=reference_loop,
                anchor=anchor,
                reference=reference,
                duplicate_faces=[face for group in duplicates for face in group],
            )
        )
    return stacks


def _deselect_stacked_duplicates(obj, stacks):
    if not stacks:
        return
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    for stack in stacks:
        for face_index in stack.duplicate_faces:
            # uv operators act only on selected faces while sync is off
            bm.faces[face_index].select = False


def _move_duplicates_onto_kept_island(obj, stacks):
    if not stacks:
        return
    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.active
    bm.faces.ensure_lookup_table()

    def loop_uv(corner):
        face_index, corner_index = corner
        return complex(*bm.faces[face_index].loops[corner_index][uv_layer].uv)

    for stack in stacks:
        new_anchor = loop_uv(stack.anchor_loop)
        ratio = (loop_uv(stack.reference_loop) - new_anchor) / (
            stack.reference - stack.anchor
        )
        exact = {
            uv: tuple(bm.faces[face_index].loops[corner_index][uv_layer].uv)
            for uv, (face_index, corner_index) in stack.kept_loops.items()
        }
        for face_index in stack.duplicate_faces:
            for loop in bm.faces[face_index].loops:
                old = tuple(loop[uv_layer].uv)
                snapped = exact.get(old)
                if snapped is None:
                    moved = new_anchor + ratio * (complex(*old) - stack.anchor)
                    snapped = (moved.real, moved.imag)
                loop[uv_layer].uv = snapped
    bmesh.update_edit_mesh(obj.data)


# def _scale_parts(obj):
#     bm = bmesh.from_edit_mesh(obj.data)
#     scale_layer = bm.faces.layers.float.get(UV_SCALE_ATTRIBUTE)
#     if scale_layer is None:
#         return
#     uv_layer = bm.loops.layers.uv.active
#     for face in bm.faces:
#         scale = face[scale_layer]
#         if scale in (0, 1) or not face.select:
#             continue
#         for loop in face.loops:
#             loop[uv_layer].uv *= scale
#     bmesh.update_edit_mesh(obj.data)


def pack_objects(objects, props, topology=None):
    if not objects:
        return
    topology = topology or {}

    def read_and_pack():
        stacks_of = {obj: island_stacks(obj, *topology.get(obj, ())) for obj in objects}
        edit_restore(objects, pack, stacks_of, props)

    # foreach_get reads come back empty in edit mode
    in_object_mode(objects[0], read_and_pack)


def select_uvs():
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.select_all(action="SELECT")


def pack(stacks_of, props):
    tool_settings = bpy.context.scene.tool_settings
    old_sync = tool_settings.use_uv_select_sync
    # with sync on, uv operators follow the mesh selection differently per version
    tool_settings.use_uv_select_sync = False
    try:
        select_uvs()
        # blender's merge_overlap can't keep stacks together
        # rotated copies have 3d areas that differ in the last few digits
        for obj, stacks in stacks_of.items():
            _deselect_stacked_duplicates(obj, stacks)
        # without the average first, a repeat pack scales a part again
        if props.fix_scale:
            bpy.ops.uv.average_islands_scale()
            # for obj in stacks_of:
            #     _scale_parts(obj)
        bpy.ops.uv.pack_islands(margin=props.margin, rotate_method="AXIS_ALIGNED")
        for obj, stacks in stacks_of.items():
            _move_duplicates_onto_kept_island(obj, stacks)
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.select_all(action="DESELECT")
    finally:
        tool_settings.use_uv_select_sync = old_sync


def show_seams():
    select_uvs()
    bpy.ops.uv.mark_seam(clear=True)
    bpy.ops.uv.seams_from_islands()
    bpy.ops.uv.select_all(action="DESELECT")
    bpy.ops.mesh.select_all(action="DESELECT")


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
