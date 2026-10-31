import math

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from ..utils.mesh import validate_obj

SEAM_RESTRICTIONS_GROUP = "UVgami_seam_restrictions"


def painted_weight_by_vertex(obj):
    if SEAM_RESTRICTIONS_GROUP not in obj.vertex_groups:
        return {}
    group_index = obj.vertex_groups[SEAM_RESTRICTIONS_GROUP].index
    return {
        vertex.index: group.weight
        for vertex in obj.data.vertices
        for group in vertex.groups
        if group.group == group_index
    }


SPHERE_DIRECTION_COUNT = 64


def _sphere_directions():
    directions = []
    golden_angle = math.pi * (3 - math.sqrt(5))
    for index in range(SPHERE_DIRECTION_COUNT):
        z = 1 - 2 * (index + 0.5) / SPHERE_DIRECTION_COUNT
        radius = math.sqrt(1 - z * z)
        angle = golden_angle * index
        directions.append(
            Vector((radius * math.cos(angle), radius * math.sin(angle), z))
        )
    return directions


SPHERE_DIRECTIONS = _sphere_directions()


# def _set_group_weights(obj, weights):
#     vertex_groups = obj.vertex_groups
#     if SEAM_RESTRICTIONS_GROUP not in vertex_groups:
#         vertex_groups.new(name=SEAM_RESTRICTIONS_GROUP)
#     group = vertex_groups[SEAM_RESTRICTIONS_GROUP]

#     unweighted = []
#     for index, weight in weights.items():
#         if weight > 0:
#             group.add([index], weight, "REPLACE")
#         else:
#             unweighted.append(index)
#     if unweighted:
#         group.remove(unweighted)


_old_mode = None
_old_active_group = None


def is_draw_active():
    active_object = bpy.context.active_object
    vertex_groups = active_object.vertex_groups
    if SEAM_RESTRICTIONS_GROUP not in vertex_groups:
        return False
    return (
        active_object.mode == "WEIGHT_PAINT"
        and vertex_groups.active_index == vertex_groups[SEAM_RESTRICTIONS_GROUP].index
    )


def _enter_draw_mode(context, obj):
    vertex_groups = obj.vertex_groups
    if not is_draw_active():
        global _old_mode
        _old_mode = obj.mode
        global _old_active_group
        _old_active_group = vertex_groups.active_index

    bpy.ops.object.mode_set(mode="WEIGHT_PAINT")
    if SEAM_RESTRICTIONS_GROUP not in vertex_groups:
        vertex_groups.new(name=SEAM_RESTRICTIONS_GROUP)
    vertex_groups.active_index = vertex_groups[SEAM_RESTRICTIONS_GROUP].index
    context.scene.uvgami.use_weights = True


class UVGAMI_OT_draw_guides(bpy.types.Operator):
    bl_idname = "uvgami.draw_guides"
    bl_label = "Draw"
    bl_description = "Paint the weights that change the unwrap. Red is weighted"
    bl_options = {"UNDO"}

    def execute(self, context):
        active_obj = context.active_object
        if active_obj is None:
            self.report({"ERROR"}, "No object selected")
            return {"CANCELLED"}
        if not validate_obj(self, active_obj, report=True):
            return {"CANCELLED"}

        _enter_draw_mode(context, active_obj)
        return {"FINISHED"}


# def _view_weights(obj, rv3d):
#     view_dir = rv3d.view_rotation @ Vector((0, 0, -1))
#     view_pos = rv3d.view_matrix.inverted().translation
#     matrix = obj.matrix_world
#     normal_matrix = matrix.to_3x3().inverted().transposed()

#     weights = {}
#     for v in obj.data.vertices:
#         normal = (normal_matrix @ v.normal).normalized()
#         if rv3d.is_perspective:
#             toward_viewer = (view_pos - matrix @ v.co).normalized()
#         else:
#             toward_viewer = -view_dir
#         facing = normal.dot(toward_viewer)
#         # fade past the silhouette so seams land on the true back
#         weights[v.index] = min(max((facing + 0.25) / 0.75, 0.0), 1.0) ** 2
#     return weights


def exposure_weights(obj):
    mesh = obj.data
    tree = BVHTree.FromPolygons(
        [vertex.co for vertex in mesh.vertices],
        [polygon.vertices[:] for polygon in mesh.polygons],
    )
    corners = [Vector(corner) for corner in obj.bound_box]
    offset = (corners[6] - corners[0]).length * 1e-3

    weights = {}
    for vertex in mesh.vertices:
        origin = vertex.co + vertex.normal * offset
        escaped = 0
        total = 0
        for direction in SPHERE_DIRECTIONS:
            # grazing rays hit their own surface on curves
            if direction.dot(vertex.normal) < 0.1:
                continue
            total += 1
            if tree.ray_cast(origin, direction)[0] is None:
                escaped += 1
        exposure = escaped / total if total else 1.0
        # exposure is about one on a flat surface and half in an inside corner
        weights[vertex.index] = min(max((exposure - 0.45) / 0.5, 0.0), 1.0) ** 2
    return weights


# FREE_FRACTION = 0.5


# def _rank_normalize(weights):
#     # a mostly-convex mesh gives a near-uniform high map, which slows the engine
#     order = sorted(weights, key=weights.get)
#     n = len(order)
#     if n < 2:
#         return dict.fromkeys(weights, 0.0)
#     return {
#         index: max(0.0, (rank / (n - 1) - FREE_FRACTION) / (1 - FREE_FRACTION))
#         for rank, index in enumerate(order)
#     }


# class UVGAMI_OT_seed_restrictions(bpy.types.Operator):
#     bl_idname = "uvgami.seed_restrictions"
#     bl_label = "Seed Restrictions"
#     bl_options = {"UNDO"}

#     mode: bpy.props.EnumProperty(
#         items=[
#             ("VIEW", "From View", ""),
#             ("CREVICES", "Crevices", ""),
#             ("BOTH", "Both", ""),
#         ]
#     )

#     @classmethod
#     def description(cls, context, properties):
#         base = {
#             "VIEW": (
#                 "Restrict the side of the mesh facing the current view"
#                 " so seams are placed on the back"
#             ),
#             "CREVICES": (
#                 "Restrict exposed areas so seams are placed in crevices"
#                 " and hidden pockets"
#             ),
#             "BOTH": (
#                 "Restrict areas that are facing the current view and exposed,"
#                 " so seams are placed on the back and in crevices"
#             ),
#         }[properties.mode]
#         return base + ". Replaces existing restrictions"

#     def execute(self, context):
#         rv3d = context.region_data
#         if self.mode != "CREVICES" and rv3d is None:
#             self.report({"ERROR"}, "Only available in the 3D viewport")
#             return {"CANCELLED"}
#         if context.mode == "EDIT_MESH":
#             bpy.ops.object.mode_set(mode="OBJECT")

#         for obj in context.selected_objects:
#             if not validate_obj(self, obj):
#                 continue

#             if self.mode == "VIEW":
#                 weights = _view_weights(obj, rv3d)
#             elif self.mode == "CREVICES":
#                 weights = _rank_normalize(exposure_weights(obj))
#             else:
#                 view = _view_weights(obj, rv3d)
#                 exposure = exposure_weights(obj)
#                 weights = _rank_normalize({i: view[i] * exposure[i] for i in view})
#             _set_group_weights(obj, weights)

#         active = context.active_object
#         if active is not None and validate_obj(self, active):
#             _enter_draw_mode(context, active)
#         else:
#             context.scene.uvgami.use_weights = True
#         return {"FINISHED"}


class UVGAMI_OT_exit_draw(bpy.types.Operator):
    bl_idname = "uvgami.exit_draw"
    bl_label = "Exit"
    bl_description = "Go back to previous mode"
    bl_options = {"UNDO"}

    def execute(self, context):
        active_obj = context.active_object
        if active_obj is None:
            return {"CANCELLED"}

        if is_draw_active():
            if _old_active_group is not None:
                active_obj.vertex_groups.active_index = _old_active_group
            # addon reloads clear _old_mode
            bpy.ops.object.mode_set(mode=_old_mode or "OBJECT")
        return {"FINISHED"}


class UVGAMI_OT_clear_draw(bpy.types.Operator):
    bl_idname = "uvgami.clear_draw"
    bl_label = "Clear"
    bl_description = "Clear the painted weights"
    bl_options = {"UNDO"}

    def execute(self, context):
        for obj in context.selected_objects:
            if not validate_obj(self, obj):
                continue

            vertex_groups = obj.vertex_groups
            if SEAM_RESTRICTIONS_GROUP in vertex_groups:
                if obj.mode == "WEIGHT_PAINT":
                    group_index = vertex_groups[SEAM_RESTRICTIONS_GROUP].index
                    for vertex in obj.data.vertices:
                        for membership in vertex.groups:
                            if membership.group == group_index:
                                membership.weight = 0
                                break
                else:
                    vertex_groups.remove(vertex_groups[SEAM_RESTRICTIONS_GROUP])
        return {"FINISHED"}
