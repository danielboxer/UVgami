import math

from mathutils import Vector
from mathutils.bvhtree import BVHTree

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
