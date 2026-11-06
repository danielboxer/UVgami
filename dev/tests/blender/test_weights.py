import bpy
import numpy
import pytest
from blender_fixtures import add_sphere, addon, needs_engine, select, uv_islands

pytestmark = [needs_engine, pytest.mark.smoke]

SEAM_RESTRICTIONS_GROUP = addon.src.guides.SEAM_RESTRICTIONS_GROUP
vertex_positions = addon.src.utils.mesh.vertex_positions

OUTPUT_NAME = "sphere_unwrapped"


def paint_top_half(obj):
    group = obj.vertex_groups.new(name=SEAM_RESTRICTIONS_GROUP)
    group.add([v.index for v in obj.data.vertices if v.co.z > 0], 1.0, "REPLACE")


def unwrap_again(sphere, unwrap, outputs):
    bpy.data.objects.remove(outputs()[OUTPUT_NAME])
    sphere.hide_set(False)
    select([sphere])
    unwrap()
    return outputs()[OUTPUT_NAME]


def top_half_seams(obj):
    positions = vertex_positions(obj.data)
    return sum(
        1
        for edge in obj.data.edges
        if edge.use_seam and sum(positions[v][2] for v in edge.vertices) / 2 > 0
    )


# blender's 16 ring sphere has rings at z 0.71 and 0.56
CAP_PAINT_Z = 0.6
CAP_OUTLINE_Z = 0.56


def paint_cap(obj, *, one_sided):
    group = obj.vertex_groups.new(name=SEAM_RESTRICTIONS_GROUP)
    painted = [
        v.index
        for v in obj.data.vertices
        if v.co.z > CAP_PAINT_Z and (v.co.x >= 0 or not one_sided)
    ]
    group.add(painted, 1.0, "REPLACE")


def cap_seams(obj):
    positions = vertex_positions(obj.data)
    seams = set()
    for edge in obj.data.edges:
        ends = [positions[v] for v in edge.vertices]
        if edge.use_seam and min(end[2] for end in ends) > CAP_OUTLINE_Z - 0.01:
            seams.add(tuple(sorted(tuple(round(c, 4) for c in end) for end in ends)))
    return seams


def cap_islands(obj):
    positions = vertex_positions(obj.data)
    polygons = obj.data.polygons
    return [
        island
        for island in uv_islands(obj.data)
        if all(
            positions[v][2] > CAP_OUTLINE_Z - 0.01 for v in polygons[island[0]].vertices
        )
    ]


def test_paint_cuts_a_symmetric_cap_out_as_one_island(unwrap, outputs):
    props = bpy.context.scene.uvgami
    props.use_weights = True
    sphere = add_sphere("sphere")
    paint_cap(sphere, one_sided=False)
    select([sphere])
    unwrap()
    both_sides = outputs()[OUTPUT_NAME]
    both_sides_seams = cap_seams(both_sides)
    islands = cap_islands(both_sides)

    bpy.data.objects.remove(both_sides)
    bpy.data.objects.remove(sphere)
    sphere = add_sphere("sphere")
    paint_cap(sphere, one_sided=True)
    select([sphere])
    unwrap()

    assert len(islands) == 1
    assert all(
        min(end[2] for end in seam) < CAP_OUTLINE_Z + 0.01 for seam in both_sides_seams
    )
    assert both_sides_seams
    assert cap_seams(outputs()[OUTPUT_NAME]) == both_sides_seams


def test_paint_outline_on_a_stored_region_border_still_unwraps(unwrap, outputs):
    props = bpy.context.scene.uvgami
    props.use_weights = True
    props.split_regions = True
    sphere = add_sphere("sphere")
    paint_cap(sphere, one_sided=False)
    mesh = sphere.data
    positions = vertex_positions(mesh)
    in_cap = [
        2 * sum(positions[v][2] > CAP_PAINT_Z for v in polygon.vertices)
        >= len(polygon.vertices)
        for polygon in mesh.polygons
    ]
    faces_of_edge = {}
    for polygon in mesh.polygons:
        for key in polygon.edge_keys:
            faces_of_edge.setdefault(key, []).append(polygon.index)
    border = [
        len({in_cap[face] for face in faces_of_edge[edge.key]}) == 2
        for edge in mesh.edges
    ]
    addon.src.segmentation.write_regions(
        mesh,
        numpy.where(in_cap, 1, 2),
        numpy.array(border),
        numpy.zeros(len(border), dtype=bool),
    )
    select([sphere])
    unwrap()

    output = outputs()[OUTPUT_NAME]
    assert len(cap_islands(output)) == 1


# def group_weights(obj):
#     group = obj.vertex_groups[SEAM_RESTRICTIONS_GROUP]
#     weights = {}
#     for vertex in obj.data.vertices:
#         for entry in vertex.groups:
#             if entry.group == group.index:
#                 weights[vertex.index] = entry.weight
#     return weights


def test_avoid_seams_moves_the_seams_off_the_painted_region(unwrap, outputs):
    props = bpy.context.scene.uvgami
    # a sphere cannot flatten without stretch
    sphere = add_sphere("sphere")
    select([sphere])
    unwrap()
    plain = top_half_seams(outputs()[OUTPUT_NAME])

    paint_top_half(sphere)
    props.use_weights = True
    painted = top_half_seams(unwrap_again(sphere, unwrap, outputs))

    assert plain > 0
    assert painted == 0


# def test_crevices_weights_the_exposed_side_above_the_hole():
#     bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=0.3)
#     torus = bpy.context.active_object

#     bpy.ops.uvgami.seed_restrictions(mode="CREVICES")

#     weights = group_weights(torus)
#     outer = [
#         weights.get(v.index, 0.0) for v in torus.data.vertices if v.co.xy.length > 1
#     ]
#     inner = [
#         weights.get(v.index, 0.0) for v in torus.data.vertices if v.co.xy.length < 1
#     ]
#     assert bpy.context.scene.uvgami.use_weights
#     assert sum(outer) / len(outer) > 2 * sum(inner) / len(inner)


# def test_from_view_refuses_outside_the_3d_viewport():
#     bpy.ops.mesh.primitive_cube_add()
#     with pytest.raises(RuntimeError, match="Only available in the 3D viewport"):
#         bpy.ops.uvgami.seed_restrictions(mode="VIEW")
