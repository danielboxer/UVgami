import bpy
import pytest
from blender_fixtures import add_sphere, addon, needs_engine, select

pytestmark = [needs_engine, pytest.mark.smoke]

SEAM_RESTRICTIONS_GROUP = addon.src.ops.guides.SEAM_RESTRICTIONS_GROUP
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
