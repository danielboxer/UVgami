import bpy
import pytest
from blender_fixtures import add_cube, manager, needs_engine

pytestmark = [needs_engine, pytest.mark.smoke]

DIGITS = 3
UNWRAPPED_COLLECTION = "UVgami Unwrapped"


def world_positions(obj, vertices=None):
    return {
        tuple(round(c, DIGITS) for c in obj.matrix_world @ v.co)
        for v in (obj.data.vertices if vertices is None else vertices)
    }


def world_normal_key(obj, face):
    normal_matrix = obj.matrix_world.to_3x3().inverted().transposed()
    return tuple(round(c) for c in (normal_matrix @ face.normal).normalized())


def by_world_normal(obj, attribute):
    return {
        world_normal_key(obj, face): getattr(face, attribute)
        for face in obj.data.polygons
    }


def test_slots_shading_and_transform_survive(unwrap, outputs):
    parent = bpy.data.objects.new("parent", None)
    bpy.context.collection.objects.link(parent)
    parent.delta_location = (0, 5, 0)
    cube = add_cube("cube", location=(3, 0, 0))
    cube.scale = (-1, 1, 1)
    cube.parent = parent
    cube.data.materials.append(bpy.data.materials.new("A"))
    cube.data.materials.append(None)
    cube.data.materials.append(bpy.data.materials.new("B"))
    for face in cube.data.polygons:
        x = round(face.normal.x)
        face.material_index = {1: 0, -1: 1}.get(x, 2)
        face.use_smooth = face.normal.z > 0
    bpy.context.view_layer.update()
    expected_positions = world_positions(cube)
    expected_material = by_world_normal(cube, "material_index")
    expected_smooth = by_world_normal(cube, "use_smooth")

    unwrap()

    assert manager.summary_failed is False, manager.summary
    output = outputs()["cube_unwrapped"]
    slots = [
        slot.material.name if slot.material else None for slot in output.material_slots
    ]
    assert slots == ["A", None, "B"]
    assert world_positions(output) == expected_positions
    assert by_world_normal(output, "material_index") == expected_material
    assert by_world_normal(output, "use_smooth") == expected_smooth


def test_unicode_name_output_is_named_hidden_and_collected(unwrap, outputs):
    name = "Ürün 模型"
    cube = add_cube(name)

    unwrap()

    assert manager.summary_failed is False, manager.summary
    assert list(outputs()) == [f"{name}_unwrapped"]
    assert cube.hide_get()
    output = outputs()[f"{name}_unwrapped"]
    assert [c.name for c in output.users_collection] == [UNWRAPPED_COLLECTION]


def test_far_small_mesh_keeps_its_place(unwrap, outputs):
    cube = add_cube("far", location=(1000, -2000, 3000), scale=(0.01, 0.01, 0.01))
    bpy.context.view_layer.update()
    expected_positions = world_positions(cube)

    unwrap()

    assert manager.summary_failed is False, manager.summary
    assert world_positions(outputs()["far_unwrapped"]) == expected_positions


def test_vertex_group_comes_back(unwrap, outputs):
    cube = add_cube("grip")
    top = [v for v in cube.data.vertices if v.co.z > 0]
    group = cube.vertex_groups.new(name="top")
    group.add([v.index for v in top], 0.75, "REPLACE")
    expected_positions = world_positions(cube, top)

    unwrap()

    assert manager.summary_failed is False, manager.summary
    output = outputs()["grip_unwrapped"]
    restored = output.vertex_groups.get("top")
    assert restored is not None
    grouped = [
        v
        for v in output.data.vertices
        if any(g.group == restored.index for g in v.groups)
    ]
    # a seam splits a vertex, so the group is compared by position
    assert world_positions(output, grouped) == expected_positions
    weights = [g.weight for v in grouped for g in v.groups if g.group == restored.index]
    assert all(w == 0.75 for w in weights)
