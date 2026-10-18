import bmesh
import bpy
import pytest
from blender_fixtures import (
    addon,
    island_count,
    manager,
    needs_engine,
    seam_count,
)
from test_settings import EXPECTED, two_loose_cubes

pytestmark = needs_engine

part_overrides = addon.src.part_overrides
SURFACE_MODE_CODES = part_overrides.SURFACE_MODE_CODES


# the second cube of two_loose_cubes sits past x = 2
def is_second_cube(mesh, face):
    return mesh.vertices[face.vertices[0]].co.x > 2


def set_override(obj, name, value, faces=None):
    mesh = obj.data
    attribute = mesh.attributes.new(
        name, part_overrides.OVERRIDE_ATTRIBUTES[name], "FACE"
    )
    for face in mesh.polygons:
        if faces is None or faces(mesh, face):
            attribute.data[face.index].value = value


def side_uvs(obj, second):
    mesh = obj.data
    layer = mesh.uv_layers.active.data
    return {
        tuple(round(c, 5) for c in layer[loop].uv)
        for face in mesh.polygons
        if is_second_cube(mesh, face) == second
        for loop in face.loop_indices
    }


# def side_uv_area(obj, second):
#     mesh = obj.data
#     layer = mesh.uv_layers.active.data
#     total = 0
#     for face in mesh.polygons:
#         if is_second_cube(mesh, face) != second:
#             continue
#         corners = [layer[loop].uv for loop in face.loop_indices]
#         total += abs(
#             sum(
#                 a.x * b.y - b.x * a.y
#                 for a, b in zip(corners, corners[1:] + corners[:1])
#             )
#         )
#     return total / 2


def test_kept_part_skips_the_engine_and_keeps_its_uvs(unwrap, outputs):
    joined = two_loose_cubes()
    bpy.context.scene.uvgami.pack_after_unwrap = False
    set_override(joined, part_overrides.KEEP_ATTRIBUTE, 1, is_second_cube)
    kept_before = side_uvs(joined, second=True)

    unwrap()

    assert manager.summary == ["UV unwrap complete"]
    assert [u.keeps_uvs for u, _ in manager.results].count(True) == 1
    output = outputs()[f"{joined.name}_unwrapped"]
    assert side_uvs(output, second=True) == kept_before
    assert side_uvs(output, second=False) != side_uvs(joined, second=False)


def test_kept_part_without_uvs_is_not_unwrapped(unwrap, invalid_objects):
    joined = two_loose_cubes()
    joined.data.uv_layers.remove(joined.data.uv_layers[0])
    set_override(joined, part_overrides.KEEP_ATTRIBUTE, 1, is_second_cube)

    unwrap()

    assert len(invalid_objects()) == 1


@pytest.mark.parametrize(
    "panel_mode, part_mode", [("ORGANIC", "HARD"), ("HARD", "ORGANIC")]
)
def test_part_mode_overrides_the_panel_mode(
    load_obj, unwrap, outputs, panel_mode, part_mode
):
    (obj,) = load_obj("cube-bevel2")
    bpy.context.scene.uvgami.optcuts.mode = panel_mode
    set_override(
        obj, part_overrides.SURFACE_MODE_ATTRIBUTE, SURFACE_MODE_CODES[part_mode]
    )

    unwrap()

    (output,) = outputs().values()
    expected = EXPECTED["cube-bevel2", part_mode]
    assert (island_count(output), seam_count(output)) == expected


# def test_part_priority_overrides_the_panel_priority(unwrap, outputs):
#     joined = two_loose_cubes()
#     bpy.context.scene.uvgami.priority = "FEWER_SEAMS"
#     code = part_overrides.PRIORITY_ORDER.index("LESS_STRETCH") + 1
#     set_override(joined, part_overrides.PRIORITY_ATTRIBUTE, code, is_second_cube)

#     unwrap()

#     priorities = sorted(manager.piece_props(u).priority for u, _ in manager.results)
#     assert priorities == ["FEWER_SEAMS", "LESS_STRETCH"]


# def test_uv_scale_override_survives_the_pack(unwrap, outputs):
#     joined = two_loose_cubes()
#     props = bpy.context.scene.uvgami
#     props.pack_after_unwrap = True
#     props.fix_scale = True
#     set_override(joined, part_overrides.UV_SCALE_ATTRIBUTE, 2.0, is_second_cube)

#     unwrap()

#     output = outputs()[f"{joined.name}_unwrapped"]
#     ratio = side_uv_area(output, second=True) / side_uv_area(output, second=False)
#     assert ratio == pytest.approx(4, rel=0.01)


def test_panel_writes_the_whole_part_of_a_selected_face():
    joined = two_loose_cubes()
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(joined.data)
    for face in bm.faces:
        face.select_set(False)
    face = next(f for f in bm.faces if f.verts[0].co.x > 2)
    face.select_set(True)
    bm.faces.active = face
    bmesh.update_edit_mesh(joined.data)

    overrides = bpy.context.scene.uvgami_part_overrides
    overrides.surface_mode = "HARD"
    assert overrides.surface_mode == "HARD"
    bpy.ops.object.mode_set(mode="OBJECT")

    mesh = joined.data
    values = part_overrides.face_values(mesh, part_overrides.SURFACE_MODE_ATTRIBUTE)
    expected = [
        SURFACE_MODE_CODES["HARD"] if is_second_cube(mesh, face) else 0
        for face in mesh.polygons
    ]
    assert values.tolist() == expected
