import importlib
import time
from types import SimpleNamespace

import bmesh
import bpy
import numpy
import pytest
from blender_fixtures import (
    addon,
    cylinder_with_loose_cube,
    is_second_cube,
    needs_partuv,
    needs_tools,
    needs_xatlas,
    two_loose_cubes,
)

plan = addon.src.ops.plan
plan_drawing = addon.src.ops.plan_drawing
part_overrides = addon.src.part_overrides
segmentation = importlib.import_module(addon.src.__name__ + ".segmentation")
SettingsCopy = addon.src.unwrap.manager.SettingsCopy
active_engine = addon.src.engines.active_engine
SURFACE_MODE_CODES = addon.src.no_bpy.SURFACE_MODE_CODES
SURFACE_MODE_ATTRIBUTE = part_overrides.SURFACE_MODE_ATTRIBUTE


def planned(obj):
    props = bpy.context.scene.uvgami
    engine = active_engine(props.engine)
    settings = SettingsCopy(props)
    run = plan.PlanRun(obj, engine, settings)
    try:
        while not run.done():
            time.sleep(0.05)
        return plan.PlanSession(obj, engine, settings, run.result())
    finally:
        run.close()


def shape_colors(session):
    return {shape.color for shape in [*session.fill_shapes, *session.plane_shapes]}


@needs_tools
def test_plan_counts_the_regions_and_undoes_a_border_click(load_obj):
    (dumbbell,) = load_obj("dumbbell")
    bpy.context.scene.uvgami.optcuts.mode = "ORGANIC"
    # the dumbbell mirrors whole, which skips the region split
    bpy.context.scene.uvgami.symmetry = False
    session = planned(dumbbell)
    regions = session.regions
    pair = next(iter(regions.on))
    lines_before = session.lines
    colors_before = shape_colors(session)

    regions.hovered = pair
    session.click(SimpleNamespace(shift=False))
    toggled_off = pair not in regions.on
    stored = segmentation.stored_regions(dumbbell.data).border_edges
    session.undo()

    assert plan.REGION_COLORS[0] in colors_before
    assert any(line.endswith(" regions") for line in lines_before)
    assert any(line.endswith(" regions (edited)") for line in session.lines)
    assert toggled_off
    assert pair in regions.on
    restored = segmentation.stored_regions(dumbbell.data).border_edges
    assert len(restored) > len(stored)


def test_plan_draws_the_symmetry_plane():
    # quad sides mirror across X, the cylinder fixture's triangles don't
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, depth=3, end_fill_type="TRIFAN")
    bpy.context.scene.uvgami.split_regions = False

    session = planned(bpy.context.active_object)

    assert any(line.startswith("Symmetrical on") for line in session.lines)
    assert (*plan_drawing.AXIS_COLORS["X"], 0.9) in shape_colors(session)


def test_plan_counts_the_parts_left_out_of_the_mirror():
    obj = cylinder_with_loose_cube()
    bpy.context.scene.uvgami.split_regions = False

    session = planned(obj)

    assert "Symmetrical on X, 1 part not mirrored" in session.lines


def test_plan_colors_a_beveled_cube_hard_surface(load_obj):
    (cube,) = load_obj("cube-bevel2")
    bpy.context.scene.uvgami.split_regions = False

    session = planned(cube)

    assert plan.HARD_COLOR in shape_colors(session)
    assert not shape_colors(session) & set(plan.REGION_COLORS)
    assert "1 hard surface part" in session.lines


def test_a_picked_part_mode_recolors_writes_and_undoes(load_obj):
    (cube,) = load_obj("cube-bevel2")
    bpy.context.scene.uvgami.split_regions = False
    session = planned(cube)

    session.pick(0, extend=False)
    session.write_picked(SURFACE_MODE_ATTRIBUTE, SURFACE_MODE_CODES["ORGANIC"])
    written = part_overrides.face_values(cube.data, SURFACE_MODE_ATTRIBUTE)
    organic_colors = shape_colors(session)
    session.undo()

    assert (written == SURFACE_MODE_CODES["ORGANIC"]).all()
    assert plan.REGION_COLORS[0] in organic_colors
    assert plan.HARD_COLOR not in organic_colors
    assert not part_overrides.face_values(cube.data, SURFACE_MODE_ATTRIBUTE).any()
    assert plan.HARD_COLOR in shape_colors(session)


def test_global_after_a_stored_override_shows_the_auto_mode(load_obj):
    (cube,) = load_obj("cube-bevel2")
    bpy.context.scene.uvgami.split_regions = False
    attribute = cube.data.attributes.new(SURFACE_MODE_ATTRIBUTE, "INT", "FACE")
    organic = numpy.full(len(cube.data.polygons), SURFACE_MODE_CODES["ORGANIC"])
    attribute.data.foreach_set("value", organic.astype(numpy.int32))
    session = planned(cube)
    overridden_colors = shape_colors(session)

    session.pick(0, extend=False)
    session.write_picked(SURFACE_MODE_ATTRIBUTE, 0)

    assert plan.HARD_COLOR not in overridden_colors
    assert plan.HARD_COLOR in shape_colors(session)
    assert "1 hard surface part" in session.lines


def test_a_picked_part_writes_every_face_of_that_part():
    joined = two_loose_cubes()
    session = planned(joined)
    second = next(
        int(session.face_loose_part[face.index])
        for face in joined.data.polygons
        if is_second_cube(joined.data, face)
    )

    session.pick(second, extend=False)
    session.write_picked(SURFACE_MODE_ATTRIBUTE, SURFACE_MODE_CODES["HARD"])

    mesh = joined.data
    values = part_overrides.face_values(mesh, SURFACE_MODE_ATTRIBUTE)
    expected = [
        SURFACE_MODE_CODES["HARD"] if is_second_cube(mesh, face) else 0
        for face in mesh.polygons
    ]
    assert values.tolist() == expected
    assert session.picked_overrides().surface_mode == "HARD"


def dense_cube(cuts):
    obj = bpy.data.objects.new("cube", bpy.data.meshes.new("cube"))
    bpy.context.scene.collection.objects.link(obj)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=cuts, use_grid_fill=True)
    bm.to_mesh(obj.data)
    bm.free()
    return obj


def dense_monkey():
    bpy.ops.mesh.primitive_monkey_add()
    obj = bpy.context.active_object
    modifier = obj.modifiers.new("subdivision", "SUBSURF")
    modifier.levels = 3
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    return obj


@needs_tools
@pytest.mark.parametrize(
    ("engine", "cuts"),
    [
        pytest.param("XATLAS", 65, marks=needs_xatlas),
        pytest.param("PARTUV", 41, marks=needs_partuv),
    ],
)
def test_plan_leaves_a_dense_hard_part_out_of_the_auto_proxy(engine, cuts):
    props = bpy.context.scene.uvgami
    props.engine = engine
    props.proxy_mode = "AUTO"
    props.split_regions = False
    props.symmetry = False

    cube_lines = planned(dense_cube(cuts)).lines
    monkey_lines = planned(dense_monkey()).lines

    assert "0 proxy parts" in cube_lines
    assert "1 proxy part" in monkey_lines
