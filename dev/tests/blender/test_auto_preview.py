import bpy
import pytest
from blender_fixtures import addon, needs_tools

auto_preview = addon.src.ops.auto_preview


@pytest.fixture
def preview(session):
    def run():
        session(
            bpy.ops.uvgami.preview_auto_settings,
            until=lambda: auto_preview._run is None,
        )
        return auto_preview._preview

    yield run
    auto_preview.stop_auto_preview()


def shape_colors(result):
    return {shape.color for shape in result.shapes}


@needs_tools
def test_preview_draws_the_region_borders(load_obj, preview):
    load_obj("dumbbell")
    bpy.context.scene.uvgami.optcuts.mode = "ORGANIC"
    # the dumbbell mirrors whole, which skips the region split
    bpy.context.scene.uvgami.symmetry = False

    result = preview()

    assert any(line.endswith(" regions") for line in result.lines)
    assert auto_preview.BORDER_COLOR in shape_colors(result)


def test_preview_draws_the_symmetry_plane(load_obj, preview):
    load_obj("cylinder")
    bpy.context.scene.uvgami.split_regions = False

    result = preview()

    assert any(line.startswith("Symmetrical on") for line in result.lines)
    assert (*auto_preview.AXIS_COLORS["X"], 0.9) in shape_colors(result)


def test_preview_colors_a_beveled_cube_hard_surface(load_obj, preview):
    load_obj("cube-bevel2")
    bpy.context.scene.uvgami.split_regions = False

    result = preview()

    assert auto_preview.HARD_COLOR in shape_colors(result)
    assert auto_preview.ORGANIC_COLOR not in shape_colors(result)
    assert "1 part: 0 organic, 1 hard surface" in result.lines


def test_preview_in_edit_mode_shows_an_unsaved_part_mode(load_obj, preview):
    load_obj("cube-bevel2")
    bpy.context.scene.uvgami.split_regions = False
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.context.scene.uvgami_part_overrides.surface_mode = "ORGANIC"

    result = preview()
    bpy.ops.object.mode_set(mode="OBJECT")

    assert auto_preview.ORGANIC_COLOR in shape_colors(result)
    assert auto_preview.HARD_COLOR not in shape_colors(result)
