import importlib
import sys
import time

import bmesh
import bpy
import mathutils
import numpy
import pytest
from blender_fixtures import (
    UNWRAP_SECONDS,
    addon,
    island_count,
    logger,
    manager,
    needs_engine,
    needs_tools,
    needs_xatlas,
    uvs_inside_unit_square,
)

segmentation = importlib.import_module(addon.src.__name__ + ".segmentation")
symmetry = importlib.import_module(addon.src.__name__ + ".symmetry")
segment_edit = addon.src.ops.segmentation
PlanRun = addon.src.ops.plan.PlanRun
RegionEditor = segment_edit.RegionEditor
SettingsCopy = addon.src.unwrap.manager.SettingsCopy
ToolsCommand = addon.src.no_bpy.engine_run.ToolsCommand
active_engine = addon.src.engines.active_engine
SURFACE_MODE_CODES = addon.src.no_bpy.SURFACE_MODE_CODES
edge_face_pairs = addon.src.no_bpy.segmentation.edge_face_pairs
pair_borders = addon.src.no_bpy.segmentation.pair_borders
loop_vertices_and_face_sizes = addon.src.utils.mesh.loop_vertices_and_face_sizes

# the dumbbell fixture's bulbs meet a straight neck of this radius around the x axis
NECK_RADIUS = 0.3
OUTPUT = "Dumbbell_unwrapped"


def axis_distance(position):
    return (position.y**2 + position.z**2) ** 0.5


def corner_positions(mesh):
    return [
        [tuple(mesh.vertices[v].co) for v in polygon.vertices]
        for polygon in mesh.polygons
    ]


def neck_ring(mesh):
    # the x of one ring on the neck between the bulbs, and its edges
    neck = [v for v in mesh.vertices if abs(axis_distance(v.co) - NECK_RADIUS) < 1e-4]
    ring_x = min(v.co.x for v in neck if abs(v.co.x) < 1.0)
    on_ring = {v.index for v in neck if abs(v.co.x - ring_x) < 1e-5}
    edges = [tuple(sorted(edge.vertices)) for edge in mesh.edges]
    return ring_x, [edge for edge in edges if set(edge) <= on_ring]


def add_triangle_at(mesh, position):
    edited = bmesh.new()
    edited.from_mesh(mesh)
    corners = [
        edited.verts.new(position + mathutils.Vector(offset))
        for offset in ((0, 0, 0), (0, 1, 0), (0, 0, 1))
    ]
    edited.faces.new(corners)
    edited.to_mesh(mesh)
    edited.free()


def test_the_split_keeps_face_order_and_the_weld_closes_it(load_obj):
    (dumbbell,) = load_obj("dumbbell")
    mesh = dumbbell.data
    # a double on the dumbbell's tip, which the weld must leave alone
    add_triangle_at(mesh, mesh.vertices[0].co.copy())
    vertex_count = len(mesh.vertices)
    before = corner_positions(mesh)
    _, ring = neck_ring(mesh)

    border_positions, parts = segmentation.split_on_borders(dumbbell, ring)

    # a closed ring has as many vertices as edges
    assert len(mesh.vertices) == vertex_count + len(ring)
    assert corner_positions(mesh) == before
    assert len(parts[0]) == len(mesh.vertices)

    segmentation.SegmentWeld(border_positions, mathutils.Matrix()).finish(dumbbell)
    assert len(mesh.vertices) == vertex_count


def test_the_proxy_trigger_counts_each_region(load_obj):
    (dumbbell,) = load_obj("dumbbell")
    _, ring = neck_ring(dumbbell.data)
    _, parts = segmentation.split_on_borders(dumbbell, ring)
    largest_region = int(parts.triangle_counts.max())

    assert len(parts.part_labels) == 2
    assert largest_region < addon.src.proxy.triangle_count(dumbbell)
    assert not addon.src.proxy.needs_proxy(dumbbell, largest_region, parts)


def test_the_proxy_keeps_each_segment_mode(load_obj, proxy_target):
    (dumbbell,) = load_obj("dumbbell")
    mesh = dumbbell.data
    ring_x, ring = neck_ring(mesh)
    _, parts = segmentation.split_on_borders(dumbbell, ring)
    centers = numpy.array([polygon.center.x for polygon in mesh.polygons])
    modes = numpy.where(
        centers < ring_x, SURFACE_MODE_CODES["ORGANIC"], SURFACE_MODE_CODES["HARD"]
    ).astype(numpy.int32)
    face_count = len(mesh.polygons)

    proxy_target(100)
    decimated = addon.src.proxy.make_proxy_with_modes(dumbbell, 100, parts, modes)

    mesh = dumbbell.data
    assert len(mesh.polygons) < face_count
    centers = numpy.array([polygon.center.x for polygon in mesh.polygons])
    assert (decimated[centers < ring_x - 1e-4] == SURFACE_MODE_CODES["ORGANIC"]).all()
    assert (decimated[centers > ring_x + 1e-4] == SURFACE_MODE_CODES["HARD"]).all()
    assert "uvgami_resolved_surface_mode" not in mesh.attributes


@needs_engine
@needs_tools
def test_the_dumbbell_splits_on_the_neck_and_welds_back(load_obj, unwrap, outputs):
    (dumbbell,) = load_obj("dumbbell")
    vertex_count = len(dumbbell.data.vertices)
    props = bpy.context.scene.uvgami
    # the split is under test, not the classifier's pick or the mirror cut
    props.optcuts.mode = "ORGANIC"
    props.symmetry = False
    unwrap()

    assert manager.error_messages == []
    assert manager.summary_failed is False, manager.summary
    assert len(manager.results) >= 2
    output = outputs()[OUTPUT]
    mesh = output.data
    assert len(mesh.vertices) == vertex_count
    assert island_count(output) >= 2
    neck_seams = [
        edge
        for edge in mesh.edges
        if edge.use_seam
        and all(
            abs(axis_distance(mesh.vertices[v].co) - NECK_RADIUS) < 1e-3
            for v in edge.vertices
        )
    ]
    assert neck_seams
    # the pack after the unwrap fits every island in the unit square
    assert uvs_inside_unit_square(mesh)


@needs_engine
@needs_tools
def test_a_mesh_with_every_region_under_the_trigger_skips_the_proxy(
    load_obj, unwrap, outputs, proxy_target
):
    (dumbbell,) = load_obj("dumbbell")
    dumbbell.modifiers.new("Subdivision", "SUBSURF")
    vertex_count = len(dumbbell.data.vertices)
    props = bpy.context.scene.uvgami
    props.optcuts.mode = "ORGANIC"
    props.symmetry = False
    props.proxy_mode = "ON"
    # the whole dumbbell is over the trigger and each region is under it
    proxy_target(addon.src.proxy.triangle_count(dumbbell) - 1)
    unwrap()

    assert manager.error_messages == []
    assert len(manager.results) >= 2
    assert all(piece.transfer_uvs_job is None for piece, _ in manager.results)
    output = outputs()[OUTPUT]
    assert len(output.data.vertices) == vertex_count
    assert [modifier.type for modifier in output.modifiers] == ["SUBSURF"]


# symmetry stays on for the uv mirror after the unwrap
@needs_xatlas
@needs_tools
def test_xatlas_splits_the_dumbbell_and_welds_back(load_obj, unwrap, outputs):
    (dumbbell,) = load_obj("dumbbell")
    vertex_count = len(dumbbell.data.vertices)
    unwrap("XATLAS")

    assert manager.error_messages == []
    assert manager.summary_failed is False, manager.summary
    assert len(manager.results) >= 2
    assert len(outputs()[OUTPUT].data.vertices) == vertex_count


@needs_engine
def test_segment_off_keeps_the_dumbbell_whole(load_obj, unwrap, outputs):
    load_obj("dumbbell")
    props = bpy.context.scene.uvgami
    props.optcuts.mode = "ORGANIC"
    props.split_regions = False
    # the dumbbell mirrors whole, and the plane seam would split it in two
    props.symmetry = False
    unwrap()

    assert manager.error_messages == []
    assert len(manager.results) == 1
    assert island_count(outputs()[OUTPUT]) == 1


@needs_engine
def test_a_hard_surface_part_is_not_segmented(load_obj, unwrap, outputs):
    (dumbbell,) = load_obj("dumbbell")
    mesh = dumbbell.data
    vertex_count = len(mesh.vertices)
    attribute = mesh.attributes.new(
        addon.src.part_overrides.SURFACE_MODE_ATTRIBUTE, "INT", "FACE"
    )
    attribute.data.foreach_set(
        "value", [SURFACE_MODE_CODES["HARD"]] * len(mesh.polygons)
    )
    unwrap()

    assert manager.error_messages == []
    assert len(manager.results) == 1
    assert len(outputs()[OUTPUT].data.vertices) == vertex_count


@needs_engine
def test_cancel_stops_a_running_segmentation(load_obj, unwrap, outputs, monkeypatch):
    load_obj("dumbbell")
    # stands in for a segmentation that runs long
    stalled = addon.src.no_bpy.engine_run.ToolsCommand(
        [sys.executable, "-c", "import time; time.sleep(120)"], None
    )
    monkeypatch.setattr(addon.src.ops.start, "tools_command", lambda: stalled)
    bpy.context.scene.uvgami.optcuts.mode = "ORGANIC"
    bpy.context.scene.uvgami.symmetry = False
    runs = addon.src.utils.paths.get_extension_dir_path() / "segment"
    pump = unwrap(until=lambda: runs.is_dir() and any(runs.iterdir()))

    manager.preparing[0].cancel()
    pump.run_until(lambda: not manager.is_active, UNWRAP_SECONDS)

    assert OUTPUT not in outputs()
    assert not any(runs.iterdir())
    assert not any("segmentation failed" in line for line in logger.get_latest().errors)


@needs_engine
def test_a_missing_segment_binary_stops_the_unwrap(
    load_obj, unwrap, outputs, monkeypatch, tmp_path
):
    load_obj("dumbbell")
    monkeypatch.setattr(
        addon.src.utils.uvgami_tools, "get_dev_build_path", lambda name: None
    )
    monkeypatch.setattr(addon.src.utils.uvgami_tools, "get_dir_path", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="UVgami tools"):
        unwrap()

    assert outputs() == {}
    assert manager.results == []


def test_stored_borders_drive_the_segment_work_without_the_engine(load_obj):
    (dumbbell,) = load_obj("dumbbell")
    mesh = dumbbell.data
    centers = numpy.array([polygon.center.x for polygon in mesh.polygons])
    fine = numpy.where(centers < 0, 1, 2).astype(numpy.int64)
    corners, totals = addon.src.utils.mesh.loop_vertices_and_face_sizes(mesh)
    border = pair_borders(edge_face_pairs(corners, totals), fine)[(1, 2)]
    border_edges = sorted(map(tuple, border.tolist()))
    flags = numpy.array(
        [tuple(sorted(edge.vertices)) in set(border_edges) for edge in mesh.edges]
    )
    segmentation.write_regions(mesh, fine, flags, numpy.zeros(len(flags), dtype=bool))

    stored = segmentation.stored_regions(mesh)
    assert stored.fine_labels.tolist() == fine.tolist()
    assert sorted(map(tuple, stored.border_edges.tolist())) == border_edges

    missing = ToolsCommand(["no-such-segment-engine.exe"], None)
    positions = addon.src.utils.mesh.vertex_position_array(mesh)
    result = segmentation.segment_work(
        mesh, positions, missing, "ORGANIC", fine_cut=False, material_seams=False
    ).result()
    assert result.hand_edited
    assert result.region_count == 2
    assert sorted(result.border_edges) == border_edges

    segmentation.clear_regions(mesh)
    assert segmentation.stored_regions(mesh) is None


def dumbbell_segments(dumbbell):
    props = bpy.context.scene.uvgami
    props.optcuts.mode = "ORGANIC"
    props.symmetry = False
    engine = active_engine(props.engine)
    run = PlanRun(dumbbell, engine, SettingsCopy(props))
    while not run.done():
        time.sleep(0.05)
    segments = run.result().segments
    run.close()
    return segments


@needs_engine
@needs_tools
def test_the_region_editor_joins_the_bulbs_and_the_unwrap_keeps_it_whole(
    load_obj, unwrap, outputs
):
    (dumbbell,) = load_obj("dumbbell")
    segments = dumbbell_segments(dumbbell)
    assert segments.region_count >= 2

    session = RegionEditor(dumbbell, segments, {})
    assert session.part_count == segments.region_count
    assert len(session.on) >= 1
    # every fine region is one of the engine's regions or inside one
    assert len(numpy.unique(session.fine)) >= segments.region_count
    for pair in list(session.on):
        session.toggle(pair)
    assert session.part_count == 1
    assert len(segmentation.stored_regions(dumbbell.data).border_edges) == 0

    unwrap()

    assert manager.error_messages == []
    assert len(manager.results) == 1
    assert island_count(outputs()[OUTPUT]) == 1


@needs_tools
def test_material_seams_split_the_regions_on_the_material_borders(load_obj):
    (dumbbell,) = load_obj("dumbbell")
    mesh = dumbbell.data
    for name in ("upper", "lower"):
        mesh.materials.append(bpy.data.materials.new(name))
    centers = numpy.empty(len(mesh.polygons) * 3)
    mesh.polygons.foreach_get("center", centers)
    # y above zero cuts both bulbs and the neck lengthwise
    materials = (centers.reshape(-1, 3)[:, 1] > 0).astype(numpy.int64)
    mesh.polygons.foreach_set("material_index", materials.tolist())
    plain = dumbbell_segments(dumbbell)

    bpy.context.scene.uvgami.material_seams = True
    segments = dumbbell_segments(dumbbell)

    assert segments.region_count > plain.region_count
    labels = segments.labels
    for region in numpy.unique(labels[labels > 0]).tolist():
        assert len(numpy.unique(materials[labels == region])) == 1
    pairs = edge_face_pairs(*loop_vertices_and_face_sizes(mesh))
    on_border = (materials[pairs.first] != materials[pairs.second]) & (
        labels[pairs.first] > 0
    )
    expected = set(map(tuple, pairs.edges[on_border].tolist()))
    assert expected <= set(segments.border_edges)


def hovered_loops(session):
    return (
        session._loops_through(
            session.regions_of_part[session.joined[face]], *sorted(corners[:2])
        )
        for face, corners in enumerate(session.face_corners)
    )


@needs_tools
def test_a_clicked_loop_splits_its_region_and_a_second_click_removes_it(load_obj):
    (dumbbell,) = load_obj("dumbbell")
    session = RegionEditor(dumbbell, dumbbell_segments(dumbbell), {})
    parts = session.part_count
    (candidate,) = next(loops for loops in hovered_loops(session) if loops)

    session.add_loops((candidate,))
    after_adding = session.part_count
    stored_loop = segmentation.stored_loop_edges(dumbbell.data)
    session.toggle(
        next(pair for pair in session.borders if session._is_loop_border(pair))
    )
    after_removal = session.part_count

    assert len(stored_loop) == len(candidate.edges)
    assert after_adding == parts + 1
    assert after_removal == parts
    assert len(segmentation.stored_loop_edges(dumbbell.data)) == 0
    assert segmentation.stored_regions(dumbbell.data) is not None


def regions_split(session, candidate):
    far_side = numpy.zeros(len(session.fine), dtype=bool)
    far_side[candidate.relabeled_faces] = True
    return numpy.intersect1d(session.fine[far_side], session.fine[~far_side])


@needs_tools
def test_a_loop_across_joined_regions_splits_their_part_until_clicked_again(
    load_obj,
):
    (dumbbell,) = load_obj("dumbbell")
    session = RegionEditor(dumbbell, dumbbell_segments(dumbbell), {})
    mesh = dumbbell.data
    centers = numpy.empty(len(mesh.polygons) * 3)
    mesh.polygons.foreach_get("center", centers)
    # y above zero halves every region lengthwise, so the rings cross two regions
    halves = (centers.reshape(-1, 3)[:, 1] > 0).astype(numpy.int64)
    session._set_fine(numpy.where(session.fine > 0, session.fine * 2 - halves, 0))
    session.on = set()
    session._join_parts()
    (candidate,) = next(
        loops
        for loops in hovered_loops(session)
        if loops and len(regions_split(session, loops[0])) > 1
    )

    session.add_loops((candidate,))
    near_side = numpy.ones(len(session.fine), dtype=bool)
    near_side[candidate.relabeled_faces] = False
    far_parts = numpy.unique(session.joined[candidate.relabeled_faces])
    near_parts = numpy.unique(session.joined[near_side & (session.fine > 0)])
    stored_loop = segmentation.stored_loop_edges(dumbbell.data)
    after_adding = session.part_count
    loop_borders = [pair for pair in session.borders if session._is_loop_border(pair)]
    session.toggle(loop_borders[0])

    assert after_adding == 2
    assert len(far_parts) == 1
    assert len(near_parts) == 1
    assert len(stored_loop) == len(candidate.edges)
    assert len(loop_borders) > 1
    assert session.part_count == 1
    assert len(segmentation.stored_loop_edges(dumbbell.data)) == 0


@needs_tools
def test_the_mirrored_border_and_loop_change_with_the_clicked_one(load_obj):
    (dumbbell,) = load_obj("dumbbell")
    segments = dumbbell_segments(dumbbell)
    center, _ = symmetry.bounding_box_center_and_diagonal(dumbbell.data)
    partners = symmetry.mesh_mirror_partners(dumbbell.data, center, "X")
    session = RegionEditor(dumbbell, segments, partners)
    pair = next(pair for pair in session.borders if session._mirrored_border(pair))
    mirrored = session._mirrored_border(pair)
    was_on = pair in session.on

    session.toggle(pair)
    toggled = (pair in session.on, mirrored in session.on)
    parts = session.part_count
    loops = next(loops for loops in hovered_loops(session) if len(loops) == 2)
    session.add_loops(loops)
    after_adding = session.part_count

    assert toggled == (not was_on, not was_on)
    assert after_adding == parts + 2


@needs_tools
def test_a_region_draws_its_quad_loops_and_the_nearest_one_is_picked(load_obj):
    (dumbbell,) = load_obj("dumbbell")
    session = RegionEditor(dumbbell, dumbbell_segments(dumbbell), {})
    part = next(
        part for part in session.regions_of_part[1:] if session._drawn_loops(part)
    )
    drawn = session._drawn_loops(part)
    tail, head = drawn.edges[0].tolist()
    middle = (session.positions[tail] + session.positions[head]) / 2

    nearest = session._nearest_drawn_loop(drawn, None, None, middle)
    (candidate,) = session._loop_and_mirror(drawn.loops[nearest.index])

    first_loop = drawn.edges[drawn.loop_of_edge == drawn.loop_of_edge[0], 0]
    assert nearest.index == drawn.loop_of_edge[0]
    assert numpy.allclose(nearest.closest, middle)
    assert sorted(candidate.vertices) == sorted(first_loop.tolist())
