import sys

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
    needs_segment,
    needs_xatlas,
)

segmentation = addon.src.segmentation
# segment_edit = addon.src.ops.segment_edit
# auto_preview = addon.src.ops.auto_preview
# merge_region = addon.src.seams.segmentation.merge_region
SURFACE_MODE_CODES = addon.src.seams.SURFACE_MODE_CODES

# the dumbbell fixture's bulbs meet a straight neck of this radius around the x axis
NECK_RADIUS = 0.3
OUTPUT = "Dumbbell_unwrapped"


def axis_distance(co):
    return (co.y**2 + co.z**2) ** 0.5


def corner_positions(mesh):
    return [
        [tuple(mesh.vertices[v].co) for v in polygon.vertices]
        for polygon in mesh.polygons
    ]


# the x of one ring on the neck between the bulbs, and its edges
def neck_ring(mesh):
    neck = [v for v in mesh.vertices if abs(axis_distance(v.co) - NECK_RADIUS) < 1e-4]
    ring_x = min(v.co.x for v in neck if abs(v.co.x) < 1.0)
    on_ring = {v.index for v in neck if abs(v.co.x - ring_x) < 1e-5}
    edges = [tuple(sorted(e.vertices)) for e in mesh.edges]
    return ring_x, [edge for edge in edges if set(edge) <= on_ring]


def add_triangle_at(mesh, position):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    corners = [
        bm.verts.new(position + mathutils.Vector(offset))
        for offset in ((0, 0, 0), (0, 1, 0), (0, 0, 1))
    ]
    bm.faces.new(corners)
    bm.to_mesh(mesh)
    bm.free()


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


def test_the_proxy_keeps_each_segment_mode(load_obj):
    (dumbbell,) = load_obj("dumbbell")
    mesh = dumbbell.data
    ring_x, ring = neck_ring(mesh)
    _, parts = segmentation.split_on_borders(dumbbell, ring)
    centers = numpy.array([polygon.center.x for polygon in mesh.polygons])
    modes = numpy.where(
        centers < ring_x, SURFACE_MODE_CODES["ORGANIC"], SURFACE_MODE_CODES["HARD"]
    ).astype(numpy.int32)
    face_count = len(mesh.polygons)

    decimated = segmentation.make_segmented_proxy(dumbbell, 100, parts, modes)

    mesh = dumbbell.data
    assert len(mesh.polygons) < face_count
    centers = numpy.array([polygon.center.x for polygon in mesh.polygons])
    assert (decimated[centers < ring_x - 1e-4] == SURFACE_MODE_CODES["ORGANIC"]).all()
    assert (decimated[centers > ring_x + 1e-4] == SURFACE_MODE_CODES["HARD"]).all()
    assert "uvgami_resolved_surface_mode" not in mesh.attributes


def uvs(mesh):
    coords = numpy.empty(len(mesh.loops) * 2)
    mesh.uv_layers.active.data.foreach_get("uv", coords)
    return coords


@needs_engine
@needs_segment
def test_the_dumbbell_splits_on_the_neck_and_welds_back(load_obj, unwrap, outputs):
    (dumbbell,) = load_obj("dumbbell")
    vertex_count = len(dumbbell.data.vertices)
    props = bpy.context.scene.uvgami
    # the split is under test, not the classifier's pick
    props.optcuts.mode = "ORGANIC"
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
    coords = uvs(mesh)
    assert coords.min() >= 0.0
    assert coords.max() <= 1.0


@needs_xatlas
@needs_segment
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
    stalled = [sys.executable, "-c", "import time; time.sleep(120)"]
    monkeypatch.setattr(
        addon.src.ops.start, "resolve_bundled_command", lambda name, label: stalled
    )
    bpy.context.scene.uvgami.optcuts.mode = "ORGANIC"
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
        addon.src.engines.bundled, "get_local_engine_path", lambda name: None
    )
    monkeypatch.setattr(addon.src.engines.bundled, "get_dir_path", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="Segmentation"):
        unwrap()

    assert outputs() == {}
    assert manager.results == []


# def test_stored_labels_drive_the_segment_work_without_the_engine(load_obj):
#     (dumbbell,) = load_obj("dumbbell")
#     mesh = dumbbell.data
#     centers = numpy.array([polygon.center.x for polygon in mesh.polygons])
#     labels = numpy.where(centers < 0, 1, 2).astype(numpy.int64)
#     segmentation.write_region_labels(mesh, labels)

#     result = segmentation.segment_work(
#         mesh, ["no-such-segment-engine.exe"], "ORGANIC"
#     ).result()

#     assert result.labels.tolist() == labels.tolist()
#     assert result.region_count == 2
#     # the border is one ring of the neck
#     border_x = {
#         round(mesh.vertices[v].co.x, 5) for edge in result.border_edges for v in edge
#     }
#     assert len(border_x) == 1

#     segmentation.clear_region_labels(mesh)
#     assert segmentation.REGION_ATTRIBUTE not in mesh.attributes
#     assert not segmentation.region_labels(mesh).any()


# def test_the_region_editor_draws_each_region_and_writes_its_labels(load_obj):
#     (dumbbell,) = load_obj("dumbbell")
#     mesh = dumbbell.data
#     session = segment_edit.EditSession(dumbbell, ["no-such-segment-engine.exe"])
#     assert not session.labels.any()
#     centers = numpy.array([polygon.center.x for polygon in mesh.polygons])
#     labels = numpy.where(centers < 0, 1, 2).astype(numpy.int64)

#     session.set_labels(labels)

#     preview = auto_preview._previews[dumbbell.name]
#     assert "2 regions" in preview.summary
#     assert [shape.kind for shape in preview.shapes] == ["TRIS", "TRIS", "LINES"]
#     assert segmentation.region_labels(mesh).tolist() == labels.tolist()

#     session.set_labels(merge_region(session.labels, session.face_pairs, 1))

#     assert segmentation.region_labels(mesh).tolist() == [2] * len(labels)
#     assert [shape.kind for shape in auto_preview._previews[dumbbell.name].shapes] == [
#         "TRIS"
#     ]
#     auto_preview.stop_auto_preview()
