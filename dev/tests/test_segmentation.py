import collections
import sys

import numpy
import pytest
from uvgami_cli._pytest_plugin import corners_and_totals

from no_bpy import SURFACE_MODE_CODES, EngineError
from no_bpy.engine_run import ToolsCommand
from no_bpy.segmentation import (
    # split_loop_job,
    # split_region_job,
    edge_face_pairs,
    joined_labels,
    nested_fine_labels,
    on_pairs_with_edges,
    pair_borders,
    pair_edges,
    pairs_between_regions,
    segment_job,
    stored_segment_job,
)
from no_bpy.worker import WorkerProcess

# the dumbbell fixture's bulbs meet a straight neck of this radius around the x axis
NECK_RADIUS = 0.3

SegmentInputs = collections.namedtuple(
    "SegmentInputs", "positions corners totals surface_modes kept"
)


def mesh_inputs(parts):
    # each part is (verts, faces, mode name, kept)
    verts, faces, modes, kept = [], [], [], []
    for part_verts, part_faces, mode, part_kept in parts:
        base = len(verts)
        verts += part_verts
        faces += [[base + v for v in face] for face in part_faces]
        modes += [SURFACE_MODE_CODES[mode]] * len(part_faces)
        kept += [part_kept] * len(part_faces)
    return SegmentInputs(
        numpy.array(verts, dtype=numpy.float64),
        *corners_and_totals(faces),
        numpy.array(modes, dtype=numpy.int32),
        numpy.array(kept, dtype=bool),
    )


def cube_at(unit_cube, y):
    verts, faces = unit_cube
    return [[x, cy + y, z] for x, cy, z in verts], faces


@pytest.mark.smoke
def test_dumbbell_splits_on_the_neck_and_skips_hard_and_kept_parts(
    tmp_path, tools, fixture_mesh, unit_cube
):
    dumbbell_verts, dumbbell_faces = fixture_mesh("dumbbell")
    inputs = mesh_inputs(
        [
            (dumbbell_verts, dumbbell_faces, "ORGANIC", False),
            (*cube_at(unit_cube, 5.0), "HARD", False),
            (*cube_at(unit_cube, 8.0), "ORGANIC", True),
            (*cube_at(unit_cube, 11.0), "AUTO", False),
        ]
    )
    positions = inputs.positions

    result = WorkerProcess(
        segment_job, tools, tmp_path, *inputs, fine_cut=False
    ).result()

    assert SURFACE_MODE_CODES["AUTO"] not in result.surface_modes
    assert result.fine_labels is None
    assert result.region_count >= 2
    assert result.border_edges
    border_vertices = numpy.unique(numpy.array(result.border_edges))
    # every vertex past the dumbbell's belongs to the hard, kept or auto cube
    assert border_vertices.max() < len(dumbbell_verts)
    radii = numpy.linalg.norm(positions[border_vertices][:, 1:], axis=1)
    assert numpy.allclose(radii, NECK_RADIUS, atol=1e-4)


def test_the_fine_cut_splits_inside_the_regions(tmp_path, tools, fixture_mesh):
    verts, faces = fixture_mesh("dumbbell")
    inputs = mesh_inputs([(verts, faces, "ORGANIC", False)])
    result = segment_job(tools, tmp_path, *inputs, fine_cut=True)
    assert result.fine_labels.min() >= 1
    region_of_fine = {}
    for fine, region in zip(result.fine_labels.tolist(), result.labels.tolist()):
        assert region_of_fine.setdefault(fine, region) == region
    assert len(region_of_fine) >= result.region_count


def test_a_missing_engine_raises(tmp_path, fixture_mesh):
    verts, faces = fixture_mesh("dumbbell")
    inputs = mesh_inputs([(verts, faces, "ORGANIC", False)])
    with pytest.raises(EngineError, match="failed to start"):
        segment_job(
            ToolsCommand(["no-such-segment-engine.exe"], None),
            tmp_path,
            *inputs,
            fine_cut=False,
        )


def test_a_failing_engine_raises(tmp_path, fixture_mesh):
    verts, faces = fixture_mesh("dumbbell")
    inputs = mesh_inputs([(verts, faces, "ORGANIC", False)])
    failing = ToolsCommand([sys.executable, "-c", "import sys; sys.exit(2)"], None)
    with pytest.raises(EngineError, match="exited 2"):
        segment_job(failing, tmp_path, *inputs, fine_cut=False)


@pytest.mark.smoke
def test_hard_and_kept_parts_get_no_regions(tmp_path, tools, unit_cube):
    inputs = mesh_inputs(
        [
            (*cube_at(unit_cube, 0.0), "HARD", False),
            (*cube_at(unit_cube, 5.0), "ORGANIC", True),
        ]
    )
    result = segment_job(tools, tmp_path, *inputs, fine_cut=False)
    assert result.region_count == 0
    assert not result.labels.any()
    assert result.border_edges == []
    assert (
        result.surface_modes.tolist()
        == [SURFACE_MODE_CODES["HARD"]] * 6 + [SURFACE_MODE_CODES["ORGANIC"]] * 6
    )


# four quads in a row, the second and third share a long edge, the others short ones
STRIP_VERTS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (2.0, -1.0, 0.0),
    (3.0, 0.0, 0.0),
    (4.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 1.0, 0.0),
    (2.0, 2.0, 0.0),
    (3.0, 1.0, 0.0),
    (4.0, 1.0, 0.0),
]
STRIP_FACES = [(0, 1, 6, 5), (1, 2, 7, 6), (2, 3, 8, 7), (3, 4, 9, 8)]


def strip_arrays():
    positions = numpy.array(STRIP_VERTS)
    corners = numpy.array([v for face in STRIP_FACES for v in face], dtype=numpy.int64)
    totals = numpy.array([4] * len(STRIP_FACES), dtype=numpy.int64)
    return positions, corners, totals


def test_pair_borders_join_and_split_the_strip():
    _, corners, totals = strip_arrays()
    fine = numpy.array([1, 1, 2, 3], dtype=numpy.int64)
    borders = pair_borders(edge_face_pairs(corners, totals), fine)

    assert set(borders) == {(1, 2), (2, 3)}
    assert borders[(1, 2)].tolist() == [[2, 7]]
    assert borders[(2, 3)].tolist() == [[3, 8]]
    assert pair_edges(borders, {(1, 2), (2, 3)}) == [(2, 7), (3, 8)]
    assert pair_edges(borders, set()) == []

    coarse = numpy.array([1, 1, 1, 2], dtype=numpy.int64)
    assert pairs_between_regions(borders, coarse, fine) == {(2, 3)}
    assert on_pairs_with_edges(borders, [(3, 8)]) == {(2, 3)}

    labels, count = joined_labels(fine, borders, {(2, 3)})
    assert labels.tolist() == [1, 1, 1, 2]
    assert count == 2
    labels, count = joined_labels(fine, borders, set())
    assert labels.tolist() == [1, 1, 1, 1]
    assert count == 1


def test_joined_labels_skip_fine_labels_without_faces():
    fine = numpy.array([1, 1, 3, 3, 4], dtype=numpy.int64)
    borders = {(1, 3): numpy.array([[0, 1]]), (3, 4): numpy.array([[2, 3]])}
    labels, count = joined_labels(fine, borders, set(borders))
    assert labels.tolist() == [1, 1, 2, 2, 3]
    assert count == 3


def test_nested_fine_labels_split_a_fine_region_the_coarse_cut_crosses():
    coarse = numpy.array([0, 1, 1, 2, 2], dtype=numpy.int64)
    fine = numpy.array([0, 1, 2, 2, 3], dtype=numpy.int64)
    assert nested_fine_labels(coarse, fine).tolist() == [0, 1, 2, 3, 4]


def test_stored_borders_skip_the_engine(tmp_path, fixture_mesh):
    verts, faces = fixture_mesh("dumbbell")
    inputs = mesh_inputs([(verts, faces, "ORGANIC", False)])
    centers = numpy.array([inputs.positions[face][:, 0].mean() for face in faces])
    fine = numpy.where(centers < 0, 1, 2).astype(numpy.int64)
    borders = pair_borders(edge_face_pairs(inputs.corners, inputs.totals), fine)
    neck = borders[(1, 2)]
    missing = ToolsCommand(["no-such-segment-engine.exe"], None)

    result = stored_segment_job(missing, tmp_path, *inputs[:4], fine, neck)
    assert result.hand_edited
    assert result.region_count == 2
    assert result.labels.tolist() == fine.tolist()
    assert sorted(result.border_edges) == sorted(map(tuple, neck.tolist()))

    result = stored_segment_job(missing, tmp_path, *inputs[:4], fine, neck[:0])
    assert result.region_count == 1
    assert result.border_edges == []
    assert result.fine_labels.tolist() == fine.tolist()


# @pytest.mark.skipif(not BUNDLED.is_file(), reason="segment.exe not built")
# def test_split_region_cuts_the_clicked_bulb_off_on_the_neck(tmp_path):
#     verts, faces = read_obj(FIXTURES / "dumbbell.obj")
#     positions, corners, totals, _, _ = mesh_inputs([(verts, faces, "ORGANIC", False)])
#     labels = numpy.ones(len(faces), dtype=numpy.int64)
#     centers = numpy.array([positions[face].mean(axis=0) for face in faces])
#     seed = int(numpy.argmin(centers[:, 0]))

#     result = split_region_job(
#         [BUNDLED], tmp_path, positions, corners, totals, labels, seed, None, None
#     )

#     split = result.labels
#     assert result.piece is not None
#     assert sorted(numpy.unique(split).tolist()) == [1, 2]
#     # the clicked bulb is the new piece, the other bulb keeps its label
#     assert split[seed] == 2
#     assert split[int(numpy.argmax(centers[:, 0]))] == 1
#     face_pairs = edge_face_pairs(positions, corners, totals)
#     border_vertices = numpy.unique(numpy.array(border_edges(face_pairs, split)))
#     radii = numpy.linalg.norm(positions[border_vertices][:, 1:], axis=1)
#     assert numpy.allclose(radii, NECK_RADIUS, atol=1e-4)


# @pytest.mark.skipif(not BUNDLED.is_file(), reason="segment.exe not built")
# def test_split_loop_cuts_the_clicked_region_on_the_ring_through_the_edge(tmp_path):
#     verts, faces = read_obj(FIXTURES / "dumbbell.obj")
#     positions, corners, totals, _, _ = mesh_inputs([(verts, faces, "ORGANIC", False)])
#     centers = numpy.array([positions[face].mean(axis=0) for face in faces])
#     # a quad on the left bulb, its corner edge the ring around the bulb
#     seed = int(numpy.argmin(numpy.abs(centers[:, 0] + 2.5)))
#     quad = faces[seed]
#     corner = next(
#         c for c in range(4) if positions[quad[c], 0] == positions[quad[(c + 1) % 4], 0]
#     )
#     ring_x = positions[quad[corner], 0]
#     labels = numpy.ones(len(faces), dtype=numpy.int64)

#     split = split_loop_job(
#         [BUNDLED], tmp_path, positions, corners, totals, labels, seed, corner
#     )

#     assert sorted(numpy.unique(split).tolist()) == [1, 2]
#     assert split[seed] == 2
#     face_pairs = edge_face_pairs(positions, corners, totals)
#     border_vertices = numpy.unique(numpy.array(border_edges(face_pairs, split)))
#     assert numpy.allclose(positions[border_vertices, 0], ring_x)
#     # a region already ending on that ring has nothing on the far side
#     bounded = numpy.where(centers[:, 0] < ring_x, 1, 2)
#     with pytest.raises(EngineError, match="does not cut"):
#         split_loop_job(
#             [BUNDLED], tmp_path, positions, corners, totals, bounded, seed, corner
#         )
