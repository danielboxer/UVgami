import sys

import numpy
import pytest

from seams import SURFACE_MODE_CODES, EngineError
from seams.segmentation import (
    border_edges,
    # split_loop_job,
    # split_region_job,
    edge_face_pairs,
    # merge_region,
    segment_job,
)
from seams.worker import WorkerProcess

# the dumbbell fixture's bulbs meet a straight neck of this radius around the x axis
NECK_RADIUS = 0.3
# an all organic mesh never starts the classifier
NO_HARDSURFACE = ["no-such-hardsurface-engine.exe"]


# (verts, faces, mode code, kept) per loose part, flattened into the worker's inputs
def mesh_inputs(parts):
    verts, faces, modes, kept = [], [], [], []
    for part_verts, part_faces, mode, part_kept in parts:
        base = len(verts)
        verts += part_verts
        faces += [[base + v for v in face] for face in part_faces]
        modes += [SURFACE_MODE_CODES[mode]] * len(part_faces)
        kept += [part_kept] * len(part_faces)
    return (
        numpy.array(verts, dtype=numpy.float64),
        numpy.array([v for face in faces for v in face], dtype=numpy.int64),
        numpy.array([len(face) for face in faces], dtype=numpy.int64),
        numpy.array(modes, dtype=numpy.int32),
        numpy.array(kept, dtype=bool),
    )


def cube_at(unit_cube, y):
    verts, faces = unit_cube
    return [[x, cy + y, z] for x, cy, z in verts], faces


@pytest.mark.smoke
def test_dumbbell_splits_on_the_neck_and_skips_hard_and_kept_parts(
    tmp_path, engine_binary, fixture_mesh, unit_cube
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
    positions = inputs[0]

    tools = [engine_binary("uvgami-tools")]
    result = WorkerProcess(
        segment_job, [*tools, "segment"], tools, tmp_path, *inputs
    ).result()

    assert SURFACE_MODE_CODES["AUTO"] not in result.surface_modes
    assert result.region_count >= 2
    assert result.border_edges
    border_vertices = numpy.unique(numpy.array(result.border_edges))
    # every vertex past the dumbbell's belongs to the hard, kept or auto cube
    assert border_vertices.max() < len(dumbbell_verts)
    radii = numpy.linalg.norm(positions[border_vertices][:, 1:], axis=1)
    assert numpy.allclose(radii, NECK_RADIUS, atol=1e-4)


def test_a_missing_engine_raises(tmp_path, fixture_mesh):
    verts, faces = fixture_mesh("dumbbell")
    inputs = mesh_inputs([(verts, faces, "ORGANIC", False)])
    with pytest.raises(EngineError, match="failed to start"):
        segment_job(["no-such-segment-engine.exe"], NO_HARDSURFACE, tmp_path, *inputs)


def test_a_failing_engine_raises(tmp_path, fixture_mesh):
    verts, faces = fixture_mesh("dumbbell")
    inputs = mesh_inputs([(verts, faces, "ORGANIC", False)])
    failing = [sys.executable, "-c", "import sys; sys.exit(2)"]
    with pytest.raises(EngineError, match="exited 2"):
        segment_job(failing, NO_HARDSURFACE, tmp_path, *inputs)


@pytest.mark.smoke
def test_hard_and_kept_parts_never_start_the_engine(tmp_path, engine_binary, unit_cube):
    inputs = mesh_inputs(
        [
            (*cube_at(unit_cube, 0.0), "HARD", False),
            (*cube_at(unit_cube, 5.0), "ORGANIC", True),
        ]
    )
    tools = [engine_binary("uvgami-tools")]
    result = segment_job(["no-such-segment-engine.exe"], tools, tmp_path, *inputs)
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


def test_edge_face_pairs_lists_each_shared_edge_once():
    _, corners, totals = strip_arrays()
    face_pairs = edge_face_pairs(corners, totals)
    pairs = sorted(zip(face_pairs.first.tolist(), face_pairs.second.tolist()))
    assert pairs == [(0, 1), (1, 2), (2, 3)]
    # assert sorted(face_pairs.lengths.tolist()) == [1.0, 1.0, 3.0]
    assert sorted(map(tuple, face_pairs.edges.tolist())) == [(1, 6), (2, 7), (3, 8)]


def test_border_edges_lists_a_three_face_edge_once():
    corners = numpy.array([0, 1, 2, 0, 1, 3, 0, 1, 4], dtype=numpy.int64)
    totals = numpy.array([3, 3, 3], dtype=numpy.int64)
    face_pairs = edge_face_pairs(corners, totals)
    assert border_edges(face_pairs, numpy.array([1, 2, 3])) == [(0, 1)]


# def test_merge_region_joins_the_neighbor_with_the_longest_border():
#     face_pairs = edge_face_pairs(*strip_arrays())
#     labels = numpy.array([1, 2, 3, 3], dtype=numpy.int64)
#     assert merge_region(labels, face_pairs, 2).tolist() == [1, 3, 3, 3]
#     assert border_edges(face_pairs, labels) == [(1, 6), (2, 7)]


# def test_merge_region_refuses_a_region_with_no_region_neighbor():
#     face_pairs = edge_face_pairs(*strip_arrays())
#     labels = numpy.array([1, 0, 0, 2], dtype=numpy.int64)
#     with pytest.raises(EngineError, match="no other region"):
#         merge_region(labels, face_pairs, 1)


# def test_stored_labels_skip_the_engine_and_drop_off_hard_faces(tmp_path):
#     inputs = mesh_inputs(
#         [(*cube_at(0.0), "ORGANIC", False), (*cube_at(5.0), "HARD", False)]
#     )
#     stored = numpy.array([1, 1, 1, 2, 2, 2] + [3] * 6, dtype=numpy.int64)

#     result = segment_job(["no-such-segment-engine.exe"], tmp_path, *inputs, stored)

#     assert result.labels.tolist() == stored[:6].tolist() + [0] * 6
#     assert result.region_count == 2
#     # the hard cube shares no vertex with the organic one, its faces have no borders
#     assert all(max(edge) < 8 for edge in result.border_edges)
#     # three faces of one cube against the other three share eight edges
#     assert len(result.border_edges) == 8


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
