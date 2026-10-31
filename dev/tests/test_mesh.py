import math

import numpy
import pytest
from uvgami_cli._pytest_plugin import grid_uvs, quad_grid

from no_bpy import (
    boxes_overlap,
    faces_by_edge,
    fit_uvs_into_box,
    fit_uvs_keeping_area,
    island_layout,
    islands_from_uvs,
    signed_area,
    uvs_collapsed_to_points,
)
from no_bpy.mesh import (
    lowest_index_part_labels,
    moved_island_uvs,
    torn_interior_edges,
)


def strip_island(quads, scale=1.0, angle=0.0):
    points, faces = [], []
    cos, sin = math.cos(angle), math.sin(angle)
    for column in range(quads + 1):
        for y in (0.0, 1.0):
            x = float(column)
            points.append((scale * (x * cos - y * sin), scale * (x * sin + y * cos)))
    for column in range(quads):
        a, b, c, d = 2 * column, 2 * column + 1, 2 * column + 2, 2 * column + 3
        faces.append([a, c, b])
        faces.append([c, d, b])
    uvs = [[points[v] for v in face] for face in faces]
    verts = [(x, y, 0.0) for x, y in points]
    return verts, faces, uvs


def test_uv_islands_follow_the_uv_map():
    # a flat strip is one island until its uvs split mid-column
    verts, faces, uvs = strip_island(4)
    edges = faces_by_edge(faces)
    assert len(islands_from_uvs(faces, uvs, edges)) == 1
    for face_index in range(4, 8):
        uvs[face_index] = [(u + 5.0, v) for u, v in uvs[face_index]]
    groups = islands_from_uvs(faces, uvs, edges)
    assert sorted(len(group) for group in groups) == [4, 4]


def test_uv_fit_scales_into_old_bounds():
    move = fit_uvs_into_box([(0, 0), (2, 1)], (10, 10, 11, 10.5))
    assert move((0, 0)) == (10, 10)
    assert move((2, 1)) == (11, 10.5)
    # aspect mismatch keeps the scale uniform and the island inside
    move = fit_uvs_into_box([(0, 0), (2, 1)], (0, 0, 1, 1))
    assert move((0, 0)) == (0, 0.25)
    assert move((2, 1)) == (1, 0.75)


def test_uv_area_fit_keeps_the_old_uv_area():
    # two charts packed in a square, the island they came from was a 4:1 strip
    charts = [
        [(0, 0), (1, 0), (1, 0.4), (0, 0.4)],
        [(0, 0.6), (1, 0.6), (1, 1), (0, 1)],
    ]
    move = fit_uvs_keeping_area(charts, 0.8, (0, 0, 0.8, 0.2))
    moved = [[move(uv) for uv in chart] for chart in charts]
    area = sum(abs(signed_area(chart)) for chart in moved)
    assert math.isclose(area, 0.8)
    # centered on the old spot, and a bbox fit would have shrunk it instead
    moved_u = [u for chart in moved for u, _ in chart]
    moved_v = [v for chart in moved for _, v in chart]
    assert math.isclose((min(moved_u) + max(moved_u)) / 2, 0.4)
    assert math.isclose((min(moved_v) + max(moved_v)) / 2, 0.1)
    assert max(moved_v) - min(moved_v) > 0.2


def test_uv_area_fit_falls_back_to_the_bbox_without_an_area():
    square = [[(0, 0), (2, 0), (2, 2), (0, 2)]]
    move = fit_uvs_keeping_area(square, 0, (0, 0, 1, 1))
    assert move((0, 0)) == (0, 0)
    assert move((2, 2)) == (1, 1)


def test_boxes_overlap_detects_stacked_boxes():
    apart = [(0, 0, 1, 1), (1.1, 0, 2, 1), (0, 1.2, 1, 2)]
    assert not boxes_overlap(apart)
    assert boxes_overlap([*apart, (0.5, 0.5, 1.5, 1.5)])


def test_island_layout_separates_and_unmirrors():
    boxes = [(0, 0, 1, 1), (0.2, 0.1, 0.8, 0.9)]
    corners = numpy.array(
        [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)] for x0, y0, x1, y1 in boxes],
        dtype=numpy.float64,
    ).reshape(-1, 2)
    moves = island_layout(boxes, [0.5, -0.3])
    moved = moved_island_uvs(corners, [[0], [1]], moves, numpy.array([4, 4]))
    moved_faces = moved.reshape(2, 4, 2)
    moved_boxes = [(*face.min(axis=0), *face.max(axis=0)) for face in moved_faces]
    assert not boxes_overlap(moved_boxes)
    # only the negative-area island mirrors, within its own bounds
    assert moves[0].mirror_sum is None
    assert moves[1].mirror_sum == 1.0
    assert moved_faces[1][0][0] > moved_faces[1][1][0]
    assert moved_boxes[1][2] - moved_boxes[1][0] == pytest.approx(
        boxes[1][2] - boxes[1][0]
    )


def test_collapsed_uvs_are_detected():
    point = (0.1, 0.9987)
    assert uvs_collapsed_to_points([[point, point, point]] * 100)


def test_real_uvs_are_not_collapsed():
    quad = [(0.0, 0.0), (0.01, 0.0), (0.01, 0.01), (0.0, 0.01)]
    assert not uvs_collapsed_to_points([quad])


def test_torn_interior_edges_finds_only_the_torn_interior_edge():
    verts, faces = quad_grid(2)
    uvs = grid_uvs(verts, faces)
    uvs[0] = [(u + 10.0, v) for u, v in uvs[0]]
    assert torn_interior_edges(faces, uvs) == {(1, 4), (3, 4)}


def test_torn_interior_edges_skips_boundary_edges():
    verts, faces = quad_grid(1)
    uvs = grid_uvs(verts, faces)
    assert torn_interior_edges(faces, uvs) == set()


def test_torn_interior_edges_ignores_a_gap_under_the_tolerance():
    verts, faces = quad_grid(2)
    uvs = grid_uvs(verts, faces)
    uvs[0] = [(u + 1e-7, v) for u, v in uvs[0]]
    assert torn_interior_edges(faces, uvs) == {(1, 4), (3, 4)}
    assert torn_interior_edges(faces, uvs, tolerance=1e-6) == set()


def test_lowest_index_part_labels_separates_loose_parts():
    edges = numpy.array(
        [[0, 1], [1, 2], [3, 4], [5, 6], [6, 7], [5, 7]], dtype=numpy.int64
    )
    labels = lowest_index_part_labels(9, edges)
    assert labels.tolist() == [0, 0, 0, 3, 3, 5, 5, 5, 8]


def test_lowest_index_part_labels_joins_a_long_chain():
    count = 1000
    ends = numpy.arange(count - 1, dtype=numpy.int64)
    edges = numpy.stack([ends, ends + 1], axis=1)
    assert lowest_index_part_labels(count, edges).tolist() == [0] * count
