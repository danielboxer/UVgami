import math

import numpy
import pytest
from uvgami_cli._pytest_plugin import corners_and_totals, quad_grid

from no_bpy import OptcutsRun
from no_bpy.painted_islands import (
    borders_outside_paint,
    outline_edges,
    painted_flatten_job,
    painted_outline_job,
    unpaint_closed_parts,
)

GRID = 6
SIDE = GRID + 1


@pytest.fixture
def optcuts(engine_binary):
    return OptcutsRun([engine_binary("optcuts")], None, ["-u", "4.2"])


def column_edges(column):
    return [[y * SIDE + column, (y + 1) * SIDE + column] for y in range(GRID)]


def bumped_grid():
    verts, faces = quad_grid(GRID)
    return [(x, y, math.sin(x) * math.cos(y)) for x, y, _ in verts], faces


def fan_areas(uvs, totals):
    areas = []
    start = 0
    for total in totals.tolist():
        for k in range(1, total - 1):
            a, b, c = uvs[start], uvs[start + k], uvs[start + k + 1]
            areas.append((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        start += total
    return numpy.array(areas)


def test_outline_edges_lists_the_edges_between_painted_and_unpainted_faces():
    _, faces = quad_grid(GRID)
    corners, totals = corners_and_totals(faces)
    painted = numpy.array([index % GRID < 2 for index in range(len(faces))])

    edges = outline_edges(corners, totals, painted)

    assert edges.tolist() == sorted(column_edges(2))


def test_region_borders_inside_the_paint_are_dropped():
    _, faces = quad_grid(GRID)
    corners, totals = corners_and_totals(faces)
    painted = numpy.array([index % GRID < 4 for index in range(len(faces))])
    inside = [tuple(edge) for edge in column_edges(2)]
    outside = [tuple(edge) for edge in column_edges(5)]
    on_outline = [tuple(edge) for edge in column_edges(4)]

    kept = borders_outside_paint(
        corners, totals, painted, inside + outside + on_outline
    )

    assert kept == outside + on_outline


def quad_cube(cells):
    index = {}
    faces = []
    for axis in range(3):
        for side in (0, cells):
            for a in range(cells):
                for b in range(cells):
                    face = []
                    for step_a, step_b in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        point = [0, 0, 0]
                        point[axis] = side
                        point[(axis + 1) % 3] = a + step_a
                        point[(axis + 2) % 3] = b + step_b
                        face.append(index.setdefault(tuple(point), len(index)))
                    faces.append(face if side else face[::-1])
    return sorted(index, key=index.get), faces


def top_cells(verts, faces, cells):
    top = {}
    for index, face in enumerate(faces):
        points = [verts[v] for v in face]
        if all(point[2] == cells for point in points):
            top[(min(p[0] for p in points), min(p[1] for p in points))] = index
    return top


def test_a_fully_painted_closed_part_is_unpainted():
    cube_verts, cube_faces = quad_cube(2)
    _, grid_faces = quad_grid(GRID)
    offset = len(cube_verts)
    faces = cube_faces + [[v + offset for v in face] for face in grid_faces]
    corners, totals = corners_and_totals(faces)
    painted = numpy.ones(len(faces), dtype=bool)
    painted[len(cube_faces) :] = [index % GRID < 2 for index in range(len(grid_faces))]

    kept = unpaint_closed_parts(corners, totals, painted)

    expected = painted.copy()
    expected[: len(cube_faces)] = False
    assert kept.tolist() == expected.tolist()


def test_outline_job_fits_the_paint_to_closed_edge_loops(tools, tmp_path):
    cells = 8
    verts, faces = quad_cube(cells)
    corners, totals = corners_and_totals(faces)
    top = top_cells(verts, faces, cells)
    square = {(x, y) for x in range(2, 6) for y in range(2, 6)}
    painted = numpy.zeros(len(faces), dtype=bool)
    painted[[top[cell] for cell in (square - {(2, 2)}) | {(6, 3)}]] = True

    snapped = painted_outline_job(
        tools, tmp_path, numpy.array(verts, dtype=float), corners, totals, painted, None
    )

    expected = {top[(x, y)] for x in range(2, 7) for y in range(2, 6)}
    assert set(numpy.flatnonzero(snapped).tolist()) == expected


def test_outline_job_keeps_a_hole_inside_the_paint(tools, tmp_path):
    cells = 8
    verts, faces = quad_cube(cells)
    corners, totals = corners_and_totals(faces)
    top = top_cells(verts, faces, cells)
    square = {(x, y) for x in range(2, 6) for y in range(2, 6)}
    painted = numpy.zeros(len(faces), dtype=bool)
    painted[[top[cell] for cell in square - {(3, 3)}]] = True

    snapped = painted_outline_job(
        tools, tmp_path, numpy.array(verts, dtype=float), corners, totals, painted, None
    )

    assert set(numpy.flatnonzero(snapped).tolist()) == set(
        numpy.flatnonzero(painted).tolist()
    )


def test_outline_job_paints_the_mirror_of_each_painted_face(tools, tmp_path):
    verts, faces = quad_grid(GRID)
    corners, totals = corners_and_totals(faces)
    painted = numpy.array([index % GRID == 0 for index in range(len(faces))])
    center = (GRID / 2, GRID / 2, 0.0)

    snapped = painted_outline_job(
        tools,
        tmp_path,
        numpy.array(verts),
        corners,
        totals,
        painted,
        (center, "X"),
    )

    columns = sorted({index % GRID for index in numpy.flatnonzero(snapped).tolist()})
    assert columns == [0, GRID - 1]


def test_flatten_job_keeps_a_hole_open_without_flips(tools, optcuts, tmp_path):
    verts, faces = bumped_grid()
    hole = {(2, 2), (3, 2), (2, 3), (3, 3)}
    faces = [
        face
        for index, face in enumerate(faces)
        if (index % GRID, index // GRID) not in hole
    ]
    corners, totals = corners_and_totals(faces)

    flattened = painted_flatten_job(
        tools,
        optcuts,
        tmp_path,
        numpy.array(verts),
        corners,
        totals,
        numpy.ones(len(faces), dtype=bool),
    )

    assert flattened.overlapping_islands == 0
    assert len(flattened.corner_uvs) == len(corners)
    assert (fan_areas(flattened.corner_uvs, totals) > 0).all()
