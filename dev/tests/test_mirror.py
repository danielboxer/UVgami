import numpy
import pytest

from seams import EngineError
from seams.mirror import (
    mirror_partners,
    mirrored_corner_sources,
    partner_map,
    symmetric_axis,
    symmetric_regions,
)

# the mirror engine's cells and share, as shares of the bounding box diagonal
MATCH_CELL = 1e-4
SYMMETRY_CELL = 3e-3
SYMMETRY_SHARE = 0.95
TOLERANCE = 1e-6


@pytest.fixture
def tools(engine_binary):
    return [engine_binary("uvgami-tools")]


def mesh_arrays(coords, faces):
    positions = numpy.array(coords, dtype=numpy.float64).reshape(-1, 3)
    corners = numpy.array([v for face in faces for v in face], dtype=numpy.int64)
    totals = numpy.array([len(face) for face in faces], dtype=numpy.int64)
    return positions, corners, totals


@pytest.fixture
def axis_of(tools, tmp_path):
    def find(coords, faces, center):
        return symmetric_axis(
            tools, tmp_path, *mesh_arrays(coords, faces), center, TOLERANCE
        )

    return find


@pytest.fixture
def partners_of(tools, tmp_path):
    def find(coords, center, axis):
        positions = numpy.array(coords, dtype=numpy.float64)
        return mirror_partners(tools, tmp_path, positions, center, axis)

    return find


WING_COORDS = [(1, 0, 0), (2, 0, 0), (1, 1, 0), (-1, 0, 0), (-2, 0, 0), (-1, 1, 0)]


def test_mirror_partners_snaps_drift_and_drops_the_asymmetric_vertex(partners_of):
    # one wing tip drifted within tolerance, plus a vertex with no counterpart
    coords = [co for co in WING_COORDS] + [(0.5, 2.0, 0.0)]
    coords[4] = (-2.002, 0, 0)

    partners = partners_of(coords, (0, 0, 0), "X")

    assert partners == {0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}


def test_mirror_partners_ignores_topology(partners_of):
    coords = [
        (1, 0, 0),
        (2, 0, 0),
        (2, 1, 0),
        (1, 1, 0),
        (-1, 0, 0),
        (-2, 0, 0),
        (-2, 1, 0),
        (-1, 1, 0),
    ]

    partners = partners_of(coords, (0, 0, 0), "X")

    assert partners == {0: 4, 1: 5, 2: 6, 3: 7, 4: 0, 5: 1, 6: 2, 7: 3}


def test_mirror_partners_is_empty_when_nothing_matches(partners_of):
    coords = [(1, 0, 0), (2, 0, 0), (1, 1, 0)]

    assert partners_of(coords, (0, 0, 0), "X") == {}


def symmetric_cloud(count, seed=1):
    rng = numpy.random.default_rng(seed)
    half = rng.random((count, 3)) * [1.0, 2.0, 3.0] + [0.05, 0.0, 0.0]
    return numpy.concatenate([half, half * [-1.0, 1.0, 1.0]])


def test_symmetric_axis_takes_the_first_axis_the_vertices_mirror_across(axis_of):
    cube = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    assert axis_of(cube, [(0, 1, 3, 2)], (0, 0, 0))[:2] == ("X", True)


def test_symmetric_axis_ignores_the_triangulation(axis_of):
    quad = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
    assert axis_of(quad, [(0, 1, 2), (0, 2, 3)], (0, 0, 0))[:2] == ("X", True)


def test_symmetric_axis_skips_an_axis_the_mesh_is_flat_along(axis_of):
    quad = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
    sheared = [(x, y, z + 0.5 * x) for x, y, z in quad]
    # sheared in z, so neither X nor Z mirrors and Y is the only candidate
    assert axis_of(sheared, [(0, 1, 2, 3)], (0, 0, 0))[:2] == ("Y", True)


def test_symmetric_axis_flags_a_face_across_the_plane(axis_of):
    quad = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
    assert axis_of(quad, [(0, 1, 2, 3)], (0, 0, 0)).crosses
    wings = [(1, 0, 0), (2, 0, 0), (1, 1, 0), (-1, 0, 0), (-2, 0, 0), (-1, 1, 0)]
    assert not axis_of(wings, [(0, 1, 2), (3, 5, 4)], (0, 0, 0)).crosses


def test_symmetric_axis_matches_a_dense_cloud_exactly(axis_of):
    coords = symmetric_cloud(2000)
    center = (coords.min(axis=0) + coords.max(axis=0)) / 2
    assert axis_of(coords, [], center)[:2] == ("X", True)


def test_symmetric_axis_calls_a_cloud_with_moved_vertices_approximate(axis_of):
    coords = symmetric_cloud(2000)
    center = (coords.min(axis=0) + coords.max(axis=0)) / 2
    diagonal = numpy.linalg.norm(coords.max(axis=0) - coords.min(axis=0))
    # a few vertices off their mirror by more than the exact cell, within the loose one
    coords[:20, 1] += 3 * MATCH_CELL * diagonal
    assert axis_of(coords, [], center)[:2] == ("X", False)

    # too many vertices moved out of the loose cell and no axis is left
    moved = int(len(coords) * (1 - SYMMETRY_SHARE)) + 1
    coords[:moved, 1] += 3 * SYMMETRY_CELL * diagonal
    assert axis_of(coords, [], center) is None


def test_symmetric_axis_prefers_an_exact_axis_over_an_earlier_approximate_one(axis_of):
    coords = symmetric_cloud(500, seed=4)
    # mirror across y too, then drift the x side so only y stays exact
    half = len(coords)
    coords = numpy.concatenate([coords, coords * [1.0, -1.0, 1.0]])
    center = (coords.min(axis=0) + coords.max(axis=0)) / 2
    diagonal = numpy.linalg.norm(coords.max(axis=0) - coords.min(axis=0))
    # moved with their y partners, so y stays an exact mirror and x does not
    coords[:10, 0] += 3 * MATCH_CELL * diagonal
    coords[half : half + 10, 0] += 3 * MATCH_CELL * diagonal
    assert axis_of(coords, [], center)[:2] == ("Y", True)


def test_symmetric_axis_accepts_drift_within_the_match_cell(axis_of):
    coords = symmetric_cloud(300, seed=2)
    center = (coords.min(axis=0) + coords.max(axis=0)) / 2
    diagonal = numpy.linalg.norm(coords.max(axis=0) - coords.min(axis=0))
    # a quarter cell per coordinate on both partners keeps each pair inside a cell
    rng = numpy.random.default_rng(3)
    coords += rng.uniform(-0.25, 0.25, coords.shape) * MATCH_CELL * diagonal
    assert axis_of(coords, [], center)[:2] == ("X", True)


def test_mirror_partners_pairs_only_vertices_that_point_at_each_other(partners_of):
    # two vertices on the right share one on the left, which pairs with the nearer
    coords = [(1, 0, 0), (1.001, 0, 0), (-1, 0, 0), (0, 5, 0), (0, -5, 0)]

    partners = partners_of(coords, (0, 0, 0), "X")

    assert partners == {0: 2, 2: 0, 3: 3, 4: 4}


def test_mirror_partners_pairs_both_copies_of_a_split_vertex(partners_of, axis_of):
    # the right vertex is stored twice, as along a uv seam, the left one once
    coords = [(1, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 5, 0), (0, -5, 0)]

    partners = partners_of(coords, (0, 0, 0), "X")

    assert partners == {0: 2, 1: 2, 2: 0, 3: 3, 4: 4}
    found = axis_of(coords, [(0, 2, 3), (1, 2, 4)], (0, 0, 0))
    assert found[:2] == ("X", True)
    # the axis carries the same map for the whole mesh route
    assert partner_map(found.partners) == partners


def test_mirrored_corner_sources_maps_each_negative_corner_to_its_mirror_image(
    tools, tmp_path
):
    # a quad on each side of x = 0 split along opposite diagonals, vertex 6 is a copy of vertex 1
    coords = [
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (-1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    positions, corners, totals = mesh_arrays(
        coords, [[0, 1, 4], [0, 4, 3], [6, 2, 4], [2, 5, 4]]
    )

    targets, sources = mirrored_corner_sources(
        tools, tmp_path, positions, corners, totals, numpy.zeros(3), "X", TOLERANCE
    )

    assert targets.tolist() == [0, 1, 2, 3, 4, 5]
    assert (sources >= 6).all()
    mirrored = positions[corners[targets]] * [-1.0, 1.0, 1.0]
    numpy.testing.assert_allclose(positions[corners[sources]], mirrored)


def test_mirrored_corner_sources_raises_when_a_vertex_has_no_mirror(tools, tmp_path):
    positions, corners, totals = mesh_arrays(
        [(-1, 0, 0), (-2, 0, 0), (-1, 1, 0), (1, 0, 0), (2, 0, 0), (1, 2, 0)],
        [(0, 2, 1), (3, 4, 5)],
    )
    with pytest.raises(EngineError, match="doesn't mirror across X"):
        mirrored_corner_sources(
            tools, tmp_path, positions, corners, totals, numpy.zeros(3), "X", TOLERANCE
        )


# a row of quads along x from -1 to 1, face k spans columns k and k + 1
def quad_row(face_count):
    xs = numpy.linspace(-1.0, 1.0, face_count + 1)
    coords = [(x, y, 0.0) for x in xs for y in (0.0, 1.0)]
    faces = [(2 * k, 2 * k + 2, 2 * k + 3, 2 * k + 1) for k in range(face_count)]
    return coords, faces


@pytest.fixture
def mirror_regions(tools, tmp_path):
    def run(coords, faces, labels):
        positions, corners, totals = mesh_arrays(coords, faces)
        return symmetric_regions(
            tools,
            tmp_path,
            positions,
            corners,
            totals,
            numpy.array(labels, dtype=numpy.int64),
            "X",
            numpy.zeros(3),
            TOLERANCE,
        )

    return run


def test_symmetric_regions_keeps_the_half_with_more_regions_then_shorter_borders(
    mirror_regions,
):
    coords, faces = quad_row(4)

    # only one half has a second region, its border length doesn't matter
    assert mirror_regions(coords, faces, [1, 1, 1, 2])[0] == 1
    assert mirror_regions(coords, faces, [2, 1, 1, 1])[0] == -1

    # both halves have two regions, the positive border has a step in it
    coords[7] = (0.6, 1.0, 0.0)
    assert mirror_regions(coords, faces, [1, 2, 2, 3])[0] == -1


def test_symmetric_regions_keeps_a_border_that_runs_across_the_plane(mirror_regions):
    # two rows of four quads, each row one region crossing x = 0
    xs = (-1.0, -0.5, 0.0, 0.5, 1.0)
    coords = [(x, y, 0.0) for x in xs for y in (0.0, 1.0, 2.0)]
    faces = [
        (
            column * 3 + row,
            column * 3 + 3 + row,
            column * 3 + 4 + row,
            column * 3 + 1 + row,
        )
        for column in range(4)
        for row in range(2)
    ]
    rows = [1, 2] * 4

    _, labels = mirror_regions(coords, faces, rows)

    assert labels.tolist() == rows


def test_symmetric_regions_copies_one_sided_regions_with_new_labels(mirror_regions):
    # face 3 is a leg region on the positive half, face 0 is its mirror image
    coords, faces = quad_row(4)

    side, labels = mirror_regions(coords, faces, [5, 1, 1, 2])

    assert side == 1
    assert labels[2:].tolist() == [1, 2]
    assert labels[1] == 1
    assert labels[0] not in (1, 2, 5)


def test_symmetric_regions_keeps_the_engine_border_where_the_regions_match(
    mirror_regions,
):
    # the negative half's border sits one face off the mirror of the positive half's
    coords, faces = quad_row(40)
    labels = [3] * 9 + [4] * 11 + [1] * 10 + [2] * 10

    side, mirrored = mirror_regions(coords, faces, labels)

    assert side == 1
    assert mirrored[20:].tolist() == labels[20:]
    copies = mirrored[:20].tolist()
    assert copies == [copies[0]] * 9 + [copies[9]] * 11
    assert copies[0] != copies[9]
    assert not {copies[0], copies[9]} & {1, 2, 3, 4}
