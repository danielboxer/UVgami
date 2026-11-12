import itertools
import math
import time

import numpy
import pytest
from uvgami_cli._pytest_plugin import corners_and_totals, grid_uvs, quad_grid

from no_bpy.engine_run import POLL_INTERVAL
from no_bpy.proxy import TransferMirror, dense_subset, start_proxy_transfer

IDENTITY = numpy.eye(4)
GRID = 6
SIDE = GRID + 1
SHIFT = 10.0
# the proxy cut sits between dense columns 2 and 3
CUT_X = 2.5


def triangulated(faces):
    return [
        triangle
        for quad in faces
        for triangle in ([quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]])
    ]


def dense_arrays(verts, faces, matrix=IDENTITY):
    corners, face_sizes = corners_and_totals(faces)
    points = numpy.asarray(verts, dtype=numpy.float64)
    summed = numpy.zeros_like(points)
    for face in faces:
        ring = points[face]
        normal = sum(
            numpy.cross(ring[k] - ring[0], ring[k + 1] - ring[0])
            for k in range(1, len(face) - 1)
        )
        summed[face] += normal
    lengths = numpy.maximum(numpy.linalg.norm(summed, axis=1, keepdims=True), 1e-30)
    return {
        "positions": verts,
        "normals": summed / lengths,
        "matrix": matrix,
        "corners": corners,
        "face_sizes": face_sizes,
    }


def proxy_arrays(verts, faces, uvs, matrix=IDENTITY):
    return {
        "positions": verts,
        "faces": faces,
        "corner_uvs": uvs,
        "matrix": matrix,
        "painted_faces": [0] * len(faces),
    }


def uncut_proxy():
    verts, faces = quad_grid(1, GRID)
    faces = triangulated(faces)
    return proxy_arrays(verts, faces, grid_uvs(verts, faces))


def split_proxy(cut_x=CUT_X):
    verts = [
        (0.0, 0.0, 0.0),
        (cut_x, 0.0, 0.0),
        (cut_x, GRID, 0.0),
        (0.0, GRID, 0.0),
        (GRID, 0.0, 0.0),
        (GRID, GRID, 0.0),
    ]
    faces = [[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]]
    uvs = grid_uvs(verts, faces)
    for face_index in (0, 1):
        uvs[face_index] = [(u + SHIFT, v) for u, v in uvs[face_index]]
    return proxy_arrays(verts, faces, uvs)


def loose_halves_proxy():
    # the halves share no vertex, like segments decimated apart
    verts = [
        (0.0, 0.0, 0.0),
        (CUT_X, 0.0, 0.0),
        (CUT_X, GRID, 0.0),
        (0.0, GRID, 0.0),
        (CUT_X, 0.0, 0.0),
        (GRID, 0.0, 0.0),
        (GRID, GRID, 0.0),
        (CUT_X, GRID, 0.0),
    ]
    faces = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]
    return proxy_arrays(verts, faces, grid_uvs(verts, faces))


@pytest.fixture
def start_transfer(tools, tmp_path):
    def start(dense, proxy, border_edges=(), mirror=None):
        return start_proxy_transfer(
            tools, tmp_path, dense, proxy, border_edges=border_edges, mirror=mirror
        )

    return start


def finished(run):
    while not run.done():
        time.sleep(POLL_INTERVAL)
    return run


@pytest.fixture
def transfer(start_transfer):
    def run(dense, proxy, border_edges=(), mirror=None):
        return finished(start_transfer(dense, proxy, border_edges, mirror)).result()

    return run


def fan_triangles(faces):
    corner = 0
    triangles = []
    for face in faces:
        for k in range(1, len(face) - 1):
            triangles.append((corner, corner + k, corner + k + 1))
        corner += len(face)
    return numpy.array(triangles)


def signed_uv_areas(uvs, faces):
    ring = uvs[fan_triangles(faces)]
    first = ring[:, 1] - ring[:, 0]
    second = ring[:, 2] - ring[:, 0]
    return first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]


def segments_cross(a, b, c, d):
    def turn(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return turn(a, b, c) * turn(a, b, d) < 0 and turn(c, d, a) * turn(c, d, b) < 0


def boundary_uv_segments(uvs, faces):
    keys = {}
    corner = 0
    for face in faces:
        size = len(face)
        for k in range(size):
            tail, head = uvs[corner + k], uvs[corner + (k + 1) % size]
            keys[(tuple(tail), tuple(head))] = (tail, head)
        corner += size
    return [
        segment for (tail, head), segment in keys.items() if (head, tail) not in keys
    ]


def assert_no_flips_or_crossings(uvs, faces):
    assert (signed_uv_areas(uvs, faces) > 0).all()
    for (a, b), (c, d) in itertools.combinations(boundary_uv_segments(uvs, faces), 2):
        assert not segments_cross(a, b, c, d)


def uv_pieces(uvs, faces):
    parent = {}

    def root(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    corner = 0
    for face in faces:
        keys = [tuple(uvs[corner + k]) for k in range(len(face))]
        for key in keys[1:]:
            parent[root(key)] = root(keys[0])
        corner += len(face)
    return len({root(key) for key in list(parent)})


def uv_sphere(rings=10, segments=16):
    verts = [(0.0, 0.0, 1.0)]
    for ring in range(1, rings):
        polar = math.pi * ring / rings
        for segment in range(segments):
            around = 2 * math.pi * segment / segments
            verts.append(
                (
                    math.sin(polar) * math.cos(around),
                    math.sin(polar) * math.sin(around),
                    math.cos(polar),
                )
            )
    verts.append((0.0, 0.0, -1.0))

    def at(ring, segment):
        return 1 + (ring - 1) * segments + segment % segments

    faces = []
    for segment in range(segments):
        faces.append([0, at(1, segment), at(1, segment + 1)])
        faces.append(
            [at(rings - 1, segment), len(verts) - 1, at(rings - 1, segment + 1)]
        )
    for ring in range(1, rings - 1):
        for segment in range(segments):
            faces.append(
                [
                    at(ring, segment),
                    at(ring + 1, segment),
                    at(ring + 1, segment + 1),
                    at(ring, segment + 1),
                ]
            )
    return verts, faces


def torus(rings=12, segments=20):
    verts = [
        (
            (2 + 0.7 * math.cos(2 * math.pi * r / rings))
            * math.cos(2 * math.pi * s / segments),
            (2 + 0.7 * math.cos(2 * math.pi * r / rings))
            * math.sin(2 * math.pi * s / segments),
            0.7 * math.sin(2 * math.pi * r / rings),
        )
        for s in range(segments)
        for r in range(rings)
    ]
    faces = []
    for s in range(segments):
        for r in range(rings):
            a = s * rings + r
            b = ((s + 1) % segments) * rings + r
            c = ((s + 1) % segments) * rings + (r + 1) % rings
            d = s * rings + (r + 1) % rings
            faces.append([a, b, c, d])
    return verts, faces


def annulus_grid():
    verts, faces = quad_grid(GRID)
    hole = {(2, 2), (3, 2), (2, 3), (3, 3)}
    kept = [
        face
        for index, face in enumerate(faces)
        if (index % GRID, index // GRID) not in hole
    ]
    return verts, kept


def test_dense_subset_renumbers_the_kept_faces():
    verts, faces = quad_grid(2)
    subset, used = dense_subset(dense_arrays(verts, faces), [3, 0])
    assert used.tolist() == [0, 1, 3, 4, 5, 7, 8]
    assert subset["face_sizes"].tolist() == [4, 4]
    assert subset["corners"].tolist() == [3, 4, 6, 5, 0, 1, 3, 2]
    numpy.testing.assert_array_equal(subset["positions"], numpy.asarray(verts)[used])


@pytest.mark.parametrize("faces_of", [lambda faces: faces, triangulated])
def test_transfer_flattens_a_flat_uncut_chart_without_stretch(transfer, faces_of):
    verts, faces = quad_grid(GRID)
    faces = faces_of(faces)
    dense = dense_arrays(verts, faces)

    seams, uvs = transfer(dense, uncut_proxy())

    assert seams == set()
    expected = numpy.array(verts)[dense["corners"]][:, :2]
    around = expected.mean(axis=0)
    # one scale and rotation away from the flat grid
    scale = (
        numpy.linalg.norm(uvs - uvs.mean(axis=0), axis=1).sum()
        / numpy.linalg.norm(expected - around, axis=1).sum()
    )
    pairs = numpy.array([[0, len(uvs) // 2], [1, len(uvs) - 1], [2, len(uvs) // 3]])
    measured = numpy.linalg.norm(uvs[pairs[:, 0]] - uvs[pairs[:, 1]], axis=1)
    original = numpy.linalg.norm(expected[pairs[:, 0]] - expected[pairs[:, 1]], axis=1)
    assert numpy.allclose(measured, scale * original, atol=1e-2 * scale * GRID)


def test_transfer_run_reaches_full_progress(start_transfer):
    verts, faces = quad_grid(GRID)

    run = finished(start_transfer(dense_arrays(verts, faces), split_proxy()))

    assert run.progress == pytest.approx(1.0)
    run.result()


def test_transfer_matches_the_meshes_in_the_proxy_space(transfer):
    verts, faces = quad_grid(GRID)
    offset = numpy.eye(4)
    offset[0, 3] = 5.0
    proxy = split_proxy()
    proxy["positions"] = [(x + 5.0, y, z) for x, y, z in proxy["positions"]]

    seams, _ = transfer(dense_arrays(verts, faces, offset), proxy)

    assert seams in [
        {(y * SIDE + column, (y + 1) * SIDE + column) for y in range(GRID)}
        for column in (2, 3)
    ]


def test_transfer_runs_the_seam_along_the_proxy_cut(transfer):
    verts, faces = quad_grid(GRID)

    seams, uvs = transfer(dense_arrays(verts, faces), split_proxy())

    assert seams in [
        {(y * SIDE + column, (y + 1) * SIDE + column) for y in range(GRID)}
        for column in (2, 3)
    ]
    assert uv_pieces(uvs, faces) == 2
    assert_no_flips_or_crossings(uvs, faces)


def column_seams(column):
    return {(y * SIDE + column, (y + 1) * SIDE + column) for y in range(GRID)}


def mirror_across_the_middle_column(verts, *, dropped_when):
    partners = [
        index - int(x) + (GRID - int(x)) for index, (x, _, _) in enumerate(verts)
    ]
    dropped = [dropped_when(x - GRID / 2) for x, _, _ in verts]
    return TransferMirror(partners, dropped)


def test_transfer_mirror_replaces_the_dropped_side_cuts(transfer):
    verts, faces = quad_grid(GRID)
    mirror = mirror_across_the_middle_column(
        verts, dropped_when=lambda offset: offset < 0
    )

    seams, uvs = transfer(dense_arrays(verts, faces), split_proxy(1.5), mirror=mirror)

    assert seams == set()
    assert uv_pieces(uvs, faces) == 1


def test_transfer_mirror_copies_the_kept_side_cut_and_uvs(transfer):
    verts, faces = quad_grid(GRID)
    mirror = mirror_across_the_middle_column(
        verts, dropped_when=lambda offset: offset > 0
    )

    seams, uvs = transfer(dense_arrays(verts, faces), split_proxy(1.5), mirror=mirror)

    assert seams in [
        column_seams(column) | column_seams(GRID - column) for column in (1, 2)
    ]
    assert uv_pieces(uvs, faces) == 3
    corner_of = {}
    corner = 0
    for face in faces:
        for vertex in face:
            corner_of[(tuple(sorted(face)), vertex)] = corner
            corner += 1
    sums, differences = [], []
    # columns past GRID - 1 lie in the outer chart for either traced column
    for face in faces:
        if any(verts[vertex][0] < GRID - 1 for vertex in face):
            continue
        image = tuple(sorted(mirror.partners[vertex] for vertex in face))
        for vertex in face:
            dropped_uv = uvs[corner_of[(tuple(sorted(face)), vertex)]]
            kept_uv = uvs[corner_of[(image, mirror.partners[vertex])]]
            sums.append(dropped_uv[0] + kept_uv[0])
            differences.append(dropped_uv[1] - kept_uv[1])
    assert sums
    assert numpy.ptp(sums) < 1e-9
    assert numpy.ptp(differences) < 1e-9


def test_transfer_seams_every_region_border_it_is_given(transfer):
    verts, faces = quad_grid(GRID)
    border = [(y * SIDE + 3, (y + 1) * SIDE + 3) for y in range(GRID)]

    seams, uvs = transfer(dense_arrays(verts, faces), loose_halves_proxy(), border)

    assert seams == set(border)
    assert uv_pieces(uvs, faces) == 2


def test_transfer_pulls_a_zigzag_seam_onto_one_edge_row(transfer):
    verts, faces = quad_grid(GRID)
    faces = triangulated(faces)
    proxy = split_proxy()
    # the cut runs along y instead of x
    proxy["positions"] = [(y, x, z) for x, y, z in proxy["positions"]]
    proxy["faces"] = [face[::-1] for face in proxy["faces"]]
    proxy["corner_uvs"] = [uvs[::-1] for uvs in proxy["corner_uvs"]]

    seams, uvs = transfer(dense_arrays(verts, faces), proxy)

    rows = [{(y * SIDE + x, y * SIDE + x + 1) for x in range(GRID)} for y in (2, 3)]
    assert seams in rows
    assert_no_flips_or_crossings(uvs, faces)


def test_transfer_follows_the_diagonals_under_a_diagonal_cut(transfer):
    verts, faces = quad_grid(GRID)
    faces = triangulated(faces)
    # two proxy triangles either side of the line y = x + 0.5
    proxy_verts = [
        (-1.0, -0.5, 0.0),
        (7.0, 7.5, 0.0),
        (-1.0, 8.0, 0.0),
        (8.0, -1.0, 0.0),
    ]
    proxy_faces = [[0, 1, 2], [0, 3, 1]]
    uvs = grid_uvs(proxy_verts, proxy_faces)
    uvs[0] = [(u + SHIFT, v) for u, v in uvs[0]]
    proxy = proxy_arrays(proxy_verts, proxy_faces, uvs)

    seams, _ = transfer(dense_arrays(verts, faces), proxy)

    off_diagonal = [edge for edge in seams if edge[1] - edge[0] != SIDE + 1]
    # the run's two ends sit on the grid boundary and may keep a step each
    assert len(off_diagonal) <= 2, off_diagonal
    assert len(seams) - len(off_diagonal) >= GRID - 1


def test_transfer_keeps_a_cut_on_its_own_part_of_overlapping_parts(transfer):
    near_verts, near_faces = quad_grid(GRID)
    # a second grid just above the first, the cut belongs to the first
    far_verts = [(x + 0.3, y + 0.3, z + 0.01) for x, y, z in near_verts]
    far_faces = [[v + len(near_verts) for v in face] for face in near_faces]
    verts, faces = near_verts + far_verts, near_faces + far_faces

    seams, uvs = transfer(dense_arrays(verts, faces), split_proxy())

    assert all(b < len(near_verts) for _, b in seams)
    assert uv_pieces(uvs, faces) == 3
    assert_no_flips_or_crossings(uvs, faces)


def test_transfer_opens_a_closed_sphere_the_proxy_left_uncut(transfer):
    verts, faces = uv_sphere()

    seams, uvs = transfer(dense_arrays(verts, faces), uncut_proxy())

    assert seams
    assert uv_pieces(uvs, faces) == 1
    assert_no_flips_or_crossings(uvs, faces)


TETRAHEDRON = (
    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
    [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]],
)
DOUBLE_SIDED_TRIANGLE = (
    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
    [[0, 1, 2], [0, 2, 1]],
)


@pytest.mark.parametrize("verts, faces", [TETRAHEDRON, DOUBLE_SIDED_TRIANGLE])
def test_transfer_opens_a_closed_part_whose_vertices_all_share_edges(
    transfer, verts, faces
):
    seams, uvs = transfer(dense_arrays(verts, faces), uncut_proxy())

    assert len(seams) >= 2
    assert uv_pieces(uvs, faces) == 1
    assert_no_flips_or_crossings(uvs, faces)


def test_transfer_cuts_a_torus_handle_the_proxy_left_uncut(transfer):
    verts, faces = torus()

    seams, uvs = transfer(dense_arrays(verts, faces), uncut_proxy())

    assert seams
    assert uv_pieces(uvs, faces) == 1
    assert_no_flips_or_crossings(uvs, faces)


def test_transfer_cuts_a_hole_to_the_outer_boundary(transfer):
    verts, faces = annulus_grid()

    seams, uvs = transfer(dense_arrays(verts, faces), uncut_proxy())

    assert seams
    assert uv_pieces(uvs, faces) == 1
    assert_no_flips_or_crossings(uvs, faces)


def test_transfer_seams_an_edge_whose_faces_share_a_winding(transfer):
    verts, faces = quad_grid(GRID)
    # the left half is wound the other way
    faces = [
        face[::-1] if max(verts[v][0] for v in face) <= 3 else face for face in faces
    ]

    seams, uvs = transfer(dense_arrays(verts, faces), uncut_proxy())

    assert seams == {(y * SIDE + 3, (y + 1) * SIDE + 3) for y in range(GRID)}
    assert (signed_uv_areas(uvs, faces) > 0).all()


def test_transfer_splits_a_bowtie_vertex(transfer):
    verts = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (-1.0, -1.0, 0.0),
    ]
    faces = [[0, 1, 2], [0, 3, 4]]

    _, uvs = transfer(dense_arrays(verts, faces), uncut_proxy())

    assert not numpy.array_equal(uvs[0], uvs[3])
    assert (signed_uv_areas(uvs, faces) > 0).all()


def test_transfer_survives_a_zero_area_face(transfer):
    verts, faces = quad_grid(GRID)
    faces = triangulated(faces)
    # three points on one line, sharing edges with the grid
    verts = verts + [(0.5, 0.0, 0.0)]
    faces = faces + [[0, len(verts) - 1, 1]]

    _, uvs = transfer(dense_arrays(verts, faces), uncut_proxy())

    assert numpy.isfinite(uvs).all()


def part_left_alone_beside_a_decimated_part(uvs_of_left_alone):
    left_alone_verts, grid_faces = quad_grid(GRID)
    left_alone_faces = triangulated(grid_faces)
    decimated_verts = [(x, y, 20.0) for x, y, _ in left_alone_verts]
    decimated_faces = [
        [v + len(left_alone_verts) for v in face] for face in triangulated(grid_faces)
    ]
    verts = left_alone_verts + decimated_verts
    dense = dense_arrays(verts, left_alone_faces + decimated_faces)

    decimated_proxy_verts = [
        (0.0, 0.0, 20.0),
        (GRID, 0.0, 20.0),
        (GRID, GRID, 20.0),
        (0.0, GRID, 20.0),
    ]
    start = len(left_alone_verts)
    proxy_verts = left_alone_verts + decimated_proxy_verts
    proxy_faces = left_alone_faces + [
        [start, start + 1, start + 2],
        [start, start + 2, start + 3],
    ]
    proxy_uvs = uvs_of_left_alone(left_alone_verts, left_alone_faces) + [
        [(0.0, 0.0), (GRID, 0.0), (GRID, GRID)],
        [(0.0, 0.0), (GRID, GRID), (0.0, GRID)],
    ]
    return dense, proxy_arrays(proxy_verts, proxy_faces, proxy_uvs), left_alone_faces


def sheared_uvs(verts, faces):
    return [
        [(verts[v][0] + 0.4 * verts[v][1], verts[v][1]) for v in face] for face in faces
    ]


def test_transfer_keeps_the_engine_uvs_on_a_part_the_proxy_left_alone(transfer):
    dense, proxy, faces = part_left_alone_beside_a_decimated_part(sheared_uvs)

    seams, uvs = transfer(dense, proxy)

    assert seams == set()
    expected = numpy.array(proxy["corner_uvs"][: len(faces)]).reshape(-1, 2)
    measured = uvs[: len(expected)]
    moved = measured - measured[0]
    target = expected - expected[0]
    scale = numpy.linalg.norm(moved) / numpy.linalg.norm(target)
    numpy.testing.assert_allclose(moved, scale * target, atol=1e-9)


def test_transfer_seams_the_engine_tears_on_a_part_the_proxy_left_alone(transfer):
    def torn_uvs(verts, faces):
        return [
            [
                (
                    verts[v][0] + (SHIFT if sum(verts[w][0] for w in face) < 9 else 0),
                    verts[v][1],
                )
                for v in face
            ]
            for face in faces
        ]

    dense, proxy, faces = part_left_alone_beside_a_decimated_part(torn_uvs)

    seams, uvs = transfer(dense, proxy)

    assert seams == {(y * SIDE + 3, (y + 1) * SIDE + 3) for y in range(GRID)}
    assert uv_pieces(uvs[: 3 * len(faces)], faces) == 2


def test_transfer_keeps_the_engine_uvs_on_painted_faces(transfer):
    verts, faces = quad_grid(GRID)
    painted_columns = 3
    painted = [
        face for index, face in enumerate(faces) if index % GRID < painted_columns
    ]
    used = sorted({vertex for face in painted for vertex in face})
    local = {vertex: index for index, vertex in enumerate(used)}
    proxy_verts = [verts[vertex] for vertex in used]
    proxy_faces = [[local[vertex] for vertex in face] for face in painted]
    proxy_uvs = sheared_uvs(proxy_verts, proxy_faces)
    start = len(proxy_verts)
    proxy_verts += [
        (painted_columns, 0.0, 0.0),
        (GRID, 0.0, 0.0),
        (GRID, GRID, 0.0),
        (painted_columns, GRID, 0.0),
    ]
    proxy_faces += [[start, start + 1, start + 2], [start, start + 2, start + 3]]
    proxy_uvs += [
        [(SHIFT, 0.0), (SHIFT + 3.0, 0.0), (SHIFT + 3.0, GRID)],
        [(SHIFT, 0.0), (SHIFT + 3.0, GRID), (SHIFT, GRID)],
    ]
    proxy = proxy_arrays(proxy_verts, proxy_faces, proxy_uvs)
    proxy["painted_faces"] = [1] * len(painted) + [0, 0]
    dense = dense_arrays(verts, faces)

    seams, uvs = transfer(dense, proxy)

    assert seams == column_seams(painted_columns)
    painted_corners = numpy.array(
        [index % GRID < painted_columns for index in range(len(faces))]
    ).repeat(4)
    positions = numpy.array(verts)[dense["corners"][painted_corners]]
    expected = positions[:, :2] + numpy.outer(positions[:, 1], [0.4, 0.0])
    measured = uvs[painted_corners]
    moved = measured - measured[0]
    target = expected - expected[0]
    scale = numpy.linalg.norm(moved) / numpy.linalg.norm(target)
    numpy.testing.assert_allclose(moved, scale * target, atol=1e-9)


def test_transfer_flattens_a_part_whose_engine_uvs_fold(transfer):
    def folded_uvs(verts, faces):
        uvs = grid_uvs(verts, faces)
        fold_vertex = SIDE + 1
        return [
            [(-3.0, -3.0) if v == fold_vertex else uv for v, uv in zip(face, face_uvs)]
            for face, face_uvs in zip(faces, uvs)
        ]

    dense, proxy, faces = part_left_alone_beside_a_decimated_part(folded_uvs)

    _, uvs = transfer(dense, proxy)

    assert_no_flips_or_crossings(uvs[: 3 * len(faces)], faces)


def test_transfer_flattens_a_chart_large_enough_to_decimate(transfer):
    verts, faces = quad_grid(90)
    verts = [(x, y, 8 * math.sin(x / 15)) for x, y, _ in verts]
    faces = triangulated(faces)

    _, uvs = transfer(dense_arrays(verts, faces), uncut_proxy())

    assert_no_flips_or_crossings(uvs, faces)
    ring = uvs[fan_triangles(faces)]
    positions = numpy.array(verts)[numpy.array(faces)]
    uv_lengths = numpy.linalg.norm(ring[:, 1] - ring[:, 0], axis=1)
    lengths = numpy.linalg.norm(positions[:, 1] - positions[:, 0], axis=1)
    ratio = uv_lengths / lengths
    assert numpy.percentile(ratio, 99) / numpy.median(ratio) < 1.1


def test_transfer_flattens_a_lone_triangle(transfer):
    verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]

    seams, uvs = transfer(dense_arrays(verts, [[0, 1, 2]]), uncut_proxy())

    assert seams == set()
    assert (signed_uv_areas(uvs, [[0, 1, 2]]) > 0).all()
