import time

import numpy
import pytest

from seams.engine_run import POLL_INTERVAL
from seams.mesh import connected_labels, uv_tears
from seams.proxy_transfer import TransferRun, dense_subset

IDENTITY = numpy.eye(4)
GRID = 6
SHIFT = 10.0
# the proxy cut sits between dense columns 2 and 3
CUT_X = 2.5


def quad_grid(size, spacing=1.0):
    side = size + 1
    verts = [(x * spacing, y * spacing, 0.0) for y in range(side) for x in range(side)]
    faces = [
        [y * side + x, y * side + x + 1, (y + 1) * side + x + 1, (y + 1) * side + x]
        for y in range(size)
        for x in range(size)
    ]
    return verts, faces


def triangulated(faces):
    return [
        f
        for quad in faces
        for f in ([quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]])
    ]


def grid_uvs(verts, faces):
    return [[(verts[v][0], verts[v][1]) for v in face] for face in faces]


# def grid_edges(faces):
#     return numpy.array(sorted(face_edges(faces)), dtype=numpy.int64)


def dense_arrays(verts, faces, matrix=IDENTITY):
    return {
        "positions": verts,
        "normals": [(0.0, 0.0, 1.0)] * len(verts),
        "matrix": matrix,
        "corners": numpy.array([v for face in faces for v in face], dtype=numpy.int64),
        "face_sizes": numpy.array([len(face) for face in faces], dtype=numpy.int64),
    }


def proxy_arrays(verts, faces, uvs, matrix=IDENTITY):
    return {"positions": verts, "faces": faces, "corner_uvs": uvs, "matrix": matrix}


def split_proxy(right_corner_uv=None):
    verts = [
        (0.0, 0.0, 0.0),
        (CUT_X, 0.0, 0.0),
        (CUT_X, GRID, 0.0),
        (0.0, GRID, 0.0),
        (GRID, 0.0, 0.0),
        (GRID, GRID, 0.0),
    ]
    faces = [[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]]
    uvs = grid_uvs(verts, faces)
    for f in (0, 1):
        uvs[f] = [(u + SHIFT, v) for u, v in uvs[f]]
    if right_corner_uv is not None:
        uvs[1][2] = right_corner_uv
    return proxy_arrays(verts, faces, uvs)


@pytest.fixture
def start_transfer(engine_binary, tmp_path):
    command = [engine_binary("uvgami-tools")]

    def start(dense, proxy):
        return TransferRun(command, tmp_path, dense, proxy)

    return start


def finished(run):
    while not run.done():
        time.sleep(POLL_INTERVAL)
    return run


@pytest.fixture
def transfer(start_transfer):
    def run(dense, proxy):
        return finished(start_transfer(dense, proxy)).result()

    return run


def test_uv_tears_finds_only_the_torn_interior_edge():
    verts, faces = quad_grid(2)
    uvs = grid_uvs(verts, faces)
    uvs[0] = [(u + SHIFT, v) for u, v in uvs[0]]
    assert uv_tears(faces, uvs) == {(1, 4), (3, 4)}


def test_uv_tears_skips_boundary_edges():
    verts, faces = quad_grid(1)
    uvs = grid_uvs(verts, faces)
    assert uv_tears(faces, uvs) == set()


def test_uv_tears_ignores_a_gap_under_the_tolerance():
    verts, faces = quad_grid(2)
    uvs = grid_uvs(verts, faces)
    uvs[0] = [(u + 1e-7, v) for u, v in uvs[0]]
    assert uv_tears(faces, uvs) == {(1, 4), (3, 4)}
    assert uv_tears(faces, uvs, tolerance=1e-6) == set()


def test_dense_subset_renumbers_the_kept_faces():
    verts, faces = quad_grid(2)
    subset, used = dense_subset(dense_arrays(verts, faces), [3, 0])
    assert used.tolist() == [0, 1, 3, 4, 5, 7, 8]
    assert subset["face_sizes"].tolist() == [4, 4]
    assert subset["corners"].tolist() == [3, 4, 6, 5, 0, 1, 3, 2]
    numpy.testing.assert_array_equal(subset["positions"], numpy.asarray(verts)[used])


# def test_snap_cuts_follows_real_edges():
#     verts, faces = quad_grid(3)
#     edges = grid_edges(faces)
#     mapped = [0, 3]
#     assert snap_cuts(verts, edges, mapped, {(0, 1)}) == {(0, 1), (1, 2), (2, 3)}


@pytest.mark.parametrize("faces_of", [lambda faces: faces, triangulated])
def test_transfer_reads_a_continuous_map_exactly(transfer, faces_of):
    verts, faces = quad_grid(GRID)
    faces = faces_of(faces)
    proxy_verts, proxy_faces = quad_grid(1, GRID)
    proxy_faces = triangulated(proxy_faces)
    proxy = proxy_arrays(proxy_verts, proxy_faces, grid_uvs(proxy_verts, proxy_faces))
    dense = dense_arrays(verts, faces)

    seams, uvs = transfer(dense, proxy)

    expected = numpy.array(verts)[dense["corners"]][:, :2]
    assert numpy.allclose(uvs, expected)
    assert seams == set()


def test_transfer_run_reaches_full_progress(start_transfer):
    verts, faces = quad_grid(GRID)
    proxy = split_proxy()

    run = finished(start_transfer(dense_arrays(verts, faces), proxy))

    assert run.progress == pytest.approx(1.0)
    run.result()


def test_transfer_matches_the_meshes_in_the_proxy_space(transfer):
    verts, faces = quad_grid(GRID)
    offset = numpy.eye(4)
    offset[0, 3] = 5.0
    proxy_verts, proxy_faces = quad_grid(1, GRID)
    proxy_faces = triangulated(proxy_faces)
    proxy_verts = [(x + 5.0, y, z) for x, y, z in proxy_verts]
    proxy = proxy_arrays(proxy_verts, proxy_faces, grid_uvs(proxy_verts, proxy_faces))
    dense = dense_arrays(verts, faces, offset)

    _, uvs = transfer(dense, proxy)

    expected = numpy.array(verts)[dense["corners"]][:, :2] + (5.0, 0.0)
    assert numpy.allclose(uvs, expected)


def test_transfer_reads_the_side_of_a_thin_wall_it_faces(transfer):
    wall = 0.1
    proxy_verts, proxy_faces = quad_grid(1, GRID)
    top = triangulated(proxy_faces)
    bottom = [[v + len(proxy_verts) for v in reversed(face)] for face in top]
    proxy_verts = proxy_verts + [(x, y, -wall) for x, y, _ in proxy_verts]
    uvs = grid_uvs(proxy_verts, top) + [
        [(u + SHIFT, v) for u, v in face] for face in grid_uvs(proxy_verts, bottom)
    ]
    proxy = proxy_arrays(proxy_verts, top + bottom, uvs)
    verts, faces = quad_grid(GRID)
    # the dense underside is nearer the proxy's top
    verts = [(x, y, -0.4 * wall) for x, y, _ in verts]
    dense = dense_arrays(verts, [face[::-1] for face in faces])
    dense["normals"] = [(0.0, 0.0, -1.0)] * len(verts)

    _, uvs = transfer(dense, proxy)

    expected = numpy.array(verts)[dense["corners"]][:, :2] + (SHIFT, 0.0)
    assert numpy.allclose(uvs, expected)


def torn_columns(faces, verts):
    xs = numpy.array([[verts[v][0] for v in face] for face in faces])
    left = numpy.flatnonzero(xs.max(axis=1) <= 2)
    straddling = numpy.flatnonzero((xs.min(axis=1) == 2) & (xs.max(axis=1) == 3))
    right = numpy.flatnonzero(xs.min(axis=1) >= 3)
    return left, straddling, right


def test_transfer_tears_the_dense_mesh_along_the_proxy_cut(transfer):
    verts, faces = quad_grid(GRID)
    proxy = split_proxy()
    dense = dense_arrays(verts, faces)

    seams, uvs = transfer(dense, proxy)

    xy = numpy.array(verts)[dense["corners"]][:, :2]
    corner_face = numpy.repeat(numpy.arange(len(faces)), 4)
    left, straddling, right = torn_columns(faces, verts)
    # a straddling face is drawn on its first corner's side, the shifted one
    shifted = numpy.isin(corner_face, numpy.concatenate([left, straddling]))
    assert numpy.allclose(uvs[shifted], xy[shifted] + (SHIFT, 0.0))
    kept = numpy.isin(corner_face, right)
    assert numpy.allclose(uvs[kept], xy[kept])
    side = GRID + 1
    assert seams == {(y * side + 3, (y + 1) * side + 3) for y in range(GRID)}


# the halves share no vertex, like segments decimated apart, packed a short gap apart
def loose_halves_proxy(gap):
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
    uvs = grid_uvs(verts, faces)
    for f in (0, 1):
        uvs[f] = [(u + gap, v) for u, v in uvs[f]]
    return proxy_arrays(verts, faces, uvs)


def test_transfer_tears_between_loose_proxy_pieces(transfer):
    verts, faces = quad_grid(GRID)
    gap = 1.5
    proxy = loose_halves_proxy(gap)
    dense = dense_arrays(verts, faces)

    seams, uvs = transfer(dense, proxy)

    xy = numpy.array(verts)[dense["corners"]][:, :2]
    corner_face = numpy.repeat(numpy.arange(len(faces)), 4)
    left, straddling, right = torn_columns(faces, verts)
    shifted = numpy.isin(corner_face, numpy.concatenate([left, straddling]))
    assert numpy.allclose(uvs[shifted], xy[shifted] + (gap, 0.0))
    kept = numpy.isin(corner_face, right)
    assert numpy.allclose(uvs[kept], xy[kept])
    side = GRID + 1
    assert seams == {(y * side + 3, (y + 1) * side + 3) for y in range(GRID)}


def test_transfer_never_tears_between_linked_proxy_faces(transfer):
    verts, faces = quad_grid(GRID)
    proxy = split_proxy(right_corner_uv=(SHIFT + 2.0, GRID + 2.0))
    dense = dense_arrays(verts, faces)

    seams, uvs = transfer(dense, proxy)

    side = GRID + 1
    assert seams == {(y * side + 3, (y + 1) * side + 3) for y in range(GRID)}
    _, straddling, _ = torn_columns(faces, verts)
    corners = dense["corners"]
    for v in range(side + 3, GRID * side, side):
        on_straddlers = [
            f * 4 + i for f in straddling for i in range(4) if corners[f * 4 + i] == v
        ]
        assert len(on_straddlers) == 2
        assert numpy.array_equal(uvs[on_straddlers[0]], uvs[on_straddlers[1]])


def test_transfer_pulls_a_zigzag_seam_onto_one_edge_row(transfer):
    verts, faces = quad_grid(GRID)
    faces = triangulated(faces)
    proxy = split_proxy()
    # the cut runs along y instead of x
    proxy["positions"] = [(y, x, z) for x, y, z in proxy["positions"]]
    dense = dense_arrays(verts, faces)

    seams, _ = transfer(dense, proxy)

    side = GRID + 1
    rows = [{(y * side + x, y * side + x + 1) for x in range(GRID)} for y in (2, 3)]
    assert seams in rows


def test_transfer_straightens_a_staircase_onto_the_diagonals(transfer):
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
    dense = dense_arrays(verts, faces)

    seams, _ = transfer(dense, proxy)

    side = GRID + 1

    def diagonal(a, b):
        return b - a == side + 1

    off_diagonal = [edge for edge in seams if not diagonal(*edge)]
    # the run's two ends sit on the grid boundary and may keep a step each
    assert len(off_diagonal) <= 2, off_diagonal
    assert len(seams) - len(off_diagonal) >= GRID - 1


# the face folded over the panel, z = 0.5 - x / 16
def folded_proxy():
    verts = [
        (0.0, 0.0, 0.0),
        (8.0, 0.0, 0.0),
        (0.0, 8.0, 0.0),
        (0.0, -0.5, 0.5),
        (0.0, 6.0, 0.5),
    ]
    faces = [[0, 1, 2], [1, 0, 3], [1, 3, 4]]
    uvs = grid_uvs(verts, faces)
    uvs[0] = [(u + SHIFT, v) for u, v in uvs[0]]
    return proxy_arrays(verts, faces, uvs)


def test_transfer_tears_between_faces_split_at_a_shared_vertex(transfer):
    proxy = folded_proxy()
    columns, rows = 6, [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    verts = [(1.0 + x, y, 0.25) for y in rows for x in range(columns)]
    faces = [
        [
            r * columns + x,
            r * columns + x + 1,
            (r + 1) * columns + x + 1,
            (r + 1) * columns + x,
        ]
        for r in range(len(rows) - 1)
        for x in range(columns - 1)
    ]
    dense = dense_arrays(verts, faces)
    # the folded face faces down, so the rows with downward normals read it
    dense["normals"] = [(0.0, 0.0, 1.0 if y < 0.6 else -1.0) for _, y, _ in verts]

    seams, uvs = transfer(dense, proxy)

    face_u = uvs[:, 0].reshape(len(faces), 4)
    assert numpy.all(face_u.max(axis=1) - face_u.min(axis=1) < 2.0)
    assert seams == {(2 * columns + x, 2 * columns + x + 1) for x in range(columns - 1)}


def test_connected_labels_separates_loose_parts():
    edges = numpy.array(
        [[0, 1], [1, 2], [3, 4], [5, 6], [6, 7], [5, 7]], dtype=numpy.int64
    )
    labels = connected_labels(9, edges)
    assert labels.tolist() == [0, 0, 0, 3, 3, 5, 5, 5, 8]


def test_connected_labels_joins_a_long_chain():
    count = 1000
    ends = numpy.arange(count - 1, dtype=numpy.int64)
    edges = numpy.stack([ends, ends + 1], axis=1)
    assert connected_labels(count, edges).tolist() == [0] * count
