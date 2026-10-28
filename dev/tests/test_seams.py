import collections
import math
import subprocess
import tempfile
from pathlib import Path

import numpy
import pytest

from seams import (
    SURFACE_MODE_CODES,
    face_edges,
    flatten_distortion,
    island_groups,
    island_layout,
    islands_overlap,
    pair,
    restriction_weights,
    signed_area,
    uv_area_fit,
    uv_fit,
    uv_island_groups,
    uv_tears,
    uvs_collapsed,
)
from seams.cuts import VISIBILITY_SHARE
from seams.engine_run import write_obj
from seams.island_repair import finish_uvs, rectify_plans, split_moves
from seams.preseed import _write_weights, resolved_surface_modes
from seams.segmentation import classify_job


# the hard surface exe's pick per face with every part on Auto
@pytest.fixture
def classify_hard(engine_binary, tmp_path):
    command = [engine_binary("uvgami-tools")]

    def run(verts, faces):
        modes = numpy.full(len(faces), SURFACE_MODE_CODES["AUTO"])
        resolved = resolved_surface_modes(command, tmp_path, verts, faces, modes)
        return resolved == SURFACE_MODE_CODES["HARD"]

    return run


# the hard surface exe's placed seams as (low, high) vertex pairs
@pytest.fixture
def exe_seams(engine_binary):
    tools = engine_binary("uvgami-tools")

    def run(verts, faces, angle=30, weights=None):
        with tempfile.TemporaryDirectory() as folder:
            obj = Path(folder) / "mesh.obj"
            write_obj(obj, verts, faces)
            _write_weights(Path(folder) / "mesh_weights", weights)
            out = Path(folder) / "seams.txt"
            subprocess.run(
                [tools, "seams", obj, out, "--angle", str(angle)],
                check=True,
                capture_output=True,
            )
            return {
                tuple(int(v) for v in line.split())
                for line in out.read_text().splitlines()
            }

    return run


# the split pass over the whole mesh as one range unless ranges are given
@pytest.fixture
def split(engine_binary, tmp_path):
    command = [engine_binary("uvgami-tools")]

    def run(verts, faces, uvs, ranges=None):
        if ranges is None:
            ranges = [(0, len(faces))]
        return split_moves(command, tmp_path, verts, faces, uvs, ranges)

    return run


# plans for the islands of the uv map
@pytest.fixture
def rectify(engine_binary, tmp_path):
    command = [engine_binary("uvgami-tools")]

    def run(faces, uvs):
        return rectify_plans(command, tmp_path, uvs, uv_groups(faces, uvs))

    return run


# the finished uv map, the whole mesh as one range
@pytest.fixture
def finish(engine_binary, tmp_path):
    command = [engine_binary("uvgami-tools")]

    def run(verts, faces, uvs):
        finished = finish_uvs(command, tmp_path, verts, faces, uvs, [(0, len(faces))])
        result = [list(corners) for corners in uvs]
        changes = zip(
            finished.faces.tolist(), finished.corners.tolist(), finished.uvs.tolist()
        )
        for face, corner, uv in changes:
            result[face][corner] = tuple(uv)
        return result

    return run


@pytest.mark.smoke
def test_seam_edges_on_plain_cube(exe_seams, fixture_mesh):
    # every cube edge is a crease, so each face is its own island
    verts, faces = fixture_mesh("cube")
    seams = exe_seams(verts, faces)
    assert len(seams) == 12
    assert all(v0 < v1 for v0, v1 in seams)
    assert len(island_groups(faces, seams, face_edges(faces))) == 6


def tube(sides=4, height=1.0):
    verts, faces = [], []
    for i in range(sides):
        angle = 2 * math.pi * i / sides
        verts.append([math.cos(angle), math.sin(angle), 0.0])
        verts.append([math.cos(angle), math.sin(angle), height])
    for i in range(sides):
        low, high = 2 * i, 2 * i + 1
        next_low, next_high = 2 * ((i + 1) % sides), 2 * ((i + 1) % sides) + 1
        faces.append([low, next_low, high])
        faces.append([next_low, next_high, high])
    return verts, faces


@pytest.mark.smoke
def test_coarse_tube_unrolls_with_one_cut(exe_seams):
    # the wall merges into one annulus, which gets one lengthwise opening
    verts, faces = tube(16)
    seams = exe_seams(verts, faces)
    assert len(seams) == 1
    ((v0, v1),) = seams
    assert {verts[v0][2], verts[v1][2]} == {0.0, 1.0}


@pytest.mark.smoke
def test_tall_coarse_tube_still_gets_one_cut(exe_seams):
    # the wall's 22.5 degree steps merge away whatever its height
    verts, faces = tube(16, height=2.0)
    seams = exe_seams(verts, faces)
    assert len(seams) == 1
    ((v0, v1),) = seams
    assert {verts[v0][2], verts[v1][2]} == {0.0, 2.0}


@pytest.mark.smoke
def test_faceted_tube_is_cut_on_every_facet_edge(exe_seams):
    # 45 degrees a segment is a crease, so each flat facet is its own island
    verts, faces = tube(8)
    seams = exe_seams(verts, faces)
    assert len(seams) == 8
    assert len(island_groups(faces, seams, face_edges(faces))) == 8


@pytest.mark.smoke
def test_lower_feature_angle_keeps_shallow_seams(exe_seams):
    # 22.5 degree panel boundaries merge away at the default 30 but survive 15
    verts, faces = elbow(sides=16)
    assert len(exe_seams(verts, faces, angle=15)) > len(exe_seams(verts, faces))


@pytest.mark.smoke
def test_beveled_cube_is_cut_into_its_six_faces(exe_seams, fixture_mesh):
    # each bevel band dissolves into one neighbour, so every cut sits on one crease
    verts, faces = fixture_mesh("cube-bevel2")
    seams = exe_seams(verts, faces)
    edges = face_edges(faces)
    assert len(island_groups(faces, seams, edges)) == 6


def capped_tube(sides=32, height=2.0):
    verts, faces = tube(sides, height)
    top = len(verts)
    verts.append([0.0, 0.0, height])
    for i in range(sides):
        faces.append([2 * i + 1, 2 * ((i + 1) % sides) + 1, top])
    return verts, faces


def elbow(rings=12, sides=12, bend_radius=3.0, tube_radius=1.0):
    verts, faces = [], []
    for i in range(rings + 1):
        t = (math.pi / 2) * i / rings
        for j in range(sides):
            p = 2 * math.pi * j / sides
            spoke = bend_radius + tube_radius * math.cos(p)
            verts.append(
                [spoke * math.cos(t), tube_radius * math.sin(p), spoke * math.sin(t)]
            )
    for i in range(rings):
        for j in range(sides):
            a = i * sides + j
            b = i * sides + (j + 1) % sides
            c = (i + 1) * sides + (j + 1) % sides
            d = (i + 1) * sides + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    return verts, faces


# the coarse tube's one cut is a side edge, and every other side edge is as long
@pytest.mark.smoke
def test_painted_restriction_moves_the_cut(exe_seams):
    verts, faces = tube(16)
    plain = exe_seams(verts, faces)
    painted = {v: 1.0 for edge in plain for v in edge}

    cuts = exe_seams(verts, faces, weights=painted)
    assert len(cuts) == 1
    assert not painted.keys() & {v for edge in cuts for v in edge}


# painting everything leaves the tube closed if the cut is dropped
@pytest.mark.smoke
def test_paint_cannot_block_a_cut_that_has_to_happen(exe_seams):
    verts, faces = tube(16)
    painted = {v: 1.0 for v in range(len(verts))}
    assert len(exe_seams(verts, faces, weights=painted)) == 1


def test_restriction_weights_take_the_larger_of_paint_and_exposure():
    exposure = {0: 1.0, 1: 0.5, 2: 0.0}
    painted = {1: 1.0, 3: 0.2}
    assert restriction_weights(painted, exposure) == {
        0: VISIBILITY_SHARE,
        1: 1.0,
        3: 0.2,
    }
    assert restriction_weights(None, {2: 0.0}) is None


def test_weights_sidecar_lists_vertex_weight_pairs(tmp_path):
    path = tmp_path / "mesh_weights"
    _write_weights(path, {3: numpy.float32(0.5), 7: 1})
    assert path.read_text() == "3,0.5,7,1.0\n"

    # no weights leaves the engine without a sidecar
    _write_weights(path.with_name("none_weights"), None)
    assert not path.with_name("none_weights").exists()


def strip_island(quads, scale=1.0, angle=0.0):
    points, faces = [], []
    cos, sin = math.cos(angle), math.sin(angle)
    for i in range(quads + 1):
        for y in (0.0, 1.0):
            x = float(i)
            points.append((scale * (x * cos - y * sin), scale * (x * sin + y * cos)))
    for i in range(quads):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        faces.append([a, c, b])
        faces.append([c, d, b])
    uvs = [[points[v] for v in f] for f in faces]
    verts = [(x, y, 0.0) for x, y in points]
    return verts, faces, uvs


def island_count(faces, seams):
    parent = list(range(len(faces)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owners = {}
    for fi, f in enumerate(faces):
        for i in range(len(f)):
            owners.setdefault(pair(f[i], f[(i + 1) % len(f)]), []).append(fi)
    for key, fs in owners.items():
        if len(fs) == 2 and key not in seams:
            a, b = find(fs[0]), find(fs[1])
            if a != b:
                parent[a] = b
    return len({find(i) for i in range(len(faces))})


def merge_islands(a, b):
    verts_a, faces_a, uvs_a = a
    verts_b, faces_b, uvs_b = b
    shift = len(verts_a)
    faces = faces_a + [[v + shift for v in f] for f in faces_b]
    return verts_a + verts_b, faces, uvs_a + uvs_b


@pytest.mark.smoke
def test_clean_long_island_splits_for_packing(split):
    # flip-free, but far longer than the square its own uv area needs
    verts, faces, uvs = strip_island(30)
    cuts = split(verts, faces, uvs).cuts
    assert island_count(faces, cuts) == 4
    # each cut is one rung straight across the strip
    assert len(cuts) == 3
    assert all(verts[a][0] == verts[b][0] for a, b in cuts)


@pytest.mark.smoke
def test_clean_short_strip_is_not_split(split):
    # a thin strip next to a much bigger island, the kind a bevel band leaves
    verts, faces, uvs = merge_islands(
        strip_island(30, scale=1 / 200), strip_island(2, scale=10.0)
    )
    assert split(verts, faces, uvs).cuts == set()


@pytest.mark.smoke
def test_each_range_splits_at_its_own_scale(split):
    # the strip is under half the cap the bigger island sets, but past its own
    verts, faces, uvs = merge_islands(strip_island(30), strip_island(2, scale=40.0))
    assert split(verts, faces, uvs).cuts == set()
    cuts = split(verts, faces, uvs, [(0, 60), (60, len(faces))]).cuts
    assert island_count(faces, cuts) == 5


@pytest.mark.smoke
def test_clean_compact_island_is_not_split(split):
    verts, faces, uvs = strip_island(4)
    assert split(verts, faces, uvs).cuts == set()


@pytest.mark.smoke
def test_split_moves_part_the_sliced_strips(split):
    # the whole scan on plain data, the long strip slices into 4
    verts, faces, uvs = strip_island(30)
    edges = face_edges(faces)
    assert len(uv_island_groups(faces, uvs, edges)) == 1

    result = split(verts, faces, uvs)
    assert len(result.moved_faces)
    moved = [list(corners) for corners in uvs]
    moves = zip(
        result.moved_faces.tolist(),
        result.moved_corners.tolist(),
        result.moved_uvs.tolist(),
    )
    for face_index, corner, uv in moves:
        moved[face_index][corner] = tuple(uv)
    after = uv_island_groups(faces, moved, edges)
    assert len(after) == 4
    assert uv_tears(faces, uvs) | result.cuts == uv_tears(faces, moved)
    assert sorted(result.islands) == sorted(after)


# the engine re-cuts a ruined island itself
@pytest.mark.smoke
def test_folded_island_is_left_to_the_engine(split):
    verts, faces, uvs = strip_island(30)
    assert split(verts, faces, uvs).cuts
    # the top vertex at x 20 drops below the strip, every corner of it at once
    for face, corners in zip(faces, uvs):
        for i, v in enumerate(face):
            if v == 41:
                corners[i] = (20.0, -1.0)
    result = split(verts, faces, uvs)
    assert result.cuts == set()
    assert not len(result.moved_faces)


@pytest.mark.smoke
def test_island_with_a_hole_is_left_to_the_engine(split):
    verts, faces, uvs = grid_island(30, 3)
    assert split(verts, faces, uvs).cuts
    hole = [faces.index([47, 48, 79]), faces.index([47, 79, 78])]
    for fi in sorted(hole, reverse=True):
        del faces[fi]
        del uvs[fi]
    assert split(verts, faces, uvs).cuts == set()


def tube_island(rows, cols, shear):
    radius = cols / (2 * math.pi)
    verts, row_of = [], []
    for r in range(rows + 1):
        for c in range(cols):
            a = 2 * math.pi * c / cols
            verts.append((radius * math.cos(a), radius * math.sin(a), float(r)))
            row_of.append(r)
    faces, uvs = [], []
    for r in range(rows):
        for c in range(cols):
            lo = r * cols
            hi = (r + 1) * cols
            a, b = lo + c, lo + (c + 1) % cols
            d, e = hi + c, hi + (c + 1) % cols
            # the wrapping column keeps counting in uv, which opens the cut
            far = c + 1
            uv = {
                a: (float(c), r + shear * c),
                b: (float(far), r + shear * far),
                d: (float(c), r + 1 + shear * c),
                e: (float(far), r + 1 + shear * far),
            }
            faces.append([a, b, d])
            uvs.append([uv[a], uv[b], uv[d]])
            faces.append([b, e, d])
            uvs.append([uv[b], uv[e], uv[d]])
    seams = {pair(r * cols, (r + 1) * cols) for r in range(rows)}
    return verts, faces, uvs, seams, row_of


@pytest.mark.smoke
def test_sheared_tube_splits_into_rings_not_slivers(split):
    # 40 rows of 6 unit quads, sheared so the seam's two sides sit 3 rows apart
    verts, faces, uvs, seams, row_of = tube_island(40, 6, 0.5)
    extra = split(verts, faces, uvs).cuts
    assert island_count(faces, seams | extra) == 2

    groups = island_groups(faces, seams | extra, face_edges(faces))
    assert min(len(g) for g in groups) >= len(faces) // 4

    # a clean ring: the cut stays within two quad rows
    rows_cut = [row_of[v] for edge in extra for v in edge]
    assert max(rows_cut) - min(rows_cut) <= 2


def cone_island(rows, cols, sector, inner):
    k = sector / (2 * math.pi)
    rise = math.sqrt(1 - k * k)
    verts, col_of = [], []
    for r in range(rows + 1):
        s = inner + r
        for c in range(cols):
            a = 2 * math.pi * c / cols
            verts.append((k * s * math.cos(a), k * s * math.sin(a), rise * s))
            col_of.append(c)

    def unrolled(r, c):
        t = sector * c / cols
        return ((inner + r) * math.cos(t), (inner + r) * math.sin(t))

    faces, uvs = [], []
    for r in range(rows):
        for c in range(cols):
            lo = r * cols
            hi = (r + 1) * cols
            a, b = lo + c, lo + (c + 1) % cols
            d, e = hi + c, hi + (c + 1) % cols
            # the wrapping column keeps counting in uv, which opens the cut
            far = c + 1
            uv = {
                a: unrolled(r, c),
                b: unrolled(r, far),
                d: unrolled(r + 1, c),
                e: unrolled(r + 1, far),
            }
            faces.append([a, b, d])
            uvs.append([uv[a], uv[b], uv[d]])
            faces.append([b, e, d])
            uvs.append([uv[b], uv[e], uv[d]])
    seams = {pair(r * cols, (r + 1) * cols) for r in range(rows)}
    return verts, faces, uvs, seams, col_of


@pytest.mark.smoke
def test_cone_fan_is_cut_along_radii(split):
    # a cone frustum unrolls into a 300 degree fan of uv area 199 against a cap of 20
    verts, faces, uvs, seams, col_of = cone_island(4, 48, 5 * math.pi / 3, 7.5)
    extra = split(verts, faces, uvs).cuts
    assert island_count(faces, seams | extra) == 3

    remaining = set(extra)
    spans = []
    while remaining:
        chain = {remaining.pop()}
        grew = True
        while grew:
            grew = False
            for other in list(remaining):
                if any(set(other) & set(edge) for edge in chain):
                    remaining.discard(other)
                    chain.add(other)
                    grew = True
        cols_cut = [col_of[v] for edge in chain for v in edge]
        spans.append(max(cols_cut) - min(cols_cut))
    assert len(spans) == 2
    assert max(spans) <= 1


def blob_strip_island(blob, strip):
    index = {}
    verts = []

    def vid(x, y):
        key = (x, y)
        if key not in index:
            index[key] = len(verts)
            verts.append((float(x), float(y), 0.0))
        return index[key]

    faces = []
    for x in range(blob):
        for y in range(blob):
            faces.append([vid(x, y), vid(x + 1, y), vid(x + 1, y + 1), vid(x, y + 1)])
    for x in range(blob, blob + strip):
        faces.append([vid(x, 0), vid(x + 1, 0), vid(x + 1, 1), vid(x, 1)])
    uvs = [[(verts[v][0], verts[v][1]) for v in f] for f in faces]
    return verts, faces, uvs


@pytest.mark.smoke
def test_long_island_cut_at_the_feature_neck(split):
    # a 20x20 blob with a 20x1 tail, the width step at the join is ~20x
    verts, faces, uvs = blob_strip_island(20, 20)
    extra = split(verts, faces, uvs).cuts
    groups = sorted(island_groups(faces, extra, face_edges(faces)), key=len)
    assert len(groups) == 2
    # the small piece is the tail, give or take a slab of blob columns
    assert 15 <= len(groups[0]) <= 60


@pytest.mark.smoke
def test_neck_scan_needs_a_long_island(split):
    # the same blob and tail next to a much larger island, now under half the cap
    verts, faces, uvs = merge_islands(
        blob_strip_island(20, 20), strip_island(1, scale=60.0)
    )
    assert split(verts, faces, uvs).cuts == set()


@pytest.mark.smoke
def test_neck_and_fill_compose(split):
    # a 45 long tail, the tail piece alone is still past the cap
    verts, faces, uvs = blob_strip_island(20, 45)
    extra = split(verts, faces, uvs).cuts
    assert island_count(faces, extra) == 3


@pytest.mark.smoke
def test_second_neck_found_on_the_piece(split):
    # a second 12x12 blob on the far end of the tail
    verts, faces, uvs = blob_strip_island(20, 20)

    index = {(x, y): i for i, (x, y, _) in enumerate(verts)}

    def vid(x, y):
        key = (x, y)
        if key not in index:
            index[key] = len(verts)
            verts.append((float(x), float(y), 0.0))
        return index[key]

    for x in range(40, 52):
        for y in range(12):
            faces.append([vid(x, y), vid(x + 1, y), vid(x + 1, y + 1), vid(x, y + 1)])
    uvs = [[(verts[v][0], verts[v][1]) for v in f] for f in faces]
    extra = split(verts, faces, uvs).cuts
    groups = sorted(island_groups(faces, extra, face_edges(faces)), key=len)
    assert [len(g) for g in groups] == [20, 144, 400]


# an island belongs to the range holding its lowest face
@pytest.mark.smoke
def test_split_scans_only_islands_in_a_range(split):
    verts, faces, uvs = strip_island(30)
    assert split(verts, faces, uvs, []).cuts == set()
    assert split(verts, faces, uvs, [(1, len(faces))]).cuts == set()
    assert split(verts, faces, uvs, [(0, 1)]).cuts


@pytest.mark.smoke
def test_split_respects_existing_seams(split):
    # a uv tear already cuts a 60 long strip in half
    verts, faces, uvs = strip_island(60)
    for fi in range(60, 120):
        uvs[fi] = [(u + 5.0, v) for u, v in uvs[fi]]
    mid = pair(60, 61)
    assert uv_tears(faces, uvs) == {mid}
    extra = split(verts, faces, uvs).cuts
    assert mid not in extra
    assert island_count(faces, extra | {mid}) == 6


def test_uv_islands_follow_the_uv_map():
    # a flat strip is one island until its uvs split mid-column
    verts, faces, uvs = strip_island(4)
    edges = face_edges(faces)
    assert len(uv_island_groups(faces, uvs, edges)) == 1
    for fi in range(4, 8):
        uvs[fi] = [(u + 5.0, v) for u, v in uvs[fi]]
    groups = uv_island_groups(faces, uvs, edges)
    assert sorted(len(g) for g in groups) == [4, 4]


# def vertex_adjacency(faces):
#     adjacent = collections.defaultdict(set)
#     for a, b in face_edges(faces):
#         adjacent[a].add(b)
#         adjacent[b].add(a)
#     return adjacent


# def test_snap_paths_redraws_a_cut_on_the_dense_mesh():
#     # a two segment cut whose three corners land on grid vertices
#     verts, faces, _ = grid_island(4, 4)
#     mapped = [0, 2, 12]
#     paths = snap_paths(verts, vertex_adjacency(faces), mapped, {(0, 1), (1, 2)})
#     assert paths == {(0, 1), (1, 2), (2, 7), (7, 12)}


# def test_snap_paths_drops_a_cut_with_nowhere_to_go():
#     # both ends on one vertex, then an end nothing connects to
#     verts, faces, _ = grid_island(4, 4)
#     adjacent = vertex_adjacency(faces)
#     mapped = [3, 3, len(verts)]
#     assert snap_paths(verts, adjacent, mapped, {(0, 1)}) == set()
#     assert snap_paths(verts, adjacent, mapped, {(0, 2)}) == set()


# def test_snap_paths_drops_a_cut_between_loose_parts():
#     # two copies of the grid side by side, a cut with an end on each
#     verts, faces, _ = grid_island(4, 4)
#     offset = len(verts)
#     verts = verts + [(x + 10.0, y, z) for x, y, z in verts]
#     faces = faces + [[v + offset for v in face] for face in faces]
#     mapped = [0, offset + 2, 2]
#     adjacent = vertex_adjacency(faces)
#     assert snap_paths(verts, adjacent, mapped, {(0, 1)}) == set()
#     assert snap_paths(verts, adjacent, mapped, {(0, 2)}) == {(0, 1), (1, 2)}


def test_uv_fit_scales_into_old_bounds():
    move = uv_fit([(0, 0), (2, 1)], (10, 10, 11, 10.5))
    assert move((0, 0)) == (10, 10)
    assert move((2, 1)) == (11, 10.5)
    # aspect mismatch keeps the scale uniform and the island inside
    move = uv_fit([(0, 0), (2, 1)], (0, 0, 1, 1))
    assert move((0, 0)) == (0, 0.25)
    assert move((2, 1)) == (1, 0.75)


def test_uv_area_fit_keeps_the_old_uv_area():
    # two charts packed in a square, the island they came from was a 4:1 strip
    charts = [
        [(0, 0), (1, 0), (1, 0.4), (0, 0.4)],
        [(0, 0.6), (1, 0.6), (1, 1), (0, 1)],
    ]
    move = uv_area_fit(charts, 0.8, (0, 0, 0.8, 0.2))
    moved = [[move(uv) for uv in chart] for chart in charts]
    area = sum(abs(signed_area(chart)) for chart in moved)
    assert math.isclose(area, 0.8)
    # centered on the old spot, and a bbox fit would have shrunk it instead
    xs = [u for chart in moved for u, _ in chart]
    ys = [v for chart in moved for _, v in chart]
    assert math.isclose((min(xs) + max(xs)) / 2, 0.4)
    assert math.isclose((min(ys) + max(ys)) / 2, 0.1)
    assert max(ys) - min(ys) > 0.2


def test_uv_area_fit_falls_back_to_the_bbox_without_an_area():
    square = [[(0, 0), (2, 0), (2, 2), (0, 2)]]
    move = uv_area_fit(square, 0, (0, 0, 1, 1))
    assert move((0, 0)) == (0, 0)
    assert move((2, 2)) == (1, 1)


def test_islands_overlap_detects_stacked_boxes():
    apart = [(0, 0, 1, 1), (1.1, 0, 2, 1), (0, 1.2, 1, 2)]
    assert not islands_overlap(apart)
    assert islands_overlap([*apart, (0.5, 0.5, 1.5, 1.5)])


def test_island_layout_separates_and_unmirrors():
    boxes = [(0, 0, 1, 1), (0.2, 0.1, 0.8, 0.9)]
    transforms = island_layout(boxes, [0.5, -0.3])
    moved = []
    for (x0, y0, x1, y1), (flip, du, dv) in zip(boxes, transforms):
        us = [x0, x1] if flip is None else [flip - x0, flip - x1]
        moved.append((min(us) + du, y0 + dv, max(us) + du, y1 + dv))
    assert not islands_overlap(moved)
    # only the negative-area island mirrors, within its own bounds
    assert transforms[0][0] is None
    assert transforms[1][0] == 1.0
    assert moved[1][2] - moved[1][0] == boxes[1][2] - boxes[1][0]


def grid_island(cols, rows):
    verts = [
        (float(x), float(y), 0.0) for y in range(rows + 1) for x in range(cols + 1)
    ]
    faces = []
    for cy in range(rows):
        for cx in range(cols):
            a = cy * (cols + 1) + cx
            b, c, d = a + 1, a + cols + 2, a + cols + 1
            faces.append([a, b, c])
            faces.append([a, c, d])
    uvs = [[(verts[v][0], verts[v][1]) for v in f] for f in faces]
    return verts, faces, uvs


def annulus_island():
    verts, faces, uvs = grid_island(3, 3)
    hole = [faces.index([5, 6, 10]), faces.index([5, 10, 9])]
    for fi in sorted(hole, reverse=True):
        del faces[fi]
        del uvs[fi]
    return verts, faces, uvs


def sphere(rings=12, sides=24):
    verts, faces = [], []
    for i in range(1, rings):
        t = math.pi * i / rings
        for j in range(sides):
            p = 2 * math.pi * j / sides
            verts.append(
                [math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)]
            )
    bottom = len(verts)
    verts.append([0.0, 0.0, 1.0])
    verts.append([0.0, 0.0, -1.0])
    for j in range(sides):
        faces.append([bottom, (j + 1) % sides, j])
        base = (rings - 2) * sides
        faces.append([bottom + 1, base + j, base + (j + 1) % sides])
    for i in range(rings - 2):
        for j in range(sides):
            a = i * sides + j
            b = i * sides + (j + 1) % sides
            c = (i + 1) * sides + (j + 1) % sides
            d = (i + 1) * sides + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    return verts, faces


@pytest.mark.smoke
def test_beveled_cube_reads_hard(fixture_mesh, classify_hard):
    verts, faces = fixture_mesh("cube-bevel2")
    assert classify_hard(verts, faces).all()


def test_beveled_cube_with_float_noise_reads_hard(fixture_mesh, classify_hard):
    # blender's float32 positions have noise of this size
    verts, faces = fixture_mesh("cube-bevel2")
    noise = numpy.random.default_rng(1).normal(0, 1e-7, (len(verts), 3))
    assert classify_hard((numpy.asarray(verts) + noise).tolist(), faces).all()


def lumpy_blob(rings=24, sides=48):
    verts, faces = [], []
    for i in range(rings + 1):
        t = math.pi * i / rings
        for j in range(sides):
            p = 2 * math.pi * j / sides
            x, y, z = math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)
            r = (
                1
                + 0.25 * math.sin(3 * x + 0.5) * math.cos(2 * y + 1.3)
                + 0.2 * math.sin(2.5 * z + 0.7)
            )
            verts.append([r * x, 1.4 * r * y, r * z])
    for i in range(rings):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            faces.append([a, b, b + sides, a + sides])
    return verts, faces


@pytest.mark.smoke
@pytest.mark.parametrize("rings, sides", [(24, 48), (12, 24)])
def test_lumpy_blob_reads_organic(rings, sides, classify_hard):
    verts, faces = lumpy_blob(rings, sides)
    assert not classify_hard(verts, faces).any()


# an auto part takes the classifier's pick, a part set by hand keeps its mode
@pytest.mark.smoke
def test_classify_resolves_only_auto_parts(fixture_mesh, engine_binary, tmp_path):
    cube = fixture_mesh("cube-bevel2")
    blob = lumpy_blob()
    parts = [
        (cube, "AUTO", "HARD"),
        (blob, "AUTO", "ORGANIC"),
        (cube, "ORGANIC", "ORGANIC"),
        (blob, "HARD", "HARD"),
    ]
    positions, corners, totals, modes, expected = [], [], [], [], []
    for index, ((verts, faces), mode, resolved) in enumerate(parts):
        base = len(positions)
        positions += [[x + 5.0 * index, y, z] for x, y, z in verts]
        corners += [base + v for face in faces for v in face]
        totals += [len(face) for face in faces]
        modes += [SURFACE_MODE_CODES[mode]] * len(faces)
        expected += [SURFACE_MODE_CODES[resolved]] * len(faces)

    result = classify_job(
        [engine_binary("uvgami-tools")],
        tmp_path,
        numpy.array(positions),
        numpy.array(corners, dtype=numpy.int64),
        numpy.array(totals, dtype=numpy.int64),
        numpy.array(modes),
    )

    assert result.tolist() == expected


# the blob is under a hundredth of the model's area beside a 100 by 100 quad
@pytest.mark.smoke
def test_tiny_part_reads_hard(classify_hard):
    verts, faces = lumpy_blob()
    blob_faces = len(faces)
    base = len(verts)
    verts += [
        [50.0, 0.0, 0.0],
        [150.0, 0.0, 0.0],
        [150.0, 100.0, 0.0],
        [50.0, 100.0, 0.0],
    ]
    faces.append([base, base + 1, base + 2, base + 3])
    assert classify_hard(verts, faces)[:blob_faces].all()


@pytest.mark.smoke
def test_smooth_cylinder_reads_hard(classify_hard):
    verts, faces = capped_tube(sides=48)
    assert classify_hard(verts, faces).all()


def ngon_capped_cylinder(sides=32, height=2.0):
    verts, faces = [], []
    for i in range(sides):
        angle = 2 * math.pi * i / sides
        verts.append([math.cos(angle), math.sin(angle), 0.0])
        verts.append([math.cos(angle), math.sin(angle), height])
    for i in range(sides):
        low, high = 2 * i, 2 * i + 1
        next_low = 2 * ((i + 1) % sides)
        faces.append([low, next_low, next_low + 1, high])
    faces.append([2 * i for i in reversed(range(sides))])
    faces.append([2 * i + 1 for i in range(sides)])
    return verts, faces


@pytest.mark.smoke
def test_default_cylinder_reads_hard(classify_hard):
    verts, faces = ngon_capped_cylinder()
    assert classify_hard(verts, faces).all()


@pytest.mark.smoke
def test_default_cylinder_cuts_rims_and_one_seam(exe_seams):
    # both rims plus a single axial cut, the wall as one island and each cap its own
    verts, faces = ngon_capped_cylinder()
    seams = exe_seams(verts, faces)
    groups = island_groups(faces, seams, face_edges(faces))
    assert sorted(len(g) for g in groups) == [1, 1, 32]


def uv_groups(faces, uvs):
    return uv_island_groups(faces, uvs, face_edges(faces))


# a uv edge only one face uses is on the outline
def outline_points(uvs):
    counts = collections.Counter()
    for corners in uvs:
        for a, b in zip(corners, corners[1:] + corners[:1]):
            counts[frozenset((a, b))] += 1
    return {point for edge, count in counts.items() if count == 1 for point in edge}


def target_box(targets):
    xs = [p[0] for p in targets.values()]
    ys = [p[1] for p in targets.values()]
    return min(xs), min(ys), max(xs), max(ys)


def on_perimeter(point, box, tol=1e-6):
    x0, y0, x1, y1 = box
    x, y = point
    inside = x0 - tol <= x <= x1 + tol and y0 - tol <= y <= y1 + tol
    edge = min(abs(x - x0), abs(x - x1), abs(y - y0), abs(y - y1)) < tol
    return inside and edge


@pytest.mark.smoke
def test_rectify_straightens_a_wavy_strip(rectify):
    verts, faces, uvs = grid_island(8, 1)

    def wobble(uv):
        x, y = uv
        lean = 0.05 if int(x) % 2 else -0.05
        return (x, y + lean) if y in (0.0, 1.0) else uv

    uvs = [[wobble(uv) for uv in face] for face in uvs]
    plans = rectify(faces, uvs)
    assert len(plans) == 1
    plan = plans[0]
    assert sorted(plan.group) == list(range(len(faces)))
    box = target_box(plan.targets)
    assert all(on_perimeter(t, box) for t in plan.targets.values())
    corners = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
    hits = sum(
        1 for c in corners if any(math.dist(c, t) < 1e-6 for t in plan.targets.values())
    )
    assert hits == 4
    # the wavy long sides actually moved onto the straight edges
    assert any(math.dist(uvs[f][c], t) > 0.01 for (f, c), t in plan.targets.items())


def arc_island(radius=2.0, width=0.5, span=240, segments=24, rows=2):
    points = []
    for k in range(segments + 1):
        a = math.radians(span) * k / segments
        for r in range(rows + 1):
            rad = radius - width / 2 + width * r / rows
            points.append((rad * math.cos(a), rad * math.sin(a)))
    faces, uvs = [], []
    for k in range(segments):
        for r in range(rows):
            base = k * (rows + 1) + r
            quad = (base, base + 1, base + rows + 2, base + rows + 1)
            faces.append(quad)
            uvs.append([points[v] for v in quad])
    return faces, uvs


@pytest.mark.smoke
def test_rectify_straightens_a_curled_strip(rectify):
    radius, width, span, segments = 2.0, 0.5, 240, 24
    faces, uvs = arc_island(radius, width, span, segments)
    plans = rectify(faces, uvs)
    assert len(plans) == 1
    plan = plans[0]
    targets = list(plan.targets.values())
    # the chord-based box fit reads a 240 degree curl far shorter than its spine
    spine = math.radians(span) * radius
    diagonal = math.sqrt(spine**2 + width**2)
    reach = max(math.dist(a, b) for a in targets for b in targets)
    assert abs(reach - diagonal) / diagonal < 0.05
    # the mid ring rides along, unrolled evenly down the rectangle's middle
    assert not plan.pinned
    placed = {uvs[f][c]: t for (f, c), t in plan.targets.items()}
    step = spine / segments
    mids = [
        placed[
            (
                radius * math.cos(math.radians(span) * k / segments),
                radius * math.sin(math.radians(span) * k / segments),
            )
        ]
        for k in range(1, segments)
    ]
    for a, b in zip(mids, mids[1:]):
        assert 0.8 * step < math.dist(a, b) < 1.2 * step


@pytest.mark.smoke
def test_rectify_skips_a_blob(rectify):
    points = [(0.0, 0.0)] + [
        (math.cos(a * math.pi / 3), math.sin(a * math.pi / 3)) for a in range(6)
    ]
    faces = [(0, i + 1, (i + 1) % 6 + 1) for i in range(6)]
    uvs = [[points[v] for v in face] for face in faces]
    assert rectify(faces, uvs) == []


@pytest.mark.smoke
def test_rectify_admits_a_toothed_strip_by_its_boundary(rectify):
    # deep teeth push the strip under the share gate
    verts, faces, uvs = grid_island(12, 1)

    def tooth(uv):
        x, y = uv
        lean = 0.5 if int(x) % 2 else -0.5
        return (x, y + lean) if y in (0.0, 1.0) else uv

    uvs = [[tooth(uv) for uv in face] for face in uvs]
    plans = rectify(faces, uvs)
    assert len(plans) == 1
    box = target_box(plans[0].targets)
    assert all(on_perimeter(t, box) for t in plans[0].targets.values())
    assert box[2] - box[0] > 14


@pytest.mark.smoke
def test_rectify_reaches_islands_bordering_their_own_cut(rectify):
    # a cut open tube carries two uvs on every cut vertex
    verts, faces, uvs, seams, row_of = tube_island(6, 8, 0.0)
    plans = rectify(faces, uvs)
    assert len(plans) == 1
    plan = plans[0]
    assert sorted(plan.group) == list(range(len(faces)))
    outline = {
        corner: target
        for corner, target in plan.targets.items()
        if uvs[corner[0]][corner[1]] in outline_points(uvs)
    }
    box = target_box(outline)
    assert all(on_perimeter(t, box) for t in outline.values())


@pytest.mark.smoke
def test_rectify_skips_an_island_with_a_hole(rectify):
    verts, faces, uvs = annulus_island()
    assert rectify(faces, uvs) == []


def quad_grid(cols, rows):
    verts = [
        (float(x), float(y), 0.0) for y in range(rows + 1) for x in range(cols + 1)
    ]
    faces = []
    for cy in range(rows):
        for cx in range(cols):
            a = cy * (cols + 1) + cx
            faces.append([a, a + 1, a + cols + 2, a + cols + 1])
    return verts, faces


# the grid's outline pulled onto a superellipse, whose corners are too round to turn
def rounded_strip(cols, rows, power):
    verts, faces = quad_grid(cols, rows)

    def round_corners(vertex):
        x, y = vertex[0], vertex[1]
        u = (x - cols / 2) / (cols / 2)
        v = (y - rows / 2) / (rows / 2)
        square = max(abs(u), abs(v))
        if square == 0:
            return (x, y)
        shrink = square / (abs(u) ** power + abs(v) ** power) ** (1 / power)
        return (cols / 2 * (1 + u * shrink), rows / 2 * (1 + v * shrink))

    uvs = [[round_corners(verts[v]) for v in face] for face in faces]
    return verts, faces, uvs


# the grid's outline pulled in towards its middle row, pointed at both ends
def lens_strip(cols, rows, squeeze):
    verts, faces = quad_grid(cols, rows)

    def pinch(vertex):
        x, y = vertex[0], vertex[1]
        width = math.sqrt(1 - squeeze * ((x - cols / 2) / (cols / 2)) ** 2)
        return (x, (y - rows / 2) * width + rows / 2)

    uvs = [[pinch(verts[v]) for v in face] for face in faces]
    return verts, faces, uvs


@pytest.mark.smoke
def test_finish_parts_a_long_strip(finish):
    verts, faces, uvs = strip_island(30)
    assert len(uv_groups(faces, finish(verts, faces, uvs))) == 4


@pytest.mark.smoke
def test_finish_solves_a_pinned_strip_inside_its_rectangle(finish, rectify):
    verts, faces, uvs = rounded_strip(12, 4, 3)
    (plan,) = rectify(faces, uvs)
    assert plan.pinned
    finished = finish(verts, faces, uvs)
    for (face, corner), target in plan.targets.items():
        assert math.dist(finished[face][corner], target) < 1e-9
    x0, y0, x1, y1 = target_box(plan.targets)
    free = [
        (face, corner)
        for face, corners in enumerate(uvs)
        for corner in range(len(corners))
        if (face, corner) not in plan.targets
    ]
    assert any(finished[f][c] != uvs[f][c] for f, c in free)
    assert all(
        x0 < finished[f][c][0] < x1 and y0 < finished[f][c][1] < y1 for f, c in free
    )
    assert all(signed_area(corners) > 0 for corners in finished)


# the pointed ends put three pinned corners of an end face on one side of the rectangle
@pytest.mark.smoke
def test_finish_keeps_the_engine_map_when_the_rectangle_crushes_a_face(finish, rectify):
    verts, faces, uvs = lens_strip(16, 3, 0.9)
    (plan,) = rectify(faces, uvs)
    assert plan.pinned
    assert finish(verts, faces, uvs) == [list(corners) for corners in uvs]


FLAT_QUAD_VERTS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
FLAT_QUAD_UVS = [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]]


def test_flatten_distortion_is_four_at_isometry():
    value = flatten_distortion(FLAT_QUAD_VERTS, [(0, 1, 2, 3)], FLAT_QUAD_UVS, [0])
    assert abs(value - 4.0) < 1e-9


def test_flatten_distortion_grows_with_stretch():
    stretched = [[(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]]
    value = flatten_distortion(FLAT_QUAD_VERTS, [(0, 1, 2, 3)], stretched, [0])
    assert value > 4.5


def test_flatten_distortion_flags_a_real_flip():
    faces = [(0, 1, 2), (0, 2, 3)]
    uvs = [
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
    ]
    assert flatten_distortion(FLAT_QUAD_VERTS, faces, uvs, [0, 1]) == math.inf


def test_flatten_distortion_reads_a_mirrored_island_like_its_source():
    mirrored = [[(0.0, 0.0), (-1.0, 0.0), (-1.0, 1.0), (0.0, 1.0)]]
    value = flatten_distortion(FLAT_QUAD_VERTS, [(0, 1, 2, 3)], mirrored, [0])
    assert abs(value - 4.0) < 1e-9


def test_collapsed_uvs_are_detected():
    point = (0.1, 0.9987)
    assert uvs_collapsed([[point, point, point]] * 100)


def test_real_uvs_are_not_collapsed():
    quad = [(0.0, 0.0), (0.01, 0.0), (0.01, 0.01), (0.0, 0.01)]
    assert not uvs_collapsed([quad])
