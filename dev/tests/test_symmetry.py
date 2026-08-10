import importlib.util
import sys
from pathlib import Path

# loaded from file so it doesn't need the bpy-only addon package
PKG = Path(__file__).parents[2] / "src" / "seams"
spec = importlib.util.spec_from_file_location(
    "seams", PKG / "__init__.py", submodule_search_locations=[str(PKG)]
)
sys.modules["seams"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sys.modules["seams"])
from seams import mirror_seams, stack_mirrored  # noqa: E402


def test_mirror_seams_closes_under_the_map():
    # 0<->1 and 2<->3 across the plane, seam (0, 2) mirrors to (1, 3)
    mirror = {0: 1, 1: 0, 2: 3, 3: 2}
    edges = {(0, 2): [0], (1, 3): [1], (0, 1): [0, 1]}
    seams = mirror_seams({(0, 2)}, [mirror], edges)
    assert seams == {(0, 2), (1, 3)}


def test_mirror_seams_skips_pairs_that_are_not_edges():
    mirror = {0: 1, 1: 0, 2: 3, 3: 2}
    edges = {(0, 2): [0]}
    assert mirror_seams({(0, 2)}, [mirror], edges) == {(0, 2)}


def test_mirror_seams_closes_across_two_maps():
    # quadrants: x sends 0->1, y sends 0->2, both needed to reach 3
    mirror_x = dict(enumerate([1, 0, 3, 2, 5, 4, 7, 6]))
    mirror_y = dict(enumerate([2, 3, 0, 1, 6, 7, 4, 5]))
    edges = {key: [] for key in ((0, 4), (1, 5), (2, 6), (3, 7))}
    seams = mirror_seams({(0, 4)}, [mirror_x, mirror_y], edges)
    assert seams == set(edges)


def mirrored_quads():
    # two separate quads, the second the first reflected across the plane,
    # with uvs from independent flattens that do not match
    faces = [(0, 1, 2, 3), (4, 5, 6, 7)]
    uvs = [
        [(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)],
        [(0.5, 0.5), (0.6, 0.5), (0.6, 0.6), (0.5, 0.6)],
    ]
    mirror = dict(enumerate([4, 5, 6, 7, 0, 1, 2, 3]))
    return faces, uvs, mirror


def apply_moves(uvs, moves):
    out = [list(face) for face in uvs]
    for target, corner, source, source_corner in moves:
        out[target][corner] = uvs[source][source_corner]
    return out


def test_stack_mirrored_copies_the_first_islands_uvs():
    faces, uvs, mirror = mirrored_quads()
    moves = apply_moves(uvs, stack_mirrored(faces, uvs, [mirror]))
    assert moves[0] == uvs[0]
    assert moves[1] == uvs[0]


def test_stack_mirrored_leaves_a_straddling_island_alone():
    # two quads welded along the plane edge (0, 1) into one island, which
    # the map sends onto itself
    faces = [(0, 1, 2, 3), (0, 1, 4, 5)]
    uvs = [
        [(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)],
        [(0.0, 0.0), (0.1, 0.0), (0.1, -0.1), (0.0, -0.1)],
    ]
    mirror = dict(enumerate([0, 1, 4, 5, 2, 3]))
    assert stack_mirrored(faces, uvs, [mirror]) == []


def test_stack_mirrored_skips_islands_of_different_size():
    # the second island is two welded faces, no whole image exists
    faces = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 4, 5)]
    uvs = [
        [(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)],
        [(0.5, 0.5), (0.6, 0.5), (0.6, 0.6), (0.5, 0.6)],
        [(0.5, 0.4), (0.6, 0.4), (0.5, 0.5), (0.6, 0.5)],
    ]
    mirror = dict(enumerate([4, 5, 6, 7, 0, 1, 2, 3, 8, 9]))
    assert stack_mirrored(faces, uvs, [mirror]) == []


def test_stack_mirrored_stacks_all_four_quadrant_copies():
    # four separate quads related by two mirror maps
    faces = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)]
    base = [(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)]
    uvs = [
        base,
        [(0.5, 0.5), (0.6, 0.5), (0.6, 0.6), (0.5, 0.6)],
        [(0.3, 0.7), (0.4, 0.7), (0.4, 0.8), (0.3, 0.8)],
        [(0.8, 0.2), (0.9, 0.2), (0.9, 0.3), (0.8, 0.3)],
    ]
    mirror_x = dict(enumerate([4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11]))
    mirror_y = dict(enumerate([8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7]))
    moves = apply_moves(uvs, stack_mirrored(faces, uvs, [mirror_x, mirror_y]))
    assert all(face == base for face in moves)


def test_stack_mirrored_ignores_islands_outside_the_map():
    # a third quad the partial map does not cover keeps its own uvs
    faces, uvs, mirror = mirrored_quads()
    faces = faces + [(8, 9, 10, 11)]
    lone = [(0.8, 0.8), (0.9, 0.8), (0.9, 0.9), (0.8, 0.9)]
    uvs = uvs + [lone]
    moves = apply_moves(uvs, stack_mirrored(faces, uvs, [mirror]))
    assert moves[1] == uvs[0]
    assert moves[2] == lone
