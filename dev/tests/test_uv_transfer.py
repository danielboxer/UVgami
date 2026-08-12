import importlib.util
import math
from pathlib import Path

import pytest

# loaded from file so importing doesn't touch the blender addon package
spec = importlib.util.spec_from_file_location(
    "addon_uv_transfer", Path(__file__).parents[2] / "src" / "uv_transfer.py"
)
uv_transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uv_transfer)
plan_transfer = uv_transfer.plan_transfer

# unit square as two triangles sharing edge v0-v2
SQUARE_POS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
SQUARE_FACES = [[0, 1, 2], [0, 2, 3]]


def approx_uvs(expected):
    return {k: pytest.approx(v, abs=1e-9) for k, v in expected.items()}


def test_exact_reordered_faces_and_verts():
    # output remaps vertices and lists faces in a different order
    out_pos = [(1, 1, 0), (0, 0, 0), (0, 1, 0), (1, 0, 0)]
    out_faces = [[1, 0, 2], [1, 3, 0]]
    out_uvs = [
        [(0, 0), (1, 1), (0, 1)],
        [(0, 0), (1, 0), (1, 1)],
    ]

    plan = plan_transfer(SQUARE_POS, SQUARE_FACES, out_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.split_faces == {}
    # every input loop gets the uv of its own vertex position (planar uv)
    assert plan.loop_uvs == {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (1.0, 1.0),
        3: (0.0, 0.0),
        4: (1.0, 1.0),
        5: (0.0, 1.0),
    }


def test_seam_duplicates_map_many_to_one():
    # output cuts the shared edge: v0 and v2 each become two coincident verts
    out_pos = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
    ]
    out_faces = [[0, 1, 2], [3, 4, 5]]
    # different uvs on each side prove they land on different input loops
    out_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(2, 0), (2, 1), (3, 1)],
    ]
    plan = plan_transfer(SQUARE_POS, SQUARE_FACES, out_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.loop_uvs == {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (1.0, 1.0),
        3: (2.0, 0.0),
        4: (2.0, 1.0),
        5: (3.0, 1.0),
    }
    assert plan.seam_edges == {(0, 2)}


def test_triangulated_quad_assigns_all_corners():
    in_pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    in_faces = [[0, 1, 2, 3]]
    out_faces = [[0, 1, 2], [0, 2, 3]]
    out_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (1, 1), (0, 1)],
    ]

    plan = plan_transfer(in_pos, in_faces, in_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.split_faces == {}
    assert plan.loop_uvs == {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (1.0, 1.0),
        3: (0.0, 1.0),
    }


def test_seam_through_quad_welds_the_cut_off():
    # two quads side by side, a uv cut runs across the first one's diagonal.
    # the far piece is redrawn from its 3d shape in the anchor's frame, so the
    # quad keeps its four loops
    in_pos = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (2, 0, 0),
        (2, 1, 0),
    ]
    in_faces = [[0, 1, 2, 3], [1, 4, 5, 2]]
    out_faces = [[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]]
    out_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(5, 5), (6, 6), (5, 6)],
        [(1, 0), (2, 0), (2, 1)],
        [(1, 0), (2, 1), (1, 1)],
    ]

    plan = plan_transfer(in_pos, in_faces, in_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.split_faces == {}
    assert plan.loop_uvs == approx_uvs(
        {
            0: (0.0, 0.0),
            1: (1.0, 0.0),
            2: (1.0, 1.0),
            3: (0.0, 1.0),
            4: (1.0, 0.0),
            5: (2.0, 0.0),
            6: (2.0, 1.0),
            7: (1.0, 1.0),
        }
    )
    # the welded diagonal is no input edge, so the cut leaves no seam
    assert plan.seam_edges == set()


def test_far_chart_scale_is_ignored():
    # the far piece's own chart sits at half the anchor's scale. its uvs are
    # not reused: the piece is redrawn from its 3d shape at the anchor's
    # density, so the quad welds with no density jump
    in_pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    in_faces = [[0, 1, 2, 3]]
    out_faces = [[0, 1, 2], [0, 2, 3]]
    out_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(5, 5), (5.5, 5.5), (5, 5.5)],
    ]

    plan = plan_transfer(in_pos, in_faces, in_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.split_faces == {}
    assert plan.loop_uvs == approx_uvs(
        {
            0: (0.0, 0.0),
            1: (1.0, 0.0),
            2: (1.0, 1.0),
            3: (0.0, 1.0),
        }
    )
    assert plan.seam_edges == set()


def test_mirrored_anchor_reflects_the_flap():
    # the anchor chart is mirrored, its uv winding is clockwise, so the flap
    # is redrawn on the reflected side to keep the patch consistent
    in_pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    in_faces = [[0, 1, 2, 3]]
    out_faces = [[0, 1, 2], [0, 2, 3]]
    out_uvs = [
        [(0, 0), (0, 1), (1, 1)],
        [(5, 5), (5.5, 5.5), (5, 5.5)],
    ]

    plan = plan_transfer(in_pos, in_faces, in_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.split_faces == {}
    assert plan.loop_uvs == approx_uvs(
        {
            0: (0.0, 0.0),
            1: (0.0, 1.0),
            2: (1.0, 1.0),
            3: (1.0, 0.0),
        }
    )


def test_stretched_anchor_does_not_square_its_stretch():
    # the anchor chart is stretched 2x along u, so the shared diagonal is
    # longer in uv than in 3d. drawing the flap through the anchor's own map
    # keeps the whole quad at the anchor's density: uv area 2, 3d area 1.
    # scaling it by the diagonal instead would put the flap at (0.5, 1.5),
    # a density of 2.5
    in_pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    in_faces = [[0, 1, 2, 3]]
    out_faces = [[0, 1, 2], [0, 2, 3]]
    out_uvs = [
        [(0, 0), (2, 0), (2, 1)],
        [(5, 5), (6, 6), (5, 6)],
    ]

    plan = plan_transfer(in_pos, in_faces, in_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.split_faces == {}
    assert plan.loop_uvs == approx_uvs(
        {
            0: (0.0, 0.0),
            1: (2.0, 0.0),
            2: (2.0, 1.0),
            3: (0.0, 1.0),
        }
    )


def test_bent_quad_flap_keeps_its_3d_shape():
    # the quad is folded along the cut diagonal and the flap triangle is
    # equilateral in 3d, so it unfolds to an equilateral apex left of the
    # uv diagonal instead of reusing its own chart's shape
    in_pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 1)]
    in_faces = [[0, 1, 2, 3]]
    out_faces = [[0, 1, 2], [0, 2, 3]]
    out_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(5, 5), (6, 6), (5, 6)],
    ]

    plan = plan_transfer(in_pos, in_faces, in_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.split_faces == {}
    assert plan.loop_uvs == approx_uvs(
        {
            0: (0.0, 0.0),
            1: (1.0, 0.0),
            2: (1.0, 1.0),
            3: ((1 - math.sqrt(3)) / 2, (1 + math.sqrt(3)) / 2),
        }
    )


def test_ngon_chain_welds_part_by_part():
    # pentagon fan, the two far pieces sit in a translated frame. the second
    # one only chains through the first one's welded uvs
    in_pos = [(0, 0, 0), (4, 0, 0), (4, 4, 0), (2, 5, 0), (0, 4, 0)]
    in_faces = [[0, 1, 2, 3, 4]]
    out_faces = [[0, 1, 2], [0, 2, 3], [0, 3, 4]]
    out_uvs = [
        [(0, 0), (4, 0), (4, 4)],
        [(10, 0), (14, 4), (12, 5)],
        [(10, 0), (12, 5), (10, 4)],
    ]

    plan = plan_transfer(in_pos, in_faces, in_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.split_faces == {}
    assert plan.loop_uvs == approx_uvs(
        {
            0: (0.0, 0.0),
            1: (4.0, 0.0),
            2: (4.0, 4.0),
            3: (2.0, 5.0),
            4: (0.0, 4.0),
        }
    )


def test_weld_landing_on_its_island_splits():
    # the far piece would land where a face of the same island already sits,
    # as across a slit, so the weld backs off and the face splits
    in_pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 1.5, 0)]
    in_faces = [[0, 1, 2, 3], [2, 3, 4]]
    out_faces = [[0, 1, 2], [0, 2, 3], [2, 3, 4]]
    out_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(5, 5), (6, 6), (5, 6)],
        # shares vertex 2 at (1, 1) with the anchor, so it is the same island,
        # and its long edge crosses the glued piece's top edge
        [(1, 1), (0.3, 0.7), (0.9, 1.5)],
    ]

    plan = plan_transfer(in_pos, in_faces, in_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.loop_uvs == {
        4: (1.0, 1.0),
        5: (0.3, 0.7),
        6: (0.9, 1.5),
    }
    assert plan.split_faces == {
        0: [
            ([0, 1, 2], [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]),
            ([0, 2, 3], [(5.0, 5.0), (6.0, 6.0), (5.0, 6.0)]),
        ]
    }
    # the split diagonal, plus edge 2-3, where the far piece's island meets
    # the neighbouring triangle's
    assert plan.seam_edges == {(0, 2), (2, 3)}


def test_weld_moves_the_cut_to_the_faces_outer_edge():
    # the same blocking face, but in its own island, which the pack pulls
    # clear of the glued piece, so the weld stands. welding undoes the cut
    # across the diagonal but not the cut itself: it lands on edge 2-3, where
    # the redrawn piece now meets the neighbouring triangle
    in_pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 1.5, 0)]
    in_faces = [[0, 1, 2, 3], [2, 3, 4]]
    out_faces = [[0, 1, 2], [0, 2, 3], [2, 3, 4]]
    out_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(5, 5), (6, 6), (5, 6)],
        [(1.05, 1.0), (0.3, 0.7), (0.9, 1.5)],
    ]

    plan = plan_transfer(in_pos, in_faces, in_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.split_faces == {}
    assert plan.loop_uvs == approx_uvs(
        {
            0: (0.0, 0.0),
            1: (1.0, 0.0),
            2: (1.0, 1.0),
            3: (0.0, 1.0),
            4: (1.05, 1.0),
            5: (0.3, 0.7),
            6: (0.9, 1.5),
        }
    )
    assert plan.seam_edges == {(2, 3)}


def test_without_a_pack_another_islands_overlap_splits():
    # the same layout the weld stands on when a pack follows. with no pack
    # nothing pulls the blocking island clear, so the overlap counts and the
    # face splits instead
    in_pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 1.5, 0)]
    in_faces = [[0, 1, 2, 3], [2, 3, 4]]
    out_faces = [[0, 1, 2], [0, 2, 3], [2, 3, 4]]
    out_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(5, 5), (6, 6), (5, 6)],
        [(1.05, 1.0), (0.3, 0.7), (0.9, 1.5)],
    ]

    plan = plan_transfer(in_pos, in_faces, in_pos, out_faces, out_uvs, repack=False)

    assert plan.ok
    assert plan.split_faces == {
        0: [
            ([0, 1, 2], [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]),
            ([0, 2, 3], [(5.0, 5.0), (6.0, 6.0), (5.0, 6.0)]),
        ]
    }
    assert plan.loop_uvs == {
        4: (1.05, 1.0),
        5: (0.3, 0.7),
        6: (0.9, 1.5),
    }
    assert plan.seam_edges == {(0, 2), (2, 3)}


def test_vertex_only_cut_splits_in_input_winding():
    # the pieces share only vertex 2, so no weld edge exists and the face
    # splits, each piece reordered to the input corner order
    in_pos = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (1, 3, 0), (0, 2, 0)]
    in_faces = [[0, 1, 2, 3, 4]]
    # rotated corner orders prove the reorder
    out_faces = [[2, 0, 1], [4, 2, 3]]
    out_uvs = [
        [(1, 1), (0, 0), (1, 0)],
        [(5, 6), (6, 5), (6, 6)],
    ]

    plan = plan_transfer(in_pos, in_faces, in_pos, out_faces, out_uvs)

    assert plan.ok
    assert plan.loop_uvs == {}
    assert plan.split_faces == {
        0: [
            ([0, 1, 2], [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]),
            ([2, 3, 4], [(6.0, 5.0), (6.0, 6.0), (5.0, 6.0)]),
        ]
    }


def test_conflicting_triangle_cannot_be_split():
    in_pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0)]
    in_faces = [[0, 1, 2]]
    # the output holds the same triangle twice with different uvs
    out_faces = [[0, 1, 2], [0, 1, 2]]
    out_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(5, 5), (6, 5), (6, 6)],
    ]

    result = plan_transfer(in_pos, in_faces, in_pos, out_faces, out_uvs)

    assert not result.ok
    assert result.reason == "ambiguous_geometry"


def test_coincident_input_faces_are_ambiguous():
    a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
    # two input triangles stacked on the exact same positions
    in_pos = [a, b, c, a, b, c]
    in_faces = [[0, 1, 2], [3, 4, 5]]
    out_faces = [[0, 1, 2]]
    out_uvs = [[(0, 0), (1, 0), (0, 1)]]

    # single output triangle, but coincidence collapses both input faces
    result = plan_transfer(in_pos, in_faces, [a, b, c], out_faces, out_uvs)

    assert not result.ok
    assert result.reason == "ambiguous_geometry"


def test_unmatched_output_face_fails():
    # verts 1, 3, 0 never share an input face
    out_faces = [[1, 3, 0]]
    out_uvs = [[(0, 0), (1, 0), (1, 1)]]

    result = plan_transfer(SQUARE_POS, SQUARE_FACES, SQUARE_POS, out_faces, out_uvs)

    assert not result.ok
    assert result.reason == "face_match"
