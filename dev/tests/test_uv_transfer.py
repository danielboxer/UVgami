import math

import numpy
import pytest

from no_bpy.uv_transfer import (
    AMBIGUOUS_GEOMETRY,
    TransferError,
    transfer_exact,
    transfer_exact_job,
)

# unit square as two triangles sharing edge v0-v2
SQUARE_POSITIONS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
SQUARE_TRIANGLES = [[0, 1, 2], [0, 2, 3]]
SQUARE_QUAD = [[0, 1, 2, 3]]


def approx_uvs(expected):
    return {loop: pytest.approx(uv, abs=1e-9) for loop, uv in expected.items()}


def test_exact_reordered_faces_and_verts():
    # output remaps vertices and lists faces in a different order
    output_positions = [(1, 1, 0), (0, 0, 0), (0, 1, 0), (1, 0, 0)]
    output_faces = [[1, 0, 2], [1, 3, 0]]
    output_uvs = [
        [(0, 0), (1, 1), (0, 1)],
        [(0, 0), (1, 0), (1, 1)],
    ]

    plan = transfer_exact(
        SQUARE_POSITIONS, SQUARE_TRIANGLES, output_positions, output_faces, output_uvs
    )

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
    output_positions = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
    ]
    output_faces = [[0, 1, 2], [3, 4, 5]]
    # different uvs on each side prove they land on different input loops
    output_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(2, 0), (2, 1), (3, 1)],
    ]
    plan = transfer_exact(
        SQUARE_POSITIONS, SQUARE_TRIANGLES, output_positions, output_faces, output_uvs
    )

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
    output_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (1, 1), (0, 1)],
    ]

    plan = transfer_exact(
        SQUARE_POSITIONS, SQUARE_QUAD, SQUARE_POSITIONS, SQUARE_TRIANGLES, output_uvs
    )

    assert plan.split_faces == {}
    assert plan.loop_uvs == {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (1.0, 1.0),
        3: (0.0, 1.0),
    }


def test_seam_through_quad_welds_the_cut_off():
    # two quads side by side, a uv cut runs across the first one's diagonal
    input_positions = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (2, 0, 0),
        (2, 1, 0),
    ]
    input_faces = [[0, 1, 2, 3], [1, 4, 5, 2]]
    output_faces = [[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]]
    output_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(5, 5), (6, 6), (5, 6)],
        [(1, 0), (2, 0), (2, 1)],
        [(1, 0), (2, 1), (1, 1)],
    ]

    plan = transfer_exact(
        input_positions, input_faces, input_positions, output_faces, output_uvs
    )

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
    # the welded diagonal is no input edge
    assert plan.seam_edges == set()


def test_far_chart_scale_is_ignored():
    # the far piece's own chart sits at half the anchor's scale
    output_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(5, 5), (5.5, 5.5), (5, 5.5)],
    ]

    plan = transfer_exact(
        SQUARE_POSITIONS, SQUARE_QUAD, SQUARE_POSITIONS, SQUARE_TRIANGLES, output_uvs
    )

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
    # the anchor chart is mirrored, its uv winding is clockwise
    output_uvs = [
        [(0, 0), (0, 1), (1, 1)],
        [(5, 5), (5.5, 5.5), (5, 5.5)],
    ]

    plan = transfer_exact(
        SQUARE_POSITIONS, SQUARE_QUAD, SQUARE_POSITIONS, SQUARE_TRIANGLES, output_uvs
    )

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
    # the anchor chart is stretched 2x along u
    output_uvs = [
        [(0, 0), (2, 0), (2, 1)],
        [(5, 5), (6, 6), (5, 6)],
    ]

    plan = transfer_exact(
        SQUARE_POSITIONS, SQUARE_QUAD, SQUARE_POSITIONS, SQUARE_TRIANGLES, output_uvs
    )

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
    # the quad is folded along the cut diagonal and the flap is equilateral in 3d
    input_positions = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 1)]
    output_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(5, 5), (6, 6), (5, 6)],
    ]

    plan = transfer_exact(
        input_positions, SQUARE_QUAD, input_positions, SQUARE_TRIANGLES, output_uvs
    )

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
    # pentagon fan, the two far pieces sit in a translated frame
    input_positions = [(0, 0, 0), (4, 0, 0), (4, 4, 0), (2, 5, 0), (0, 4, 0)]
    input_faces = [[0, 1, 2, 3, 4]]
    output_faces = [[0, 1, 2], [0, 2, 3], [0, 3, 4]]
    output_uvs = [
        [(0, 0), (4, 0), (4, 4)],
        [(10, 0), (14, 4), (12, 5)],
        [(10, 0), (12, 5), (10, 4)],
    ]

    plan = transfer_exact(
        input_positions, input_faces, input_positions, output_faces, output_uvs
    )

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


# a quad cut on its diagonal with a far piece, and a triangle on its top edge
WELD_POSITIONS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 1.5, 0)]
WELD_INPUT_FACES = [[0, 1, 2, 3], [2, 3, 4]]
WELD_OUTPUT_FACES = [[0, 1, 2], [0, 2, 3], [2, 3, 4]]
# the triangle in its own island, crossing where the far piece would weld
BLOCKING_ISLAND_UVS = [
    [(0, 0), (1, 0), (1, 1)],
    [(5, 5), (6, 6), (5, 6)],
    [(1.05, 1.0), (0.3, 0.7), (0.9, 1.5)],
]


def test_weld_landing_on_its_island_splits():
    # the far piece would land where a face of the same island already sits
    output_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(5, 5), (6, 6), (5, 6)],
        # same island as the anchor, its long edge crosses the glued piece
        [(1, 1), (0.3, 0.7), (0.9, 1.5)],
    ]

    plan = transfer_exact(
        WELD_POSITIONS, WELD_INPUT_FACES, WELD_POSITIONS, WELD_OUTPUT_FACES, output_uvs
    )

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
    # edge 2-3 is where the far piece's island meets the neighbouring triangle
    assert plan.seam_edges == {(0, 2), (2, 3)}


def test_weld_moves_the_cut_to_the_faces_outer_edge():
    # the pack pulls the blocking island clear
    plan = transfer_exact(
        WELD_POSITIONS,
        WELD_INPUT_FACES,
        WELD_POSITIONS,
        WELD_OUTPUT_FACES,
        BLOCKING_ISLAND_UVS,
    )

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
    plan = transfer_exact(
        WELD_POSITIONS,
        WELD_INPUT_FACES,
        WELD_POSITIONS,
        WELD_OUTPUT_FACES,
        BLOCKING_ISLAND_UVS,
        repack=False,
    )

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
    # the pieces share only vertex 2, no weld edge exists
    input_positions = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (1, 3, 0), (0, 2, 0)]
    input_faces = [[0, 1, 2, 3, 4]]
    # rotated corner orders prove the reorder
    output_faces = [[2, 0, 1], [4, 2, 3]]
    output_uvs = [
        [(1, 1), (0, 0), (1, 0)],
        [(5, 6), (6, 5), (6, 6)],
    ]

    plan = transfer_exact(
        input_positions, input_faces, input_positions, output_faces, output_uvs
    )

    assert plan.loop_uvs == {}
    assert plan.split_faces == {
        0: [
            ([0, 1, 2], [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]),
            ([2, 3, 4], [(6.0, 5.0), (6.0, 6.0), (5.0, 6.0)]),
        ]
    }


def test_conflicting_triangle_cannot_be_split():
    input_positions = [(0, 0, 0), (1, 0, 0), (1, 1, 0)]
    input_faces = [[0, 1, 2]]
    # the output holds the same triangle twice with different uvs
    output_faces = [[0, 1, 2], [0, 1, 2]]
    output_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(5, 5), (6, 5), (6, 6)],
    ]

    with pytest.raises(TransferError) as raised:
        transfer_exact(
            input_positions, input_faces, input_positions, output_faces, output_uvs
        )
    assert raised.value.reason == AMBIGUOUS_GEOMETRY


def test_coincident_input_faces_each_take_their_own_output():
    a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
    # stacked input triangles, one output face per input face
    input_positions = [a, b, c, a, b, c]
    input_faces = [[0, 1, 2], [3, 4, 5]]
    output_faces = [[0, 1, 2], [3, 4, 5]]
    output_uvs = [[(0, 0), (1, 0), (0, 1)], [(2, 0), (3, 0), (2, 1)]]

    plan = transfer_exact(
        input_positions, input_faces, input_positions, output_faces, output_uvs
    )

    assert plan.loop_uvs == {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (0.0, 1.0),
        3: (2.0, 0.0),
        4: (3.0, 0.0),
        5: (2.0, 1.0),
    }


def test_cut_copy_near_a_second_vertex_is_ambiguous():
    a, b, c = (0, 0, 0), (1, 0, 0), (0, 1, 0)
    # vertex 3 is a needle away from vertex 0
    input_positions = [a, b, c, (0, 0, 1e-6)]
    input_faces = [[0, 1, 2], [3, 1, 0]]
    output_positions = [a, b, c, (0, 0, 1e-6), a]
    output_faces = [[0, 1, 2], [3, 1, 4]]
    output_uvs = [[(0, 0), (1, 0), (0, 1)], [(2, 0), (3, 0), (2, 1)]]

    with pytest.raises(TransferError) as raised:
        transfer_exact(
            input_positions, input_faces, output_positions, output_faces, output_uvs
        )
    assert raised.value.reason == AMBIGUOUS_GEOMETRY


def test_merged_output_still_matches_by_position():
    a, b, c, d = (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)
    # two input triangles with their shared edge doubled, merged by the engine
    input_positions = [a, b, c, b, c, d]
    input_faces = [[0, 1, 2], [3, 4, 5]]
    output_positions = [a, b, c, d]
    output_faces = [[0, 1, 2], [1, 2, 3]]
    output_uvs = [[(0, 0), (1, 0), (0, 1)], [(1, 0), (0, 1), (1, 1)]]

    plan = transfer_exact(
        input_positions, input_faces, output_positions, output_faces, output_uvs
    )

    assert plan.loop_uvs == {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (0.0, 1.0),
        3: (1.0, 0.0),
        4: (0.0, 1.0),
        5: (1.0, 1.0),
    }


def test_small_piece_far_from_the_origin_matches():
    # a 0.1 wide triangle 138 units out, at the engine's 7 digits
    input_positions = [(138.0, 0, 0), (138.1, 0, 0), (138.0, 0.1, 0)]
    output_positions = [
        tuple(float("%.6e" % (coordinate + 3e-5)) for coordinate in position)
        for position in input_positions
    ]
    output_faces = [[0, 1, 2]]
    output_uvs = [[(0, 0), (1, 0), (0, 1)]]

    plan = transfer_exact(
        input_positions, [[0, 1, 2]], output_positions, output_faces, output_uvs
    )

    assert plan.loop_uvs == {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0)}


def test_unmatched_output_face_fails():
    # verts 1, 3, 0 never share an input face
    output_faces = [[1, 3, 0]]
    output_uvs = [[(0, 0), (1, 0), (1, 1)]]

    with pytest.raises(TransferError) as raised:
        transfer_exact(
            SQUARE_POSITIONS,
            SQUARE_TRIANGLES,
            SQUARE_POSITIONS,
            output_faces,
            output_uvs,
        )
    assert raised.value.reason == "face_match"


# two loose triangles, the output only unwrapped the first
LOOSE_POSITIONS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (5, 0, 0), (6, 0, 0), (5, 1, 0)]
LOOSE_FACES = [[0, 1, 2], [3, 4, 5]]
FIRST_TRIANGLE_UVS = [[(0, 0), (1, 0), (0, 1)]]


def test_missing_piece_fails_unless_partial():
    with pytest.raises(TransferError) as raised:
        transfer_exact(
            LOOSE_POSITIONS,
            LOOSE_FACES,
            LOOSE_POSITIONS[:3],
            [[0, 1, 2]],
            FIRST_TRIANGLE_UVS,
        )
    assert raised.value.reason == "incomplete_coverage"


def test_partial_leaves_the_missing_piece_alone():
    plan = transfer_exact(
        LOOSE_POSITIONS,
        LOOSE_FACES,
        LOOSE_POSITIONS[:3],
        [[0, 1, 2]],
        FIRST_TRIANGLE_UVS,
        partial=True,
    )

    assert plan.untouched_faces == {1}
    assert plan.loop_uvs == approx_uvs({0: (0, 0), 1: (1, 0), 2: (0, 1)})
    assert not plan.split_faces
    assert not plan.seam_edges


def test_partial_with_nothing_covered_fails():
    with pytest.raises(TransferError) as raised:
        transfer_exact(LOOSE_POSITIONS, LOOSE_FACES, [], [], [], partial=True)
    assert raised.value.reason == "incomplete_coverage"


def test_job_splits_flat_arrays_per_face():
    # a quad and a triangle
    positions = numpy.array(
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0)], dtype=float
    )
    faces = [[0, 1, 2, 3], [1, 4, 2]]
    uvs = [[(0, 0), (1, 0), (1, 1), (0, 1)], [(1, 0), (2, 0), (1, 1)]]
    corners = numpy.array([v for face in faces for v in face], dtype=numpy.int64)
    totals = numpy.array([4, 3], dtype=numpy.int64)
    loop_uvs = numpy.array([uv for face in uvs for uv in face], dtype=float)

    plan = transfer_exact_job(
        positions, corners, totals, positions, corners, totals, loop_uvs
    )

    assert plan == transfer_exact(positions, faces, positions, faces, uvs)
    assert plan.loop_uvs[6] == (1.0, 1.0)
