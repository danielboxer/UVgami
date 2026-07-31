import importlib.util
from pathlib import Path

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


def test_exact_reordered_faces_and_verts():
    # output remaps vertices and lists faces in a different order
    out_pos = [(1, 1, 0), (0, 0, 0), (0, 1, 0), (1, 0, 0)]
    out_faces = [[1, 0, 2], [1, 3, 0]]
    out_uvs = [
        [(0, 0), (1, 1), (0, 1)],
        [(0, 0), (1, 0), (1, 1)],
    ]

    plan = plan_transfer(SQUARE_POS, SQUARE_FACES, out_pos, out_faces, out_uvs, [])

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
    # both cut edges point at input edge v0-v2
    out_seams = [(0, 2), (3, 4)]

    plan = plan_transfer(
        SQUARE_POS, SQUARE_FACES, out_pos, out_faces, out_uvs, out_seams
    )

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

    plan = plan_transfer(in_pos, in_faces, in_pos, out_faces, out_uvs, [])

    assert plan.ok
    assert plan.split_faces == {}
    assert plan.loop_uvs == {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (1.0, 1.0),
        3: (0.0, 1.0),
    }


def test_seam_through_quad_splits_only_that_quad():
    # two quads side by side, a uv cut runs across the first one's diagonal
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
        # v0 and v2 land elsewhere in uv space: the quad straddles the cut
        [(5, 5), (6, 6), (5, 6)],
        [(1, 0), (2, 0), (2, 1)],
        [(1, 0), (2, 1), (1, 1)],
    ]

    plan = plan_transfer(in_pos, in_faces, in_pos, out_faces, out_uvs, [(0, 2)])

    assert plan.ok
    # the untouched quad keeps its four loops, the split one has none
    assert plan.loop_uvs == {
        4: (1.0, 0.0),
        5: (2.0, 0.0),
        6: (2.0, 1.0),
        7: (1.0, 1.0),
    }
    assert plan.split_faces == {
        0: [
            ([0, 1, 2], [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]),
            ([0, 2, 3], [(5.0, 5.0), (6.0, 6.0), (5.0, 6.0)]),
        ]
    }
    # the cut edge only exists once the quad is split
    assert plan.seam_edges == {(0, 2)}


def test_split_pieces_follow_the_input_winding():
    in_pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    in_faces = [[0, 1, 2, 3]]
    # same two triangles, but listed with a rotated corner order
    out_faces = [[2, 0, 1], [3, 0, 2]]
    out_uvs = [
        [(1, 1), (0, 0), (1, 0)],
        [(5, 6), (5, 5), (6, 6)],
    ]

    plan = plan_transfer(in_pos, in_faces, in_pos, out_faces, out_uvs, [])

    assert plan.ok
    assert plan.split_faces == {
        0: [
            ([0, 1, 2], [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]),
            ([0, 2, 3], [(5.0, 5.0), (6.0, 6.0), (5.0, 6.0)]),
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

    result = plan_transfer(in_pos, in_faces, in_pos, out_faces, out_uvs, [])

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
    result = plan_transfer(in_pos, in_faces, [a, b, c], out_faces, out_uvs, [])

    assert not result.ok
    assert result.reason == "ambiguous_geometry"


def test_unmatched_output_face_fails():
    # verts 1, 3, 0 never share an input face
    out_faces = [[1, 3, 0]]
    out_uvs = [[(0, 0), (1, 0), (1, 1)]]

    result = plan_transfer(SQUARE_POS, SQUARE_FACES, SQUARE_POS, out_faces, out_uvs, [])

    assert not result.ok
    assert result.reason == "face_match"
