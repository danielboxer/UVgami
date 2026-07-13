import importlib.util
from pathlib import Path

# loaded from file so importing doesn't touch the blender addon package
spec = importlib.util.spec_from_file_location(
    "addon_uv_transfer", Path(__file__).parents[1] / "src" / "uv_transfer.py"
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
    assert plan.exact_topology
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
    assert not plan.exact_topology
    assert plan.loop_uvs == {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (1.0, 1.0),
        3: (0.0, 1.0),
    }


def test_conflicting_ngon_corner_fails_atomically():
    in_pos = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    in_faces = [[0, 1, 2, 3]]
    out_faces = [[0, 1, 2], [0, 2, 3]]
    # v0 gets a different uv in each triangle
    out_uvs = [
        [(0, 0), (1, 0), (1, 1)],
        [(0.9, 0.9), (1, 1), (0, 1)],
    ]

    result = plan_transfer(in_pos, in_faces, in_pos, out_faces, out_uvs, [])

    assert not result.ok
    assert result.reason == "uv_conflict"


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
