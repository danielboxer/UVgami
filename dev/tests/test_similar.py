import importlib.util
from pathlib import Path

import numpy

# loaded from file so importing doesn't touch the blender addon package
spec = importlib.util.spec_from_file_location(
    "addon_similar", Path(__file__).parents[2] / "src" / "similar.py"
)
similar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(similar)

ROTATE_Z = numpy.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
MIRROR_X = numpy.diag([-1.0, 1.0, 1.0])
IDENTITY = numpy.identity(4)


class Elements:
    def __init__(self, count, values):
        self.count = count
        self.values = values

    def __len__(self):
        return self.count

    def foreach_get(self, attr, array):
        array[:] = self.values[attr]


class FakeMesh:
    def __init__(self, coords, faces):
        corners = [v for face in faces for v in face]
        self.vertices = Elements(len(coords), {"co": [c for co in coords for c in co]})
        self.loops = Elements(len(corners), {"vertex_index": corners})
        self.polygons = Elements(
            len(faces), {"loop_total": [len(face) for face in faces]}
        )


class FakeObject:
    def __init__(self, coords, faces):
        self.data = FakeMesh(coords, faces)
        self.matrix_world = IDENTITY


TETRA_COORDS = [(0, 0, 0), (1, 0, 0), (0, 2, 0), (0, 0, 3)]
TETRA_FACES = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]


def moved_coords(rotation, translation):
    return [tuple(rotation @ co + translation) for co in TETRA_COORDS]


def test_rigid_fit_recovers_rotation():
    source = numpy.array(TETRA_COORDS, dtype=float)
    target = source @ ROTATE_Z.T + (5.0, 2.0, -1.0)

    matrix, error = similar.rigid_fit(source, target)

    assert error < 1e-9
    assert numpy.allclose(matrix[:3, :3], ROTATE_Z)
    assert numpy.allclose(matrix[:3, 3], (5.0, 2.0, -1.0))


def test_find_twins_matches_moved_copy():
    rep = FakeObject(TETRA_COORDS, TETRA_FACES)
    twin = FakeObject(moved_coords(ROTATE_Z, numpy.array([4.0, 0, 0])), TETRA_FACES)

    twins = similar.find_twins([rep, twin])

    assert set(twins) == {twin}
    matched_rep, matrix, exact = twins[twin]
    assert matched_rep is rep
    assert exact
    moved = numpy.array(TETRA_COORDS, dtype=float) @ matrix[:3, :3].T + matrix[:3, 3]
    assert numpy.allclose(
        moved, numpy.array(twin.data.vertices.values["co"]).reshape(-1, 3)
    )


def test_find_twins_matches_mirrored_copy():
    rep = FakeObject(TETRA_COORDS, TETRA_FACES)
    twin = FakeObject(moved_coords(MIRROR_X, numpy.array([9.0, 1, 2])), TETRA_FACES)

    twins = similar.find_twins([rep, twin])

    assert set(twins) == {twin}
    _, matrix, _ = twins[twin]
    assert numpy.linalg.det(matrix[:3, :3]) < 0


def test_find_twins_rejects_deformed_copy():
    rep = FakeObject(TETRA_COORDS, TETRA_FACES)
    stretched = [(x * 1.1, y, z) for x, y, z in TETRA_COORDS]
    other = FakeObject(stretched, TETRA_FACES)

    assert similar.find_twins([rep, other]) == {}


def reordered(coords, faces, order):
    new_coords = [None] * len(coords)
    for old, new in enumerate(order):
        new_coords[new] = coords[old]
    new_faces = [tuple(order[v] for v in face) for face in faces]
    return new_coords, new_faces


def test_find_twins_matches_reordered_copy():
    rep = FakeObject(TETRA_COORDS, TETRA_FACES)
    coords, faces = reordered(
        moved_coords(ROTATE_Z, numpy.array([4.0, 0, 0])), TETRA_FACES, [2, 0, 3, 1]
    )
    twin = FakeObject(coords, faces[::-1])

    twins = similar.find_twins([rep, twin])

    assert set(twins) == {twin}
    matched_rep, matrix, exact = twins[twin]
    assert matched_rep is rep
    assert not exact
    moved = numpy.array(TETRA_COORDS, dtype=float) @ matrix[:3, :3].T + matrix[:3, 3]
    twin_coords = numpy.array(coords, dtype=float)
    assert sorted(map(tuple, numpy.round(moved, 6))) == sorted(
        map(tuple, numpy.round(twin_coords, 6))
    )


def test_find_twins_matches_mirrored_reordered_copy():
    rep = FakeObject(TETRA_COORDS, TETRA_FACES)
    coords, faces = reordered(
        moved_coords(MIRROR_X, numpy.array([9.0, 1, 2])), TETRA_FACES, [1, 3, 0, 2]
    )
    twin = FakeObject(coords, faces)

    twins = similar.find_twins([rep, twin])

    assert set(twins) == {twin}
    _, matrix, _ = twins[twin]
    assert numpy.linalg.det(matrix[:3, :3]) < 0


def octagon_tube():
    coords = []
    for level in (0.0, 1.0):
        for step in range(8):
            angle = step * numpy.pi / 4
            coords.append((numpy.cos(angle), numpy.sin(angle), level))
    faces = []
    for step in range(8):
        after = (step + 1) % 8
        faces.append((step, after, after + 8, step + 8))
    return coords, faces


# pca axes are arbitrary in the tube's round plane, needs the rotation search
def test_find_twins_matches_symmetric_rotated_copy():
    coords, faces = octagon_tube()
    rep = FakeObject(coords, faces)
    turn = numpy.pi / 6
    rotation = numpy.array(
        [
            [numpy.cos(turn), -numpy.sin(turn), 0.0],
            [numpy.sin(turn), numpy.cos(turn), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = [tuple(rotation @ co + (2.0, -1.0, 0.5)) for co in coords]
    order = [(index * 5 + 3) % 16 for index in range(16)]
    twin_coords, twin_faces = reordered(moved, faces, order)
    twin = FakeObject(twin_coords, twin_faces)

    twins = similar.find_twins([rep, twin])

    assert set(twins) == {twin}
    _, matrix, _ = twins[twin]
    fitted = numpy.array(coords) @ matrix[:3, :3].T + matrix[:3, 3]
    assert sorted(map(tuple, numpy.round(fitted, 6))) == sorted(
        map(tuple, numpy.round(numpy.array(twin_coords), 6))
    )


# same quad region and vertex positions, opposite diagonal: not a duplicate
def test_find_twins_rejects_different_topology():
    coords = [(0, 0, 0), (3, 0, 0), (2, 1, 0), (0, 1, 0)]
    rep = FakeObject(coords, [(0, 1, 2), (0, 2, 3)])
    other = FakeObject(coords, [(0, 1, 3), (1, 2, 3)])

    assert similar.find_twins([rep, other]) == {}


SQUARE = [(0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2)]


def test_find_stacks_groups_exact_copies():
    uvs = [SQUARE, SQUARE, [(0.5, 0.5), (0.7, 0.5), (0.7, 0.7), (0.5, 0.7)]]
    groups = [[0], [1], [2]]

    stacks = similar.find_stacks(groups, uvs)

    assert stacks == [([0], [[1]])]


def test_find_stacks_ignores_partial_overlap():
    shifted = [(u + 0.05, v) for u, v in SQUARE]
    uvs = [SQUARE, shifted]

    assert similar.find_stacks([[0], [1]], uvs) == []


def test_write_twin_output_transforms_vertices(tmp_path):
    source = tmp_path / "rep.obj"
    source.write_text("o rep\nv 1 0 0\nv 0 1 0\nv 0 0 1\nvt 0 0\nf 1/1 2/1 3/1\n")
    target = tmp_path / "twin.obj"
    matrix = numpy.identity(4)
    matrix[:3, :3] = ROTATE_Z
    matrix[:3, 3] = (10.0, 0.0, 0.0)

    similar.write_twin_output(source, target, matrix)
    lines = target.read_text().splitlines()

    assert lines[1] == "v 10.000000000 1.000000000 0.000000000"
    assert lines[4] == "vt 0 0"
    assert lines[5] == "f 1/1 2/1 3/1"


def test_write_twin_output_flips_mirrored_winding(tmp_path):
    source = tmp_path / "rep.obj"
    source.write_text("v 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\n")
    target = tmp_path / "twin.obj"
    matrix = numpy.identity(4)
    matrix[:3, :3] = MIRROR_X

    similar.write_twin_output(source, target, matrix)
    lines = target.read_text().splitlines()

    assert lines[0] == "v -1.000000000 0.000000000 0.000000000"
    assert lines[3] == "f 3 2 1"
