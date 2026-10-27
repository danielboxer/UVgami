import math

import numpy
import pytest

from seams import (
    SURFACE_MODE_CODES,
    EngineError,
    OptcutsRun,
    pair,
    preseed_job,
)
from seams.islands import SPLIT_TARGET
from seams.mirror import mirror_partners
from seams.preseed import submesh

# the addon's balanced bound and default seam weight
ENGINE_ARGS = ["-u", "4.2", "-s", "100"]


def test_submesh_compacts_the_subset(unit_cube):
    verts, faces = unit_cube
    sub_verts, sub_faces = submesh(verts, faces, [0, 2])
    assert len(sub_faces) == 2
    assert len(sub_verts) == 6
    assert [sub_verts[v] for v in sub_faces[0]] == [verts[v] for v in faces[0]]


# the hard surface command and optcuts' run, found like the addon finds them
@pytest.fixture
def engines(engine_binary):
    return [engine_binary("uvgami-tools")], OptcutsRun(
        [engine_binary("optcuts")], None, ENGINE_ARGS
    )


def mesh_arrays(faces):
    corners = numpy.array([v for face in faces for v in face], dtype=numpy.int64)
    totals = numpy.array([len(face) for face in faces], dtype=numpy.int64)
    return corners, totals


# the seams, the flattened faces' corner uvs and the flattened faces, or None
def run_preseed(engines, tmp_path, positions, faces, surface_modes, plane=None):
    return preseed_job(
        *engines,
        tmp_path,
        numpy.asarray(positions, dtype=numpy.float64),
        *mesh_arrays(faces),
        None,
        None,
        plane,
        numpy.asarray(surface_modes),
        30,
        SPLIT_TARGET,
    )


def all_hard(faces):
    return [SURFACE_MODE_CODES["HARD"]] * len(faces)


def uv_sphere(rings=16, segments=32):
    verts = [(0.0, 0.0, 1.0)]
    for ring in range(1, rings):
        theta = math.pi * ring / rings
        for segment in range(segments):
            phi = 2 * math.pi * segment / segments
            verts.append(
                (
                    math.sin(theta) * math.cos(phi),
                    math.sin(theta) * math.sin(phi),
                    math.cos(theta),
                )
            )
    verts.append((0.0, 0.0, -1.0))
    bottom = len(verts) - 1

    def at(ring, segment):
        return 1 + (ring - 1) * segments + segment % segments

    faces = [(0, at(1, s), at(1, s + 1)) for s in range(segments)]
    for ring in range(1, rings - 1):
        for s in range(segments):
            faces.append(
                (at(ring, s), at(ring + 1, s), at(ring + 1, s + 1), at(ring, s + 1))
            )
    faces += [(bottom, at(rings - 1, s + 1), at(rings - 1, s)) for s in range(segments)]
    return verts, faces


def test_a_missing_hard_surface_engine_raises(tmp_path, unit_cube):
    verts, faces = unit_cube
    missing = [tmp_path / "missing.exe"]
    with pytest.raises(EngineError):
        run_preseed(
            (missing, OptcutsRun(missing, None, ENGINE_ARGS)),
            tmp_path,
            verts,
            faces,
            all_hard(faces),
        )


@pytest.mark.smoke
def test_preseed_flattens_a_cube_inside_the_unit_square(tmp_path, engines, unit_cube):
    verts, faces = unit_cube
    seams, uvs, flattened = run_preseed(
        engines, tmp_path, verts, faces, all_hard(faces)
    )
    assert seams
    assert flattened.tolist() == list(range(len(faces)))
    assert uvs.shape == (4 * len(faces), 2)
    assert ((uvs >= 0.0) & (uvs <= 1.0)).all()


# two loose quads, a preseed holds whole loose parts, never a slice
@pytest.mark.smoke
def test_preseed_leaves_faces_that_are_not_hard_alone(tmp_path, engines, unit_cube):
    square = unit_cube[0][:4]
    verts = square + [(x + 5.0, y, z) for x, y, z in square]
    faces = [(0, 1, 2, 3), (4, 5, 6, 7)]
    modes = [SURFACE_MODE_CODES["HARD"], SURFACE_MODE_CODES["ORGANIC"]]
    _, _, flattened = run_preseed(engines, tmp_path, verts, faces, modes)
    assert flattened.tolist() == [0]


# a closed part no seam touches folds onto itself in the flatten
@pytest.mark.smoke
def test_a_closed_part_no_seam_touches_is_not_flattened(tmp_path, engines, unit_cube):
    cube_verts, cube_faces = unit_cube
    sphere_verts, sphere_faces = uv_sphere()
    verts = cube_verts + [(x + 5.0, y, z) for x, y, z in sphere_verts]
    faces = cube_faces + [
        tuple(v + len(cube_verts) for v in face) for face in sphere_faces
    ]
    _, _, flattened = run_preseed(engines, tmp_path, verts, faces, all_hard(faces))
    assert flattened.tolist() == list(range(len(cube_faces)))


# the unit cube is symmetric across x = 0.5
@pytest.mark.smoke
def test_preseed_mirrors_close_the_seam_set(tmp_path, engines, unit_cube):
    verts, faces = unit_cube
    mirror = dict(enumerate([1, 0, 3, 2, 5, 4, 7, 6]))
    seams, _, _ = run_preseed(
        engines, tmp_path, verts, faces, all_hard(faces), plane=(0, mirror)
    )
    assert seams
    assert {pair(mirror[a], mirror[b]) for a, b in seams} == seams


# returns the seams, the flattened faces and the partner map
def symmetric_preseed(engines, tmp_path, positions, faces, surface_modes):
    low, high = positions.min(axis=0), positions.max(axis=0)
    center = (low + high) / 2
    diagonal = numpy.linalg.norm(high - low)
    # drifted off an exact mirror, as a mesh that ships whole is
    positions[0] += 0.001 * diagonal
    partners = mirror_partners(engines[0], tmp_path, positions, center, "X")
    seams, _, flattened = run_preseed(
        engines, tmp_path, positions, faces, surface_modes, plane=(0, partners)
    )
    return seams, flattened, partners


def mirrored_share(seams, partners):
    mirrored = {
        pair(partners[a], partners[b])
        for a, b in seams
        if a in partners and b in partners
    }
    return len(mirrored & seams) / len(seams)


@pytest.mark.smoke
def test_symmetric_preseed_mirrors_the_engine_cuts(tmp_path, engines, fixture_mesh):
    verts, faces = fixture_mesh("cylinder")
    modes = numpy.full(len(faces), SURFACE_MODE_CODES["ORGANIC"])

    seams, flattened, partners = symmetric_preseed(
        engines, tmp_path, numpy.array(verts), faces, modes
    )

    assert len(flattened) == len(faces)
    assert mirrored_share(seams, partners) > 0.9


@pytest.mark.smoke
def test_symmetric_preseed_mirrors_hard_and_organic_parts_together(
    tmp_path, engines, fixture_mesh
):
    cylinder, cylinder_faces = fixture_mesh("cylinder")
    cube, cube_faces = fixture_mesh("cube")
    # both parts straddle the plane at x = 0
    positions = numpy.concatenate(
        [numpy.array(cylinder), numpy.array(cube) + [0.0, 4.0, 0.0]]
    )
    faces = cylinder_faces + [[v + len(cylinder) for v in face] for face in cube_faces]
    modes = numpy.array(
        [SURFACE_MODE_CODES["ORGANIC"]] * len(cylinder_faces)
        + [SURFACE_MODE_CODES["HARD"]] * len(cube_faces)
    )

    seams, flattened, partners = symmetric_preseed(
        engines, tmp_path, positions, faces, modes
    )

    assert len(flattened) == len(faces)
    cube_vertices = set(range(len(cylinder), len(positions)))
    assert any(a in cube_vertices for a, _ in seams)
    assert any(a not in cube_vertices for a, _ in seams)
    assert mirrored_share(seams, partners) > 0.9
