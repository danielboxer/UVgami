import importlib
import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy
import pytest

# loaded from file, the addon package imports bpy
PKG = Path(__file__).parents[2] / "src" / "seams"
spec = importlib.util.spec_from_file_location(
    "seams", PKG / "__init__.py", submodule_search_locations=[str(PKG)]
)
sys.modules["seams"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sys.modules["seams"])
from seams import (  # noqa: E402
    SURFACE_MODE_CODES,
    EngineError,
    OptcutsRun,
    pair,
    preseed_job,
)
from seams.islands import SPLIT_TARGET  # noqa: E402
from seams.preseed import submesh  # noqa: E402

# the addon package imports bpy, a bare package lets similar import seams
package = types.ModuleType("addon_src")
package.__path__ = [str(Path(__file__).parents[2] / "src")]
sys.modules.setdefault("addon_src", package)
similar = importlib.import_module("addon_src.similar")

FIXTURES = Path(__file__).parent / "fixtures"
BUNDLED = Path(__file__).parents[2] / "engine-builds" / "windows" / "optcuts.exe"
HARDSURFACE = BUNDLED.with_name("hardsurface.exe")
# the addon's balanced bound and default seam weight
ENGINE_ARGS = ["-u", "4.2", "-s", "100"]

CUBE_VERTS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
]
CUBE_FACES = [
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
]


def read_obj(path):
    rows = [line.split() for line in path.read_text().splitlines()]
    positions = numpy.array(
        [[float(x) for x in r[1:4]] for r in rows if r and r[0] == "v"]
    )
    faces = [
        [int(t.split("/")[0]) - 1 for t in r[1:]] for r in rows if r and r[0] == "f"
    ]
    return positions, faces


def test_submesh_compacts_and_remaps_seams():
    sub_verts, sub_faces, sub_seams = submesh(
        CUBE_VERTS, CUBE_FACES, [0, 2], {(0, 1), (6, 7)}
    )
    assert len(sub_faces) == 2
    assert len(sub_verts) == 6
    # (0, 1) survives remapped, (6, 7) is outside the subset
    assert len(sub_seams) == 1
    (a, b) = next(iter(sub_seams))
    assert sub_verts[a] == CUBE_VERTS[0]
    assert sub_verts[b] == CUBE_VERTS[1]


needs_engines = pytest.mark.skipif(
    not (BUNDLED.is_file() and HARDSURFACE.is_file()), reason="engine builds missing"
)


def mesh_arrays(faces):
    corners = numpy.array([v for face in faces for v in face], dtype=numpy.int64)
    totals = numpy.array([len(face) for face in faces], dtype=numpy.int64)
    return corners, totals


# the seams, the flattened faces' corner uvs and the flattened faces, or None
def run_preseed(tmp_path, positions, faces, surface_modes, plane=None):
    return preseed_job(
        [HARDSURFACE],
        OptcutsRun([BUNDLED], None, ENGINE_ARGS),
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


def test_a_missing_hard_surface_engine_raises(tmp_path):
    with pytest.raises(EngineError):
        preseed_job(
            [tmp_path / "missing.exe"],
            OptcutsRun([BUNDLED], None, ENGINE_ARGS),
            tmp_path,
            numpy.asarray(CUBE_VERTS),
            *mesh_arrays(CUBE_FACES),
            None,
            None,
            None,
            numpy.asarray(all_hard(CUBE_FACES)),
            30,
            SPLIT_TARGET,
        )


@pytest.mark.smoke
@needs_engines
def test_preseed_flattens_a_cube_inside_the_unit_square(tmp_path):
    seams, uvs, flattened = run_preseed(
        tmp_path, CUBE_VERTS, CUBE_FACES, all_hard(CUBE_FACES)
    )
    assert seams
    assert flattened.tolist() == list(range(len(CUBE_FACES)))
    assert uvs.shape == (4 * len(CUBE_FACES), 2)
    assert ((uvs >= 0.0) & (uvs <= 1.0)).all()


# two loose quads, a preseed holds whole loose parts, never a slice
@pytest.mark.smoke
@needs_engines
def test_preseed_leaves_faces_that_are_not_hard_alone(tmp_path):
    verts = CUBE_VERTS[:4] + [(x + 5.0, y, z) for x, y, z in CUBE_VERTS[:4]]
    faces = [(0, 1, 2, 3), (4, 5, 6, 7)]
    modes = [SURFACE_MODE_CODES["HARD"], SURFACE_MODE_CODES["ORGANIC"]]
    _, _, flattened = run_preseed(tmp_path, verts, faces, modes)
    assert flattened.tolist() == [0]


# a closed part no seam touches folds onto itself in the flatten
@pytest.mark.smoke
@needs_engines
def test_a_closed_part_no_seam_touches_is_not_flattened(tmp_path):
    sphere_verts, sphere_faces = uv_sphere()
    verts = CUBE_VERTS + [(x + 5.0, y, z) for x, y, z in sphere_verts]
    faces = list(CUBE_FACES) + [
        tuple(v + len(CUBE_VERTS) for v in face) for face in sphere_faces
    ]
    _, _, flattened = run_preseed(tmp_path, verts, faces, all_hard(faces))
    assert flattened.tolist() == list(range(len(CUBE_FACES)))


# the unit cube is symmetric across x = 0.5
@pytest.mark.smoke
@needs_engines
def test_preseed_mirrors_close_the_seam_set(tmp_path):
    mirror = dict(enumerate([1, 0, 3, 2, 5, 4, 7, 6]))
    seams, _, _ = run_preseed(
        tmp_path, CUBE_VERTS, CUBE_FACES, all_hard(CUBE_FACES), plane=(0, mirror)
    )
    assert seams
    assert {pair(mirror[a], mirror[b]) for a, b in seams} == seams


# returns the seams, the flattened faces and the partner map
def symmetric_preseed(tmp_path, positions, faces, surface_modes):
    low, high = positions.min(axis=0), positions.max(axis=0)
    center = (low + high) / 2
    diagonal = numpy.linalg.norm(high - low)
    # drifted off an exact mirror, as a mesh that ships whole is
    positions[0] += 0.001 * diagonal
    partner = similar._mutual_partners(
        positions, center, "X", similar.SYMMETRY_CELL * diagonal
    )
    partners = similar.partner_map(partner)
    seams, _, flattened = run_preseed(
        tmp_path, positions, faces, surface_modes, plane=(0, partners)
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
@needs_engines
def test_symmetric_preseed_mirrors_the_engine_cuts(tmp_path):
    positions, faces = read_obj(FIXTURES / "cylinder.obj")
    modes = numpy.full(len(faces), SURFACE_MODE_CODES["ORGANIC"])

    seams, flattened, partners = symmetric_preseed(tmp_path, positions, faces, modes)

    assert len(flattened) == len(faces)
    assert mirrored_share(seams, partners) > 0.9


@pytest.mark.smoke
@needs_engines
def test_symmetric_preseed_mirrors_hard_and_organic_parts_together(tmp_path):
    cylinder, cylinder_faces = read_obj(FIXTURES / "cylinder.obj")
    cube, cube_faces = read_obj(FIXTURES / "cube.obj")
    # both parts straddle the plane at x = 0
    positions = numpy.concatenate([cylinder, cube + [0.0, 4.0, 0.0]])
    faces = cylinder_faces + [[v + len(cylinder) for v in face] for face in cube_faces]
    modes = numpy.array(
        [SURFACE_MODE_CODES["ORGANIC"]] * len(cylinder_faces)
        + [SURFACE_MODE_CODES["HARD"]] * len(cube_faces)
    )

    seams, flattened, partners = symmetric_preseed(tmp_path, positions, faces, modes)

    assert len(flattened) == len(faces)
    cube_vertices = set(range(len(cylinder), len(positions)))
    assert any(a in cube_vertices for a, _ in seams)
    assert any(a not in cube_vertices for a, _ in seams)
    assert mirrored_share(seams, partners) > 0.9
