import collections
import math

import numpy
import pytest
from uvgami_cli._pytest_plugin import corners_and_totals
from uvgami_cli.common import load_addon_module

from no_bpy import (
    SURFACE_MODE_CODES,
    EngineError,
    OptcutsRun,
    preseed_job,
    sorted_pair,
)
from no_bpy.engine_run import ToolsCommand
from no_bpy.mirror import mutual_mirror_partners
from no_bpy.preseed import (
    VISIBILITY_SHARE,
    _write_weights,
    classify_job,
    resolved_surface_modes,
    restriction_weights,
)

# from seams.preseed import submesh

flags = load_addon_module("src/engines/optcuts/flags.py")
ENGINE_ARGS = flags.bound_and_weight_args("BALANCED", 3)
SPLIT_TARGET = flags.SPLIT_TARGETS["BALANCED"]


# def test_submesh_compacts_the_subset(unit_cube):
#     verts, faces = unit_cube
#     sub_verts, sub_faces = submesh(verts, faces, [0, 2])
#     assert len(sub_faces) == 2
#     assert len(sub_verts) == 6
#     assert [sub_verts[v] for v in sub_faces[0]] == [verts[v] for v in faces[0]]


@pytest.fixture
def engines(engine_binary, tools):
    # the hard surface command and optcuts' run, found like the addon finds them
    return tools, OptcutsRun([engine_binary("optcuts")], None, ENGINE_ARGS)


def run_preseed(engines, tmp_path, positions, faces, surface_modes, plane=None):
    return preseed_job(
        *engines,
        tmp_path,
        numpy.asarray(positions, dtype=numpy.float64),
        *corners_and_totals(faces),
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
            (ToolsCommand(missing, None), OptcutsRun(missing, None, ENGINE_ARGS)),
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


@pytest.mark.smoke
def test_preseed_leaves_faces_that_are_not_hard_alone(tmp_path, engines, unit_cube):
    # two loose quads, a preseed holds whole loose parts, never a slice
    square = unit_cube[0][:4]
    verts = square + [(x + 5.0, y, z) for x, y, z in square]
    faces = [(0, 1, 2, 3), (4, 5, 6, 7)]
    modes = [SURFACE_MODE_CODES["HARD"], SURFACE_MODE_CODES["ORGANIC"]]
    _, _, flattened = run_preseed(engines, tmp_path, verts, faces, modes)
    assert flattened.tolist() == [0]


@pytest.mark.smoke
def test_a_closed_part_no_seam_touches_is_not_flattened(tmp_path, engines, unit_cube):
    # a closed part no seam touches folds onto itself in the flatten
    cube_verts, cube_faces = unit_cube
    sphere_verts, sphere_faces = uv_sphere()
    verts = cube_verts + [(x + 5.0, y, z) for x, y, z in sphere_verts]
    faces = cube_faces + [
        tuple(v + len(cube_verts) for v in face) for face in sphere_faces
    ]
    _, _, flattened = run_preseed(engines, tmp_path, verts, faces, all_hard(faces))
    assert flattened.tolist() == list(range(len(cube_faces)))


@pytest.mark.smoke
def test_preseed_mirrors_close_the_seam_set(tmp_path, engines, unit_cube):
    verts, faces = unit_cube
    # the unit cube is symmetric across x = 0.5
    mirror = dict(enumerate([1, 0, 3, 2, 5, 4, 7, 6]))
    seams, _, _ = run_preseed(
        engines, tmp_path, verts, faces, all_hard(faces), plane=(0, mirror)
    )
    assert seams
    assert {sorted_pair(mirror[a], mirror[b]) for a, b in seams} == seams


SymmetricPreseed = collections.namedtuple(
    "SymmetricPreseed", "seams flattened partners"
)


def symmetric_preseed(engines, tmp_path, positions, faces, surface_modes):
    low, high = positions.min(axis=0), positions.max(axis=0)
    center = (low + high) / 2
    diagonal = numpy.linalg.norm(high - low)
    # drifted off an exact mirror, as a mesh that ships whole is
    positions[0] += 0.001 * diagonal
    partners = mutual_mirror_partners(engines[0], tmp_path, positions, center, "X")
    seams, _, flattened = run_preseed(
        engines, tmp_path, positions, faces, surface_modes, plane=(0, partners)
    )
    return SymmetricPreseed(seams, flattened, partners)


def mirrored_share(seams, partners):
    mirrored = {
        sorted_pair(partners[a], partners[b])
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


@pytest.fixture
def classify_hard(tools, tmp_path):
    # the hard surface exe's pick per face with every part on Auto
    def run(verts, faces):
        modes = numpy.full(len(faces), SURFACE_MODE_CODES["AUTO"])
        resolved = resolved_surface_modes(tools, tmp_path, verts, faces, modes)
        return resolved == SURFACE_MODE_CODES["HARD"]

    return run


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


def capped_tube(sides=32, height=2.0):
    verts, faces = tube(sides, height)
    top = len(verts)
    verts.append([0.0, 0.0, height])
    for i in range(sides):
        faces.append([2 * i + 1, 2 * ((i + 1) % sides) + 1, top])
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
    for ring in range(rings + 1):
        polar = math.pi * ring / rings
        for side in range(sides):
            azimuth = 2 * math.pi * side / sides
            x = math.sin(polar) * math.cos(azimuth)
            y = math.sin(polar) * math.sin(azimuth)
            z = math.cos(polar)
            radius = (
                1
                + 0.25 * math.sin(3 * x + 0.5) * math.cos(2 * y + 1.3)
                + 0.2 * math.sin(2.5 * z + 0.7)
            )
            verts.append([radius * x, 1.4 * radius * y, radius * z])
    for ring in range(rings):
        for side in range(sides):
            a, b = ring * sides + side, ring * sides + (side + 1) % sides
            faces.append([a, b, b + sides, a + sides])
    return verts, faces


@pytest.mark.smoke
@pytest.mark.parametrize("rings, sides", [(24, 48), (12, 24)])
def test_lumpy_blob_reads_organic(rings, sides, classify_hard):
    verts, faces = lumpy_blob(rings, sides)
    assert not classify_hard(verts, faces).any()


@pytest.mark.smoke
def test_classify_resolves_only_auto_parts(fixture_mesh, tools, tmp_path):
    # an auto part takes the classifier's pick, a part set by hand keeps its mode
    cube = fixture_mesh("cube-bevel2")
    blob = lumpy_blob()
    parts = [
        (cube, "AUTO", "HARD"),
        (blob, "AUTO", "ORGANIC"),
        (cube, "ORGANIC", "ORGANIC"),
        (blob, "HARD", "HARD"),
    ]
    positions, all_faces, modes, expected = [], [], [], []
    for index, ((verts, faces), mode, resolved) in enumerate(parts):
        base = len(positions)
        positions += [[x + 5.0 * index, y, z] for x, y, z in verts]
        all_faces += [[base + v for v in face] for face in faces]
        modes += [SURFACE_MODE_CODES[mode]] * len(faces)
        expected += [SURFACE_MODE_CODES[resolved]] * len(faces)

    result = classify_job(
        tools,
        tmp_path,
        numpy.array(positions),
        *corners_and_totals(all_faces),
        numpy.array(modes),
    )

    assert result.tolist() == expected


@pytest.mark.smoke
def test_tiny_part_reads_hard(classify_hard):
    # the blob is under a hundredth of the model's area beside a 100 by 100 quad
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
