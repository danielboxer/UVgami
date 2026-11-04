import bpy
import mathutils
import numpy
import pytest
from blender_fixtures import add_cube, add_dense_sphere, addon, join

make_proxy = addon.src.proxy.make_proxy
mirror_proxy = addon.src.proxy.mirror_proxy
needs_proxy = addon.src.proxy.needs_proxy
part_triangles = addon.src.proxy.part_triangles
proxy_trigger_faces = addon.src.proxy.proxy_trigger_faces
parts_for_auto_proxy = addon.src.proxy.parts_for_auto_proxy
triangle_count = addon.src.proxy.triangle_count
mutual_mirror_partners = addon.src.symmetry.mesh_mirror_partners

PROXY_FACES = 500
SURFACE_MODE_CODES = addon.src.no_bpy.SURFACE_MODE_CODES


@pytest.fixture(autouse=True)
def proxy_budget(proxy_target):
    proxy_target(PROXY_FACES)


def test_only_the_part_over_the_budget_is_decimated():
    sphere = add_dense_sphere("sphere")
    obj = join([sphere, add_cube("cube", location=(5, 0, 0))])

    assert needs_proxy(obj, PROXY_FACES)
    assert make_proxy(obj, PROXY_FACES)

    _, _, counts = part_triangles(obj.data)
    assert sorted(counts.tolist()) == [12, PROXY_FACES]


def test_on_forces_the_proxy_down_to_the_target_and_auto_uses_the_trigger():
    props = bpy.context.scene.uvgami

    props.proxy_mode = "ON"
    assert proxy_trigger_faces(props) == PROXY_FACES

    props.proxy_mode = "AUTO"
    assert proxy_trigger_faces(props) == addon.src.proxy.PROXY_AUTO_TRIGGER_FACES


def test_a_part_under_the_trigger_stays_at_full_resolution():
    obj = add_dense_sphere("sphere")
    triangles = triangle_count(obj)

    assert not make_proxy(obj, triangles)
    assert triangle_count(obj) == triangles


def test_a_part_over_the_trigger_lands_on_the_target():
    obj = add_dense_sphere("sphere")

    assert make_proxy(obj, triangle_count(obj) - 1)
    assert triangle_count(obj) == PROXY_FACES


def test_a_hard_surface_part_is_not_decimated():
    sphere = add_dense_sphere("sphere")
    sphere_faces = len(sphere.data.polygons)
    obj = join([sphere, add_cube("cube", location=(5, 0, 0))])
    parts = part_triangles(obj.data)
    modes = numpy.full(len(obj.data.polygons), SURFACE_MODE_CODES["ORGANIC"])
    centers = numpy.array([polygon.center.x for polygon in obj.data.polygons])
    modes[centers < 2] = SURFACE_MODE_CODES["HARD"]
    assert (modes == SURFACE_MODE_CODES["HARD"]).sum() == sphere_faces

    candidates = parts_for_auto_proxy(parts, obj.data, modes)

    assert not make_proxy(obj, PROXY_FACES, candidates)
    assert len(obj.data.polygons) == len(modes)


def test_a_hard_surface_part_over_the_limit_stays_a_candidate():
    obj = add_dense_sphere("sphere")
    limit = addon.src.proxy.PROXY_AUTO_HARD_SURFACE_FACES
    modes = numpy.full(len(obj.data.polygons), SURFACE_MODE_CODES["HARD"])
    parts = part_triangles(obj.data)._replace(triangle_counts=numpy.array([limit + 1]))

    assert parts_for_auto_proxy(parts, obj.data, modes).triangle_counts[0] == limit + 1


def test_parts_all_under_the_budget_are_left_alone():
    obj = join([add_cube(f"cube{i}", location=(3 * i, 0, 0)) for i in range(3)])

    assert triangle_count(obj) > 20
    assert not needs_proxy(obj, 20)
    assert not make_proxy(obj, 20)
    assert triangle_count(obj) == 36


def test_a_single_part_lands_on_the_budget():
    obj = add_dense_sphere("sphere")

    assert needs_proxy(obj, PROXY_FACES)
    assert make_proxy(obj, PROXY_FACES)
    assert triangle_count(obj) == PROXY_FACES


def test_a_part_at_the_face_count_keeps_its_faces():
    obj = add_dense_sphere("sphere")
    triangles = triangle_count(obj)

    assert not needs_proxy(obj, triangles)
    assert not make_proxy(obj, triangles)
    assert triangle_count(obj) == triangles


def test_a_mirrored_proxy_has_a_partner_for_every_vertex():
    obj = add_dense_sphere("sphere")
    make_proxy(obj, PROXY_FACES)
    center = mathutils.Vector((0, 0, 0))
    assert len(mutual_mirror_partners(obj.data, center, "X")) < len(obj.data.vertices)
    faces = len(obj.data.polygons)

    plan = addon.src.symmetry.SymmetryPlan("BISECT", "X", center, 1e-4, None, 1)
    source_faces = mirror_proxy(obj, plan)

    partners = mutual_mirror_partners(obj.data, center, "X")
    assert len(partners) == len(obj.data.vertices)
    # each kept face and its mirror image come from one decimated face
    uses = numpy.bincount(source_faces, minlength=faces)
    assert set(uses[uses > 0].tolist()) == {2}
