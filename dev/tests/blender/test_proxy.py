import mathutils
import numpy
from blender_fixtures import add_cube, add_dense_sphere, addon, join

make_proxy = addon.src.proxy.make_proxy
mirror_proxy = addon.src.proxy.mirror_proxy
needs_proxy = addon.src.proxy.needs_proxy
part_triangles = addon.src.proxy.part_triangles
triangle_count = addon.src.proxy.triangle_count
mutual_mirror_partners = addon.src.symmetry.mesh_mirror_partners

PROXY_FACES = 500


def test_only_the_part_over_the_budget_is_decimated():
    sphere = add_dense_sphere("sphere")
    obj = join([sphere, add_cube("cube", location=(5, 0, 0))])

    assert needs_proxy(obj, PROXY_FACES)
    assert make_proxy(obj, PROXY_FACES)

    _, _, counts = part_triangles(obj.data)
    assert sorted(counts.tolist()) == [12, PROXY_FACES]


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
