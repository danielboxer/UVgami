import math

import bpy
import mathutils
import mathutils.kdtree
import numpy
import pytest
from blender_fixtures import addon, manager, needs_engine, needs_xatlas


def uvs_by_vertex(mesh, faces=None):
    layer = mesh.uv_layers.active
    polygons = mesh.polygons if faces is None else [mesh.polygons[fi] for fi in faces]
    result = {}
    for polygon in polygons:
        for loop_index in polygon.loop_indices:
            uv = layer.data[loop_index].uv
            result.setdefault(mesh.loops[loop_index].vertex_index, set()).add(
                (round(uv.x, 6), round(uv.y, 6))
            )
    return result


def plane_vertices(mesh, axis):
    coords = [v.co[axis] for v in mesh.vertices]
    center = (min(coords) + max(coords)) / 2
    tolerance = 1e-4 * (max(coords) - min(coords))
    return [v.index for v in mesh.vertices if abs(v.co[axis] - center) <= tolerance]


def test_auto_takes_the_first_axis_the_mesh_mirrors_across():
    bpy.ops.mesh.primitive_cube_add()
    cube = bpy.context.active_object
    center = mathutils.Vector((0, 0, 0))
    assert addon.src.similar.symmetric_axis(cube.data, center)[:2] == ("X", True)

    # a mesh that only mirrors across Y
    cube.data.vertices[0].co.x += 0.1
    cube.data.vertices[2].co.x += 0.1
    assert addon.src.similar.symmetric_axis(cube.data, center)[:2] == ("Y", True)

    cube.data.vertices[0].co.x += 0.1
    assert addon.src.similar.symmetric_axis(cube.data, center) is None


def auto_plan(mesh):
    job = addon.src.job
    center, diagonal = job.symmetry_center(mesh)
    tolerance = addon.src.utils.geometry.PLANE_SNAP * diagonal
    return job.mirror_plan(mesh, center, tolerance)


def test_auto_keeps_a_mesh_the_cut_would_add_edges_to_whole(load_obj):
    bpy.ops.mesh.primitive_cube_add()
    assert auto_plan(bpy.context.active_object.data)[:2] == ("X", True)

    (cylinder,) = load_obj("cylinder")
    assert auto_plan(cylinder.data)[:2] == ("X", False)


def test_auto_keeps_a_drifted_mesh_whole(load_obj):
    (cylinder,) = load_obj("cylinder")
    mesh = cylinder.data
    _, diagonal = addon.src.job.symmetry_center(mesh)
    # on every axis, since an exact axis is taken before an approximate one
    mesh.vertices[0].co += mathutils.Vector((1, 1, 1)) * 0.001 * diagonal
    assert auto_plan(mesh)[:2] == ("X", True)


def test_auto_skips_a_mesh_with_part_overrides(load_obj):
    (cylinder,) = load_obj("cylinder")
    mesh = cylinder.data
    name = addon.src.part_overrides.KEEP_ATTRIBUTE
    mesh.attributes.new(name, "INT", "FACE").data[0].value = 1
    assert auto_plan(mesh) is None


@needs_engine
def test_symmetry_welds_the_uv_map_across_the_plane(load_obj, unwrap, outputs):
    load_obj("cylinder")
    props = bpy.context.scene.uvgami
    # the pack moves islands apart, which hides how the halves meet
    props.pack_after_unwrap = False
    unwrap()

    assert manager.error_messages == []
    assert manager.summary_failed is False, manager.summary
    assert_welded_across_x(outputs()["cylinder_unwrapped"].data)


@needs_engine
@needs_xatlas
def test_xatlas_gives_the_negative_side_the_mirrored_uvs(load_obj, unwrap, outputs):
    load_obj("cylinder")
    props = bpy.context.scene.uvgami
    props.pack_after_unwrap = False
    unwrap("XATLAS")

    assert manager.error_messages == []
    assert manager.summary_failed is False, manager.summary
    assert_negative_side_mirrored(outputs()["cylinder_unwrapped"].data)


def assert_negative_side_mirrored(mesh):
    layer = mesh.uv_layers.active
    uvs = {}
    for polygon in mesh.polygons:
        side = polygon.center.x < 0
        for loop_index in polygon.loop_indices:
            co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            key = (round(abs(co.x), 4), round(co.y, 4), round(co.z, 4))
            uv = layer.data[loop_index].uv
            uvs.setdefault(side, {}).setdefault(key, []).append((uv.x, uv.y))
    negative, positive = uvs[True], uvs[False]
    assert negative.keys() == positive.keys()
    right = max(u for values in positive.values() for u, _ in values)
    left = min(u for values in negative.values() for u, _ in values)
    for key, values in negative.items():
        mirrored = {(round(left + right - u, 4), round(v, 4)) for u, v in values}
        assert mirrored == {(round(u, 4), round(v, 4)) for u, v in positive[key]}


# mirrors across local X only, and after a 45 degree turn about Z on no world axis
def turned_lopsided_cylinder(load_obj, scale):
    (cylinder,) = load_obj("cylinder")
    for vertex in cylinder.data.vertices:
        if vertex.co.z > 0:
            vertex.co.x *= 0.8
            vertex.co.y *= 0.8
        if vertex.co.y > 0:
            vertex.co.y *= 1.5
    cylinder.rotation_euler = (0, 0, math.radians(45))
    cylinder.scale = scale
    bpy.context.view_layer.update()
    return cylinder


# the output vertices are the source's moved by the object's rotation and scale
def move_to_source_positions(mesh, source, matrix):
    mesh.transform(matrix.to_3x3().inverted().to_4x4())
    tree = mathutils.kdtree.KDTree(len(source.vertices))
    for vertex in source.vertices:
        tree.insert(vertex.co, vertex.index)
    tree.balance()
    for vertex in mesh.vertices:
        position, _, distance = tree.find(vertex.co)
        assert distance < 1e-4
        vertex.co = position


@needs_engine
@pytest.mark.parametrize("scale", [(1, 1, 1), (1.5, 1, 0.7)])
def test_a_turned_mesh_mirrors_across_its_own_plane(load_obj, unwrap, outputs, scale):
    cylinder = turned_lopsided_cylinder(load_obj, scale)
    props = bpy.context.scene.uvgami
    props.pack_after_unwrap = False
    engine = addon.src.engines.get_engine("OPTCUTS")
    plan = addon.src.job.unwrap_plan(engine, props, cylinder.data, painted=False)
    assert plan.symmetry[:2] == ("BISECT", "X")
    unwrap()

    assert manager.error_messages == []
    assert manager.summary_failed is False, manager.summary
    mesh = outputs()[f"{cylinder.name}_unwrapped"].data
    move_to_source_positions(mesh, cylinder.data, cylinder.matrix_world)
    assert_welded_across_x(mesh)
    assert_negative_side_mirrored(mesh)


@needs_engine
@needs_xatlas
def test_xatlas_mirrors_a_turned_and_stretched_mesh(load_obj, unwrap, outputs):
    cylinder = turned_lopsided_cylinder(load_obj, (1.5, 1, 0.7))
    props = bpy.context.scene.uvgami
    props.pack_after_unwrap = False
    engine = addon.src.engines.get_engine("XATLAS")
    plan = addon.src.job.unwrap_plan(engine, props, cylinder.data, painted=False)
    assert plan.symmetry[:2] == ("MIRROR_UVS", "X")
    unwrap("XATLAS")

    assert manager.error_messages == []
    assert manager.summary_failed is False, manager.summary
    mesh = outputs()[f"{cylinder.name}_unwrapped"].data
    move_to_source_positions(mesh, cylinder.data, cylinder.matrix_world)
    assert_negative_side_mirrored(mesh)


def uv_islands(mesh):
    faces = addon.src.utils.mesh.face_vertices(mesh)
    uvs = addon.src.utils.mesh.face_uvs(mesh)
    edges = addon.src.seams.face_edges(faces)
    return addon.src.seams.uv_island_groups(faces, uvs, edges)


def mirror_vertices_across_x(mesh):
    def key(x, y, z):
        return (round(x, 4), round(y, 4), round(z, 4))

    index_of = {key(*v.co): v.index for v in mesh.vertices}
    return {v.index: index_of[key(-v.co.x, v.co.y, v.co.z)] for v in mesh.vertices}


# the island's mirror axis runs through the glued uvs and the midpoints of
# mirrored vertex pairs, and every uv at a plane vertex reflects onto another
# uv at that vertex. the hard surface passes re-solve some islands afterwards,
# which rotates them and keeps the mirror only to about a percent
def assert_island_mirrors_across_its_axis(by_vertex, plane, mirror_of):
    glued = [v for v in plane if len(by_vertex.get(v, ())) == 1]
    points = [next(iter(by_vertex[v])) for v in glued]
    for v, uvs in by_vertex.items():
        partner = by_vertex.get(mirror_of[v])
        if v in plane or partner is None or len(uvs) != 1 or len(partner) != 1:
            continue
        (a,), (b,) = uvs, partner
        points.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
    points = numpy.array(points)
    origin = points.mean(axis=0)
    direction = numpy.linalg.svd(points - origin)[2][0]
    offsets = points - origin
    across = offsets - numpy.outer(offsets @ direction, direction)
    assert numpy.abs(across).max() < 1e-2

    def reflect(uv):
        offset = numpy.array(uv) - origin
        along = (offset @ direction) * direction
        return origin + 2 * along - offset

    for v in plane & by_vertex.keys():
        for uv in by_vertex[v]:
            mirrored = reflect(uv)
            assert any(
                numpy.abs(mirrored - numpy.array(other)).max() < 1e-2
                for other in by_vertex[v]
            )


def assert_welded_across_x(mesh):
    xs = [v.co.x for v in mesh.vertices]
    assert abs(min(xs) + max(xs)) < 1e-4
    plane = set(plane_vertices(mesh, 0))
    mirror_of = mirror_vertices_across_x(mesh)
    welded_islands = 0
    for island in uv_islands(mesh):
        by_vertex = uvs_by_vertex(mesh, island)
        glued = any(len(by_vertex.get(v, ())) == 1 for v in plane)
        # a plane vertex on a one sided island has one uv too, nothing meets it there
        two_sided = any(mirror_of[v] in by_vertex for v in by_vertex if v not in plane)
        if not glued or not two_sided:
            continue
        welded_islands += 1
        assert_island_mirrors_across_its_axis(by_vertex, plane, mirror_of)
    assert welded_islands


# share of seam length with a seam within one edge length of its mirror image across x
def mirrored_seam_share(mesh):
    faces = addon.src.utils.mesh.face_vertices(mesh)
    uvs = addon.src.utils.mesh.face_uvs(mesh)
    seams = list(
        {
            addon.src.seams.pair(a, b)
            for a, b in addon.src.seams.proxy_transfer.uv_tears(faces, uvs)
        }
    )
    positions = numpy.array([v.co[:] for v in mesh.vertices])
    center = (positions[:, 0].min() + positions[:, 0].max()) / 2
    ends = positions[numpy.array(seams)]
    middles = ends.mean(axis=1)
    lengths = numpy.linalg.norm(ends[:, 0] - ends[:, 1], axis=1)
    mirrored = middles.copy()
    mirrored[:, 0] = 2 * center - mirrored[:, 0]
    nearest = numpy.linalg.norm(mirrored[:, None, :] - middles[None, :, :], axis=2)
    reach = max(
        (mesh.vertices[e.vertices[0]].co - mesh.vertices[e.vertices[1]].co).length
        for e in mesh.edges
    )
    hit = nearest.min(axis=1) <= reach
    return float(lengths[hit].sum() / lengths.sum())


# the plane loop is the edges between faces either side of the X plane
def plane_loop_and_seams(mesh):
    faces = addon.src.utils.mesh.face_vertices(mesh)
    torn = {
        addon.src.seams.pair(a, b)
        for a, b in addon.src.seams.proxy_transfer.uv_tears(
            faces, addon.src.utils.mesh.face_uvs(mesh)
        )
    }
    positions = numpy.array([v.co[:] for v in mesh.vertices])
    center = (positions[:, 0].min() + positions[:, 0].max()) / 2
    positive = [positions[face, 0].mean() >= center for face in faces]
    edges = addon.src.seams.face_edges(faces)
    plane_loop = {
        key
        for key, owners in edges.items()
        if len(owners) == 2 and positive[owners[0]] != positive[owners[1]]
    }
    return plane_loop, torn


@needs_engine
def test_hard_surface_mirrors_the_placed_seams_on_a_whole_mesh(
    load_obj, unwrap, outputs
):
    (cube,) = load_obj("cube-bevel2")
    props = bpy.context.scene.uvgami
    props.optcuts.mode = "HARD"
    props.pack_after_unwrap = False
    unwrap()

    assert manager.error_messages == []
    assert manager.summary_failed is False, manager.summary
    mesh = outputs()[f"{cube.name}_unwrapped"].data
    # the mesh ships whole, so every input vertex is in the output
    assert len(mesh.vertices) == len(cube.data.vertices)
    assert mirrored_seam_share(mesh) > 0.9


@needs_engine
def test_organic_mirrors_the_half_run_cuts_on_a_whole_mesh(load_obj, unwrap, outputs):
    (cylinder,) = load_obj("cylinder")
    source = cylinder.data
    _, diagonal = addon.src.job.symmetry_center(source)
    # drifted off exact, so the mesh ships whole with the half run's cuts mirrored
    source.vertices[0].co += mathutils.Vector((1, 1, 1)) * 0.001 * diagonal
    props = bpy.context.scene.uvgami
    props.optcuts.mode = "ORGANIC"
    props.pack_after_unwrap = False
    unwrap()

    assert manager.error_messages == []
    assert manager.summary_failed is False, manager.summary
    mesh = outputs()[f"{cylinder.name}_unwrapped"].data
    assert len(mesh.vertices) == len(source.vertices)
    assert mirrored_seam_share(mesh) > 0.9
    # the plain engine unwrap mirrors well on a cylinder too, but leaves the plane uncut
    plane_loop, torn = plane_loop_and_seams(mesh)
    assert plane_loop <= torn


@needs_engine
def test_a_proxied_mesh_mirrors_the_half_run_cuts(load_obj, unwrap, outputs):
    (cylinder,) = load_obj("cylinder")
    bpy.context.view_layer.objects.active = cylinder
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.subdivide(number_cuts=3)
    bpy.ops.object.mode_set(mode="OBJECT")
    _, diagonal = addon.src.job.symmetry_center(cylinder.data)
    # mostly symmetrical, whose mirror map the decimate would leave stale
    cylinder.data.vertices[0].co += mathutils.Vector((1, 1, 1)) * 0.001 * diagonal
    props = bpy.context.scene.uvgami
    props.optcuts.mode = "ORGANIC"
    props.use_proxy = True
    props.proxy_faces = 1000
    unwrap()

    results = [result for _, result in manager.results]
    assert results
    assert set(results) == {addon.src.job.Result.FINISHED}
    mesh = outputs()[f"{cylinder.name}_unwrapped"].data
    assert mirrored_seam_share(mesh) > 0.9
    plane_loop, torn = plane_loop_and_seams(mesh)
    # the transfer's straightened seams cut across the cap corners
    assert len(plane_loop & torn) > len(plane_loop) / 2


# def test_hard_surface_keeps_its_seams_and_welds_the_plane(load_obj, unwrap, outputs):
#     load_obj("cube-bevel2")
#     props = bpy.context.scene.uvgami
#     props.optcuts.mode = "HARD"
#     # Auto skips a mesh whose faces cross the plane
#     props.symmetry = "AXES"
#     props.sym_axes = {"X"}
#     props.pack_after_unwrap = False
#     unwrap()

#     assert manager.error_messages == []
#     assert manager.summary_failed is False, manager.summary
#     output = outputs()["Cube_unwrapped"]
#     assert_welded_across_x(output.data)
#     # a glued half cube is one island, the bevel seams split it
#     assert island_count(output) > 1


# def test_overlap_folds_the_mirrored_half_onto_the_same_uvs(load_obj, unwrap, outputs):
#     load_obj("cylinder")
#     props = bpy.context.scene.uvgami
#     props.sym_overlap = True
#     unwrap()

#     assert manager.error_messages == []
#     mesh = outputs()["cylinder_unwrapped"].data
#     layer = mesh.uv_layers.active
#     half = len(mesh.polygons) // 2
#     kept = {
#         (round(layer.data[i].uv.x, 6), round(layer.data[i].uv.y, 6))
#         for face in mesh.polygons[:half]
#         for i in face.loop_indices
#     }
#     mirrored = {
#         (round(layer.data[i].uv.x, 6), round(layer.data[i].uv.y, 6))
#         for face in mesh.polygons[half:]
#         for i in face.loop_indices
#     }
#     assert kept == mirrored
