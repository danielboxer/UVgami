"""The uv editor operators: each on a face selection in edit mode, each with
the proxy option, and the refusals that leave edit mode intact."""

import bpy
import mathutils
import pytest
from blender_fixtures import manager, needs_engine

pytestmark = [needs_engine, pytest.mark.smoke]

GRID = 16
# the prop minimum, under the grid's 512 triangles
PROXY_FACES = 100
# far enough off that a relax moves it back
DISTORTION = (0.03, 0.02)


def add_grid():
    """A subdivided plane with blender's own uv map, one island."""
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=GRID, y_subdivisions=GRID)
    return bpy.context.active_object


def center_vertex(obj):
    return min(obj.data.vertices, key=lambda v: v.co.length).index


def faces_around(obj, vertex):
    return {f.index for f in obj.data.polygons if vertex in f.vertices}


def block_edge_vertex(obj, block):
    """A vertex on the border of the block: pinned by the block alone, free
    once the area grows by a ring."""
    counts = {}
    for index in block:
        for vertex in obj.data.polygons[index].vertices:
            counts[vertex] = counts.get(vertex, 0) + 1
    return next(vertex for vertex, count in counts.items() if count == 2)


def vertex_uvs(obj, vertex):
    """The uv(s) on a vertex's corners. Read through the loops, which a face
    split elsewhere leaves alone."""
    bpy.ops.object.mode_set(mode="OBJECT")
    layer = obj.data.uv_layers.active.data
    uvs = {
        tuple(round(c, 5) for c in layer[loop.index].uv)
        for loop in obj.data.loops
        if loop.vertex_index == vertex
    }
    bpy.ops.object.mode_set(mode="EDIT")
    return uvs


def distort(obj, vertex):
    layer = obj.data.uv_layers.active.data
    for loop in obj.data.loops:
        if loop.vertex_index == vertex:
            layer[loop.index].uv += mathutils.Vector(DISTORTION)


def select_faces(obj, indices):
    """Select faces in object mode, then enter edit mode as the operators'
    poll wants. Sync selection on, so the mesh flags are what count."""
    bpy.context.scene.tool_settings.use_uv_select_sync = True
    mesh = obj.data
    # a primitive comes with every vertex selected, and edit mode flushes
    # that up to the faces
    for element in (*mesh.vertices, *mesh.edges, *mesh.polygons):
        element.select = False
    for index in indices:
        mesh.polygons[index].select = True
    bpy.ops.object.mode_set(mode="EDIT")


OPERATORS = ["unwrap_island", "relax_island", "unwrap_area", "relax_area"]


@pytest.mark.parametrize("operator", OPERATORS)
@pytest.mark.parametrize("proxy", [False, True], ids=["direct", "proxy"])
def test_operator_moves_the_distorted_vertex_and_restores_edit_mode(
    session, operator, proxy
):
    grid = add_grid()
    props = bpy.context.scene.uvgami
    props.use_proxy = proxy
    props.proxy_faces = PROXY_FACES
    center = center_vertex(grid)
    distort(grid, center)
    select_faces(grid, faces_around(grid, center))
    before = vertex_uvs(grid, center)

    session(getattr(bpy.ops.uvgami, operator))

    assert manager.summary[0] == "UV unwrap complete!", manager.summary
    assert manager.error_messages == []
    assert grid.mode == "EDIT"
    assert vertex_uvs(grid, center) != before


def test_combine_islands_joins_two_islands_into_one(session, make_mesh, face_uvs):
    obj = make_mesh(
        "two",
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0)],
        [(0, 1, 2, 3), (1, 4, 5, 2)],
        [[(0, 0), (1, 0), (1, 1), (0, 1)], [(2, 0), (3, 0), (3, 1), (2, 1)]],
    )
    select_faces(obj, {0, 1})

    session(bpy.ops.uvgami.combine_islands)

    assert manager.summary[0] == "UV unwrap complete!", manager.summary
    bpy.ops.object.mode_set(mode="OBJECT")
    first, second = face_uvs(obj)
    # the shared 3d edge 1-2 now has one uv per vertex
    assert first[1] == second[0]
    assert first[2] == second[3]


@pytest.mark.parametrize("rings, moves", [(0, False), (1, True)])
def test_expand_rings_free_the_border_of_the_selection(session, rings, moves):
    grid = add_grid()
    bpy.context.scene.uvgami.area_expand = rings
    block = faces_around(grid, center_vertex(grid))
    border = block_edge_vertex(grid, block)
    distort(grid, border)
    select_faces(grid, block)
    before = vertex_uvs(grid, border)

    session(bpy.ops.uvgami.relax_area)

    assert manager.summary[0] == "UV unwrap complete!", manager.summary
    assert (vertex_uvs(grid, border) != before) is moves


@pytest.mark.parametrize(
    "operator, message",
    [
        ("unwrap_island", "Select the faces of the islands to fix"),
        ("relax_area", "Select the faces of the area to fix"),
    ],
)
def test_no_selection_is_refused(operator, message):
    grid = add_grid()
    select_faces(grid, set())

    with pytest.raises(RuntimeError, match=message):
        getattr(bpy.ops.uvgami, operator)()

    assert grid.mode == "EDIT"
    assert not manager.is_active


def test_combine_needs_two_islands():
    grid = add_grid()
    select_faces(grid, faces_around(grid, center_vertex(grid)))

    with pytest.raises(RuntimeError, match="Select faces on at least two islands"):
        bpy.ops.uvgami.combine_islands()

    assert grid.mode == "EDIT"
    assert not manager.is_active


def test_no_uv_map_is_refused():
    grid = add_grid()
    grid.data.uv_layers.remove(grid.data.uv_layers[0])
    select_faces(grid, faces_around(grid, center_vertex(grid)))

    with pytest.raises(RuntimeError, match="Mesh has no uv map"):
        bpy.ops.uvgami.relax_island()

    assert grid.mode == "EDIT"
    assert not manager.is_active
