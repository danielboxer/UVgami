import math
import pathlib

import bpy
import pytest
from blender_fixtures import (
    addon,
    island_count,
    manager,
    needs_engine,
    needs_partuv,
    needs_xatlas,
)

pytestmark = [needs_engine, pytest.mark.smoke]

OTHER_ENGINES = [
    pytest.param("XATLAS", marks=needs_xatlas),
    pytest.param("PARTUV", marks=needs_partuv),
]


def test_unwrap_runs_to_completion(load_obj, unwrap, outputs):
    load_obj("cylinder")
    unwrap()

    assert manager.error_messages == []
    assert manager.summary_failed is False, manager.summary
    assert list(outputs()) == ["cylinder_unwrapped"]
    layer = outputs()["cylinder_unwrapped"].data.uv_layers.active
    assert layer is not None
    assert any(any(datum.uv) for datum in layer.data)

    # a many-part model runs out of file descriptors partway through
    assert manager.results
    for piece, _ in manager.results:
        if piece.process is None:
            continue
        assert piece.process.returncode is not None
        assert piece.process.stdin.closed


# without the lock the torus handle cut runs along diagonals
def test_keep_quads_rebuilds_every_face(load_obj, unwrap, outputs):
    obj = load_obj("torus")[0]
    sizes = sorted(len(polygon.vertices) for polygon in obj.data.polygons)
    assert 4 in sizes
    unwrap()

    assert manager.error_messages == []
    assert manager.summary_failed is False, manager.summary
    assert len(outputs()) == 1
    output = next(iter(outputs().values()))
    assert sorted(len(polygon.vertices) for polygon in output.data.polygons) == sizes
    assert output.data.uv_layers.active is not None


def test_flipped_face_is_rewound_instead_of_refused(load_obj, unwrap, outputs):
    load_obj("flipped-face")
    unwrap()

    assert manager.error_messages == []
    assert manager.summary_failed is False, manager.summary
    assert len(outputs()) == 1
    layer = next(iter(outputs().values())).data.uv_layers.active
    assert layer is not None
    assert any(any(datum.uv) for datum in layer.data)


# shade_auto_smooth cancels on headless 4.3
def smooth_by_angle_node_group():
    name = addon.src.utils.mesh.AUTO_SMOOTH_MODIFIER_NAME
    assets = pathlib.Path(bpy.utils.system_resource("DATAFILES")) / "assets"
    for path in sorted(assets.rglob("*.blend")):
        with bpy.data.libraries.load(str(path)) as (source, target):
            found = name in source.node_groups
            if found:
                target.node_groups = [name]
        if found:
            return bpy.data.node_groups[name]
    raise LookupError(f"no bundled asset holds {name}")


def test_shading_modifiers_carry_over(load_obj, unwrap, outputs):
    obj = load_obj("cylinder")[0]
    weighted = obj.modifiers.new("WeightedNormal", "WEIGHTED_NORMAL")
    weighted.keep_sharp = True
    angle = 0.5
    smooth = obj.modifiers.new("Smooth by Angle", "NODES")
    smooth.node_group = smooth_by_angle_node_group()
    addon.src.utils.mesh._set_node_input_values(smooth, {"Input_1": angle})
    unwrap()

    assert manager.error_messages == []
    output = outputs()["cylinder_unwrapped"]
    assert [m.type for m in output.modifiers] == ["WEIGHTED_NORMAL", "NODES"]
    assert output.modifiers[0].keep_sharp is True
    assert output.modifiers[1].node_group is not None
    records = addon.src.utils.mesh.get_shading_modifiers(output)
    assert records[1][2][1]["Input_1"] == pytest.approx(angle)


@pytest.mark.parametrize("engine", OTHER_ENGINES)
def test_other_engines_unwrap_the_cylinder(load_obj, unwrap, outputs, engine):
    load_obj("cylinder")
    if engine == "PARTUV":
        bpy.context.scene.uvgami.partuv.segmentation = "GEOMETRIC"

    unwrap(engine)

    assert manager.summary == ["UV unwrap complete"]
    assert manager.error_messages == []
    layer = outputs()["cylinder_unwrapped"].data.uv_layers.active
    assert layer is not None
    assert any(any(datum.uv) for datum in layer.data)


def capped_quad_cylinder(around=24, along=6):
    verts = [
        (math.cos(2 * math.pi * i / around), math.sin(2 * math.pi * i / around), r / 2)
        for r in range(along + 1)
        for i in range(around)
    ]
    faces = []
    for r in range(along):
        for i in range(around):
            j = (i + 1) % around
            faces.append(
                (
                    r * around + i,
                    r * around + j,
                    (r + 1) * around + j,
                    (r + 1) * around + i,
                )
            )
    faces.append(tuple(reversed(range(around))))
    faces.append(tuple(along * around + i for i in range(around)))
    return verts, faces


# hard surface locks the triangulation diagonals, so no seam may run along one
def test_hard_surface_keeps_seams_off_the_diagonals(make_mesh, unwrap, outputs):
    verts, faces = capped_quad_cylinder()
    obj = make_mesh("cylinder", verts, faces, [[(0.0, 0.0)] * len(f) for f in faces])
    mesh = obj.data
    mesh_edges = {
        frozenset(
            tuple(round(c, 5) for c in mesh.vertices[v].co) for v in edge.vertices
        )
        for edge in mesh.edges
    }
    bpy.context.scene.uvgami.optcuts.mode = "HARD"
    bpy.context.scene.uvgami.keep_quads = False
    # a mirrored seam set would split the tube in two
    bpy.context.scene.uvgami.symmetry = False
    unwrap()

    assert manager.error_messages == []
    assert manager.summary_failed is False, manager.summary
    output = next(iter(outputs().values())).data
    layer = output.uv_layers.active.data
    corner_uv = {}
    for polygon in output.polygons:
        for loop_index in polygon.loop_indices:
            vertex = output.loops[loop_index].vertex_index
            corner_uv.setdefault((polygon.index, vertex), tuple(layer[loop_index].uv))
    owners_of = {}
    for polygon in output.polygons:
        for key in polygon.edge_keys:
            owners_of.setdefault(key, []).append(polygon)
    diagonal_seams = 0
    for edge in output.edges:
        key = frozenset(
            tuple(round(c, 5) for c in output.vertices[v].co) for v in edge.vertices
        )
        if key in mesh_edges:
            continue
        owners = owners_of.get(edge.key, [])
        if len(owners) != 2:
            continue
        a, b = owners
        for vertex in edge.vertices:
            if corner_uv[(a.index, vertex)] != corner_uv[(b.index, vertex)]:
                diagonal_seams += 1
                break
    assert diagonal_seams == 0
    # the two caps and the wall opened once
    assert island_count(next(iter(outputs().values()))) == 3
