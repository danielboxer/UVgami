import importlib.util
import sys
from pathlib import Path

import numpy
import pytest

# loaded from file so it doesn't need the bpy-only addon package
PKG = Path(__file__).parents[2] / "src" / "seams"
spec = importlib.util.spec_from_file_location(
    "seams", PKG / "__init__.py", submodule_search_locations=[str(PKG)]
)
sys.modules["seams"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sys.modules["seams"])
from seams import Cancelled, face_edges  # noqa: E402
from seams.proxy_transfer import (  # noqa: E402
    cut_edges,
    finish_proxy,
    moved_weights,
    repair_islands,
    snap_cuts,
)

IDENTITY = numpy.eye(4)


def quad_grid(size, spacing=1.0):
    """size by size quads on the z=0 plane, verts row major."""
    side = size + 1
    verts = [(x * spacing, y * spacing, 0.0) for y in range(side) for x in range(side)]
    faces = [
        [y * side + x, y * side + x + 1, (y + 1) * side + x + 1, (y + 1) * side + x]
        for y in range(size)
        for x in range(size)
    ]
    return verts, faces


def grid_uvs(verts, faces):
    return [[(verts[v][0], verts[v][1]) for v in face] for face in faces]


def grid_edges(faces):
    return numpy.array(sorted(face_edges(faces)), dtype=numpy.int64)


def up_normals(verts):
    return [(0.0, 0.0, 1.0)] * len(verts)


def torn_proxy(spacing=2.0):
    """A 2x2 quad proxy whose first face is torn away from its neighbours."""
    verts, faces = quad_grid(2, spacing)
    uvs = grid_uvs(verts, faces)
    uvs[0] = [(u + 10.0, v) for u, v in uvs[0]]
    return verts, faces, uvs


class StubEngine:
    def __init__(self):
        self.verts = None
        self.faces = None
        self.seams = None

    def flatten(self, verts, faces, seams, cancelled=None, progress=None):
        self.verts, self.faces, self.seams = verts, faces, set(seams)
        return [[(0.25, 0.75)] * len(face) for face in faces]


def test_cut_edges_finds_only_the_torn_interior_edge():
    verts, faces = quad_grid(2)
    uvs = grid_uvs(verts, faces)
    assert cut_edges(faces, uvs) == set()

    uvs[0] = [(u + 10.0, v) for u, v in uvs[0]]
    # face 0 is (0, 1, 4, 3): (1, 4) and (3, 4) are its interior edges
    assert cut_edges(faces, uvs) == {(1, 4), (3, 4)}


def test_cut_edges_skips_boundary_edges():
    verts, faces = quad_grid(2)
    uvs = grid_uvs(verts, faces)
    uvs[0] = [(u + 10.0, v) for u, v in uvs[0]]
    boundary = {key for key, owners in face_edges(faces).items() if len(owners) == 1}
    assert boundary
    assert not (cut_edges(faces, uvs) & boundary)


def test_moved_weights_moves_onto_output_indices():
    assert moved_weights(None, [0, 1]) is None
    assert moved_weights({5: 1.0}, [0, 1]) is None
    assert moved_weights({0: 0.5, 2: 1.0}, [2, 7, 0]) == {0: 1.0, 2: 0.5}


def test_repair_islands_opens_an_annulus():
    verts, faces = quad_grid(3)
    del faces[4]  # the middle quad, leaving an inner boundary loop
    cuts = repair_islands(verts, faces, set())
    assert cuts
    edges = set(face_edges(faces))
    assert cuts <= edges


def test_repair_islands_leaves_a_disk_alone():
    verts, faces = quad_grid(3)
    assert repair_islands(verts, faces, set()) == set()
    torn = {(1, 5)}
    assert repair_islands(verts, faces, torn) == torn


def test_snap_cuts_follows_real_edges():
    verts, faces = quad_grid(3)
    edges = grid_edges(faces)
    mapped = [0, 3]
    assert snap_cuts(verts, edges, mapped, {(0, 1)}) == {(0, 1), (1, 2), (2, 3)}


def plane_inputs():
    """A 4x4 dense grid and its torn 2x2 proxy, with the exact vertex map."""
    dense_verts, dense_faces = quad_grid(4)
    proxy_verts, proxy_faces, proxy_uvs = torn_proxy()
    dense = {
        "positions": dense_verts,
        "faces": dense_faces,
        "edges": grid_edges(dense_faces),
        "normals": up_normals(dense_verts),
        "matrix": IDENTITY,
    }
    proxy = {
        "positions": proxy_verts,
        "faces": proxy_faces,
        "corner_uvs": proxy_uvs,
        "normals": up_normals(proxy_verts),
        "matrix": IDENTITY,
    }
    # proxy verts sit on every second dense vert of the 5 by 5 grid
    mapped = [row * 10 + column * 2 for row in range(3) for column in range(3)]
    return dense, proxy, mapped


def test_finish_proxy_cuts_the_dense_mesh_and_returns_its_uvs():
    dense, proxy, mapped = plane_inputs()
    engine = StubEngine()
    reported = []
    seams, uvs = finish_proxy(dense, proxy, None, engine, mapped, reported.append)

    # the proxy tear runs from dense vert 2 to 12 and from 10 to 12
    assert seams == {(2, 7), (7, 12), (10, 11), (11, 12)}
    assert engine.seams == seams
    assert engine.faces == dense["faces"]
    assert len(engine.verts) == len(dense["positions"])
    assert len(uvs) == len(dense["faces"])
    assert uvs[0] == [(0.25, 0.75)] * 4
    assert reported == sorted(reported)
    assert reported[-1] == 1.0


def test_finish_proxy_stops_on_cancel():
    dense, proxy, mapped = plane_inputs()
    engine = StubEngine()
    with pytest.raises(Cancelled):
        finish_proxy(dense, proxy, None, engine, mapped, cancelled=lambda: True)
    # cancelled before the flatten was asked for anything
    assert engine.faces is None
