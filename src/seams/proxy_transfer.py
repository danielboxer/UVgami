"""Finish a proxy unwrap on plain data.

The cuts the engine drew on the proxy are read off its uv map, repaired into
disks, snapped onto the dense mesh's own edges, and the dense mesh is
flattened once. proxy.py adapts a Blender mesh onto these calls."""

import collections
import time

import numpy

from .cuts import connect_loops, snap_paths
from .islands import uv_topology
from .mesh import face_edges, island_groups, pair
from .preseed import FlattenError, check_manifold

# the flatten is nearly all the work, so it gets the rest of the bar
CUT_PROGRESS = 0.01
REPAIR_PROGRESS = 0.06
SNAP_PROGRESS = 0.1
FLATTEN_POLL_SECONDS = 0.05


def cut_edges(faces, corner_uvs):
    """Proxy edges the uv map is torn across, as vertex index pairs.

    An edge already on the mesh boundary is a cut the original has of its own,
    so only interior tears count."""

    def corner(f, v):
        return corner_uvs[f][faces[f].index(v)]

    torn = set()
    for (u, v), owners in face_edges(faces).items():
        if len(owners) != 2:
            continue
        f, g = owners
        if corner(f, u) != corner(g, u) or corner(f, v) != corner(g, v):
            torn.add((u, v))
    return torn


def moved_weights(weights, mapped):
    """The painted restrictions on the proxy's own vertices, read off the
    original through the same map the cuts are snapped back with."""
    if not weights:
        return None
    moved = {i: weights[v] for i, v in enumerate(mapped) if v in weights}
    return moved or None


def repair_islands(verts, faces, torn_seams, weights=None):
    """Every island opened into a disk, as the full cut set.

    The engine's cuts are slits that never reach the mesh's own holes, so an
    island can keep extra boundary loops. The flatten only maps a disk, and
    the extra loops collapse into one crushed circle."""
    seams = set(torn_seams)
    edges = face_edges(faces)
    for group in island_groups(faces, seams, edges):
        loops = uv_topology(group, faces, edges, seams)[1]
        if len(loops) < 2:
            continue
        adjacent = collections.defaultdict(set)
        for f in group:
            face = faces[f]
            n = len(face)
            for i in range(n):
                key = pair(face[i], face[(i + 1) % n])
                if len(edges[key]) == 2 and key not in seams:
                    adjacent[key[0]].add(key[1])
                    adjacent[key[1]].add(key[0])
        seams |= connect_loops(verts, adjacent, loops, weights)
    return seams


def snap_cuts(verts, edges, mapped, cuts):
    """Another mesh's cut network redrawn along this mesh's own edges."""
    adjacent = collections.defaultdict(set)
    for a, b in numpy.asarray(edges).reshape(-1, 2).tolist():
        adjacent[a].add(b)
        adjacent[b].add(a)
    return snap_paths(verts, adjacent, mapped, cuts)


def finish_proxy(dense, proxy, weights, engine, mapped, progress=None, cancelled=None):
    """Cut the dense mesh along the proxy's cuts and flatten it.

    Returns (seams, uvs) for the dense mesh. mapped is each proxy vertex's
    dense vertex, weights are painted restrictions on dense indices, and
    cancelled is polled while the flatten runs."""

    def report(fraction):
        if progress is not None:
            progress(fraction)

    def is_cancelled():
        return cancelled is not None and cancelled()

    dense_verts = numpy.asarray(dense["positions"], dtype=numpy.float64).tolist()
    dense_faces = dense["faces"]

    cuts = cut_edges(proxy["faces"], proxy["corner_uvs"])
    report(CUT_PROGRESS)
    proxy_verts = numpy.asarray(proxy["positions"], dtype=numpy.float64).tolist()
    cuts = repair_islands(
        proxy_verts, proxy["faces"], cuts, moved_weights(weights, mapped)
    )
    report(REPAIR_PROGRESS)
    seams = snap_cuts(dense_verts, dense["edges"], mapped, cuts)
    report(SNAP_PROGRESS)

    check_manifold(dense_faces)
    if is_cancelled():
        raise FlattenError("cancelled")
    run = engine.start(dense_verts, dense_faces, seams)
    while run.poll() is None:
        if is_cancelled():
            run.stop()
            raise FlattenError("cancelled")
        report(SNAP_PROGRESS + (1 - SNAP_PROGRESS) * run.progress)
        time.sleep(FLATTEN_POLL_SECONDS)
    uvs = run.result()
    report(1.0)
    return seams, uvs
