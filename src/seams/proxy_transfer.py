"""Finish a proxy unwrap on plain data.

The cuts the engine drew on the proxy are read off its uv map, repaired into
disks, snapped onto the dense mesh's own edges, and the dense mesh is
flattened once. proxy.py adapts a Blender mesh onto these calls."""

import collections

import numpy

from .cancel import check_cancelled
from .cuts import connect_loops, part_labels, snap_paths
from .islands import uv_topology
from .mesh import face_edges, island_groups, pair
from .preseed import check_manifold

# the flatten is nearly all the work, so it gets the rest of the bar
CUT_PROGRESS = 0.01
REPAIR_PROGRESS = 0.06
SNAP_PROGRESS = 0.1


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


def edge_adjacency(edges):
    adjacent = collections.defaultdict(set)
    for a, b in numpy.asarray(edges).reshape(-1, 2).tolist():
        adjacent[a].add(b)
        adjacent[b].add(a)
    return adjacent


def snap_cuts(verts, edges, mapped, cuts):
    """Another mesh's cut network redrawn along this mesh's own edges."""
    return snap_paths(verts, edge_adjacency(edges), mapped, cuts)


def vertex_parts(count, adjacent):
    """A loose part label per vertex, -1 for a vertex on no edge."""
    labels = numpy.full(count, -1)
    for v, label in part_labels(adjacent).items():
        labels[v] = label
    return labels


def match_within_parts(dense_parts, proxy_parts, nearest):
    """Each proxy vertex's dense vertex, kept inside the dense loose part its
    own part sits on.

    nearest(dense_indices, proxy_indices) gives each listed proxy vertex its
    nearest listed dense vertex. Where two parts touch that lands on the other
    part, and a cut with an end there breaks the part's cut network, so each
    proxy part is paired with the dense part most of it lands on and the rest
    matched again within it."""
    dense_parts = numpy.asarray(dense_parts)
    proxy_parts = numpy.asarray(proxy_parts)
    mapped = numpy.asarray(
        nearest(numpy.arange(len(dense_parts)), numpy.arange(len(proxy_parts)))
    )
    for part in numpy.unique(proxy_parts):
        members = numpy.flatnonzero(proxy_parts == part)
        landed = dense_parts[mapped[members]]
        labels, counts = numpy.unique(landed, return_counts=True)
        home = labels[counts.argmax()]
        strays = members[landed != home]
        if len(strays):
            mapped[strays] = nearest(numpy.flatnonzero(dense_parts == home), strays)
    return mapped.tolist()


def finish_proxy(dense, proxy, weights, engine, nearest, progress=None, cancelled=None):
    """Cut the dense mesh along the proxy's cuts and flatten it.

    Returns (seams, uvs) for the dense mesh. nearest is match_within_parts'
    matcher, weights are painted restrictions on dense indices, and cancelled
    is polled while the flatten runs."""

    def report(fraction):
        if progress is not None:
            progress(fraction)

    dense_verts = numpy.asarray(dense["positions"], dtype=numpy.float64).tolist()
    dense_faces = dense["faces"]
    dense_adjacent = edge_adjacency(dense["edges"])
    proxy_adjacent = edge_adjacency(list(face_edges(proxy["faces"])))
    mapped = match_within_parts(
        vertex_parts(len(dense_verts), dense_adjacent),
        vertex_parts(len(proxy["positions"]), proxy_adjacent),
        nearest,
    )

    cuts = cut_edges(proxy["faces"], proxy["corner_uvs"])
    report(CUT_PROGRESS)
    proxy_verts = numpy.asarray(proxy["positions"], dtype=numpy.float64).tolist()
    cuts = repair_islands(
        proxy_verts, proxy["faces"], cuts, moved_weights(weights, mapped)
    )
    report(REPAIR_PROGRESS)
    seams = snap_paths(dense_verts, dense_adjacent, mapped, cuts)
    report(SNAP_PROGRESS)

    check_manifold(dense_faces)
    check_cancelled(cancelled)
    uvs = engine.flatten(
        dense_verts,
        dense_faces,
        seams,
        cancelled,
        lambda fraction: report(SNAP_PROGRESS + (1 - SNAP_PROGRESS) * fraction),
    )
    report(1.0)
    return seams, uvs
