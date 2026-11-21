from collections import namedtuple

import numpy

from .engine_run import (
    FLOAT,
    INTEGER,
    packed_reader,
    poll_run,
    start_packed,
)
from .mesh import (
    face_starts,
    lowest_index_part_labels,
    next_corners_in_face,
    undirected_edge_keys,
)
from .symmetry import face_partners, mutual_mirror_partners

PaintedFlatten = namedtuple("PaintedFlatten", "overlapping_islands corner_uvs")


def outline_edges(corners, totals, painted):
    following = corners[next_corners_in_face(totals)]
    low = numpy.minimum(corners, following)
    high = numpy.maximum(corners, following)
    keys = low * (int(corners.max()) + 1) + high
    corner_painted = numpy.repeat(painted, totals)
    painted_keys = numpy.unique(keys[corner_painted])
    unpainted_keys = numpy.unique(keys[~corner_painted])
    on_outline = numpy.isin(keys, numpy.intersect1d(painted_keys, unpainted_keys))
    pairs = numpy.stack([low[on_outline], high[on_outline]], axis=1)
    return numpy.unique(pairs, axis=0)


def borders_outside_paint(corners, totals, painted, border_edges):
    following = corners[next_corners_in_face(totals)]
    size = int(corners.max()) + 1
    keys = numpy.minimum(corners, following) * size + numpy.maximum(corners, following)
    unpainted_keys = set(keys[~numpy.repeat(painted, totals)].tolist())
    return [
        edge for edge in border_edges if min(edge) * size + max(edge) in unpainted_keys
    ]


def unpaint_closed_parts(corners, totals, painted):
    corner_painted = numpy.repeat(painted, totals)
    tails = corners[corner_painted]
    heads = corners[next_corners_in_face(totals)][corner_painted]
    keys = undirected_edge_keys(tails, heads)
    unique, counts = numpy.unique(keys, return_counts=True)
    on_outline = numpy.isin(keys, unique[counts == 1])
    labels = lowest_index_part_labels(
        int(corners.max()) + 1, numpy.stack([tails, heads], axis=1)
    )
    face_labels = labels[corners[face_starts(totals)]]
    return painted & numpy.isin(face_labels, labels[tails[on_outline]])


def _mesh_input(positions, corners, totals):
    return [
        numpy.array([len(positions), len(totals)], dtype=INTEGER),
        numpy.asarray(positions, dtype=FLOAT).reshape(-1),
        numpy.asarray(totals, dtype=INTEGER),
        numpy.asarray(corners, dtype=INTEGER),
    ]


def painted_outline_job(
    command, workdir, positions, corners, totals, painted, plane, cancelled=None
):
    if plane is not None:
        found = mutual_mirror_partners(command, workdir, positions, *plane)
        partners = numpy.full(len(positions), -1, dtype=numpy.int64)
        partners[list(found)] = list(found.values())
        mirrors = face_partners(corners, totals, partners)
        has_mirror = mirrors >= 0
        painted = painted.copy()
        painted[has_mirror] |= painted[mirrors[has_mirror]]
    arrays = [*_mesh_input(positions, corners, totals), painted.astype(INTEGER)]
    run = start_packed(
        command,
        workdir,
        "paint-outline",
        arrays,
        lambda data: numpy.frombuffer(data, INTEGER) == 1,
    )
    return poll_run(run, cancelled)


def _flatten(command, workdir, positions, corners, totals, cancelled):
    def read(data):
        take = packed_reader(data)
        in_overlapping_island = take(INTEGER, len(totals)) == 1
        return in_overlapping_island, take(FLOAT, 2 * len(corners)).reshape(-1, 2)

    arrays = _mesh_input(positions, corners, totals)
    return poll_run(
        start_packed(command, workdir, "paint-flatten", arrays, read), cancelled
    )


def painted_flatten_job(
    command,
    workdir,
    positions,
    corners,
    totals,
    painted,
    cancelled=None,
):
    painted_corners = corners[numpy.repeat(painted, totals)]
    used, local_corners = numpy.unique(painted_corners, return_inverse=True)
    painted_totals = totals[painted]
    in_overlapping_island, corner_uvs = _flatten(
        command,
        workdir,
        numpy.asarray(positions, dtype=FLOAT)[used],
        local_corners,
        painted_totals,
        cancelled,
    )
    following = local_corners[next_corners_in_face(painted_totals)]
    labels = lowest_index_part_labels(
        len(used), numpy.stack([local_corners, following], axis=1)
    )
    face_labels = labels[local_corners[face_starts(painted_totals)]]
    overlapping_islands = len(numpy.unique(face_labels[in_overlapping_island]))
    return PaintedFlatten(overlapping_islands, corner_uvs)
