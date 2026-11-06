import os
from collections import namedtuple

import numpy

from .engine_run import (
    FLOAT,
    INTEGER,
    new_run_folder,
    packed_reader,
    poll_run,
    start_packed,
    start_run,
)
from .mesh import face_starts, lowest_index_part_labels, next_corners_in_face
from .symmetry import face_partners, mutual_mirror_partners

PaintedFlatten = namedtuple("PaintedFlatten", "overlapping_islands corner_uvs")

RELAX_INPUT_STEM = "island"


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


def _fan_triangles(corners, totals):
    starts = face_starts(totals)
    face_of_triangle = numpy.repeat(numpy.arange(len(totals)), totals - 2)
    step = numpy.arange(len(face_of_triangle)) - numpy.repeat(
        numpy.cumsum(totals - 2) - (totals - 2), totals - 2
    )
    first = starts[face_of_triangle]
    triangles = numpy.stack(
        [corners[first], corners[first + step + 1], corners[first + step + 2]],
        axis=1,
    )
    return triangles, face_of_triangle


def _read_relaxed(path, _):
    uvs = []
    uv_corners = []
    for line in path.read_text().splitlines():
        if line.startswith("vt "):
            uvs.append(line.split()[1:3])
        elif line.startswith("f "):
            uv_corners.append([int(c.split("/")[1]) - 1 for c in line.split()[1:]])
    return numpy.array(uvs, dtype=FLOAT), numpy.array(uv_corners, dtype=INTEGER)


def _relaxed(optcuts, workdir, positions, triangles, uvs, cancelled):
    folder = new_run_folder(workdir)
    obj_path = folder / f"{RELAX_INPUT_STEM}.obj"
    with obj_path.open("w") as file:
        file.write(
            ("v %r %r %r\n" * len(positions)) % tuple(positions.ravel().tolist())
        )
        file.write(("vt %r %r\n" * len(uvs)) % tuple(uvs.ravel().tolist()))
        file.write(
            ("f %d/%d %d/%d %d/%d\n" * len(triangles))
            % tuple(numpy.repeat(triangles + 1, 2, axis=1).ravel().tolist())
        )
    # optcuts _fixed files hold the pins on the first line and the mode on the second
    (folder / f"{RELAX_INPUT_STEM}_fixed").write_text("\nnocut")
    output = folder / "output"
    args = [
        *optcuts.engine_command,
        "-i",
        obj_path,
        "-o",
        f"{output}{os.sep}",
        *optcuts.engine_args,
    ]
    out_path = output / f"{RELAX_INPUT_STEM}.obj"
    run = start_run(folder, args, optcuts.env, out_path, 0, _read_relaxed)
    out_uvs, uv_corners = poll_run(run, cancelled)
    relaxed = numpy.empty_like(uvs)
    relaxed[triangles.ravel()] = out_uvs[uv_corners.ravel()]
    # optcuts moves and scales its output into the unit square
    edges = positions[triangles[:, 1:]] - positions[triangles[:, :1]]
    area = numpy.linalg.norm(numpy.cross(edges[:, 0], edges[:, 1]), axis=1).sum()
    uv_edges = relaxed[triangles[:, 1:]] - relaxed[triangles[:, :1]]
    uv_area = numpy.abs(
        uv_edges[:, 0, 0] * uv_edges[:, 1, 1] - uv_edges[:, 0, 1] * uv_edges[:, 1, 0]
    ).sum()
    return relaxed * numpy.sqrt(area / uv_area)


def painted_flatten_job(
    command,
    optcuts,
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
    island_positions = numpy.asarray(positions, dtype=FLOAT)[used]
    in_overlapping_island, corner_uvs = _flatten(
        command,
        workdir,
        island_positions,
        local_corners,
        painted_totals,
        cancelled,
    )
    vertex_uvs = numpy.empty((len(used), 2))
    vertex_uvs[local_corners] = corner_uvs

    following = local_corners[next_corners_in_face(painted_totals)]
    labels = lowest_index_part_labels(
        len(used), numpy.stack([local_corners, following], axis=1)
    )
    face_labels = labels[local_corners[face_starts(painted_totals)]]
    triangles, face_of_triangle = _fan_triangles(local_corners, painted_totals)
    # optcuts nocut refuses a folded input chart
    for label in numpy.unique(face_labels[~in_overlapping_island]).tolist():
        vertices = numpy.flatnonzero(labels == label)
        local = numpy.full(len(used), -1, dtype=INTEGER)
        local[vertices] = numpy.arange(len(vertices))
        vertex_uvs[vertices] = _relaxed(
            optcuts,
            workdir,
            island_positions[vertices],
            local[triangles[face_labels[face_of_triangle] == label]],
            vertex_uvs[vertices],
            cancelled,
        )
    overlapping_islands = len(numpy.unique(face_labels[in_overlapping_island]))
    return PaintedFlatten(overlapping_islands, vertex_uvs[local_corners])
