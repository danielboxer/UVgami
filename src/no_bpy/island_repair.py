from collections import namedtuple

import numpy

from .engine_run import (
    FLOAT,
    INTEGER,
    corner_array,
    face_arrays,
    packed_reader,
    poll_run,
    start_packed,
)

FinishedUVs = namedtuple("FinishedUVs", "faces corners uvs")


def engine_output_arrays(positions, faces, uvs, ranges):
    sizes, flat_uvs = face_arrays(uvs)
    return [
        numpy.asarray([len(positions), len(faces), len(ranges)], dtype=INTEGER),
        numpy.asarray(positions, dtype=FLOAT),
        sizes,
        corner_array(faces, sizes),
        flat_uvs,
        numpy.asarray(ranges, dtype=INTEGER).reshape(-1),
    ]


def finish_uvs(command, workdir, positions, faces, uvs, ranges):
    def read(data):
        take = packed_reader(data)
        (count,) = take(INTEGER, 1).tolist()
        changed_faces = take(INTEGER, count)
        changed_corners = take(INTEGER, count)
        return FinishedUVs(
            changed_faces, changed_corners, take(FLOAT, 2 * count).reshape(-1, 2)
        )

    arrays = engine_output_arrays(positions, faces, uvs, ranges)
    return poll_run(start_packed(command, workdir, "finish", arrays, read), None)
