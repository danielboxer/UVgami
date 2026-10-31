import numpy

from .engine_run import (
    FLOAT,
    INTEGER,
    corner_array,
    face_arrays,
    packed_reader,
    start_packed,
)
from .mesh import corners_of_faces, face_starts, transform_positions


def dense_in_proxy_space(dense, proxy):
    matrix = numpy.linalg.inv(proxy["matrix"]) @ numpy.asarray(
        dense["matrix"], dtype=numpy.float64
    )
    positions = numpy.asarray(dense["positions"], dtype=numpy.float64).reshape(-1, 3)
    positions = transform_positions(positions, matrix)
    normals = numpy.asarray(dense["normals"], dtype=numpy.float64).reshape(-1, 3)
    normals = normals @ numpy.linalg.inv(matrix[:3, :3])
    return positions, normals


def dense_subset(dense, faces):
    sizes = numpy.asarray(dense["face_sizes"], dtype=numpy.int64)
    faces = numpy.asarray(faces, dtype=numpy.int64)
    corners = numpy.asarray(dense["corners"], dtype=numpy.int64)
    corners = corners[corners_of_faces(face_starts(sizes), sizes, faces)]
    used, local = numpy.unique(corners, return_inverse=True)
    subset = {
        "positions": numpy.asarray(dense["positions"], dtype=numpy.float64)[used],
        "normals": numpy.asarray(dense["normals"], dtype=numpy.float64)[used],
        "matrix": dense["matrix"],
        "corners": local,
        "face_sizes": sizes[faces],
    }
    return subset, used


def start_proxy_transfer(command, workdir, dense, proxy):
    positions, normals = dense_in_proxy_space(dense, proxy)
    corners = numpy.asarray(dense["corners"], dtype=INTEGER)
    proxy_positions = numpy.asarray(proxy["positions"], dtype=FLOAT).reshape(-1, 3)
    proxy_sizes, proxy_uvs = face_arrays(proxy["corner_uvs"])
    face_sizes = numpy.asarray(dense["face_sizes"], dtype=INTEGER)
    arrays = [
        numpy.array(
            [
                len(positions),
                len(face_sizes),
                len(proxy_positions),
                len(proxy_sizes),
            ],
            dtype=INTEGER,
        ),
        positions,
        normals,
        face_sizes,
        corners,
        proxy_positions,
        proxy_sizes,
        corner_array(proxy["faces"], proxy_sizes),
        proxy_uvs,
    ]

    def read(data):
        take = packed_reader(data)
        (seam_count,) = take(INTEGER, 1).tolist()
        uvs = take(FLOAT, 2 * len(corners)).reshape(-1, 2)
        pairs = take(INTEGER, 2 * seam_count).reshape(-1, 2).tolist()
        return {(a, b) for a, b in pairs}, uvs

    return start_packed(command, workdir, "proxy-transfer", arrays, read)
