import itertools

import numpy

from .engine_run import (
    FLOAT,
    INTEGER,
    face_arrays,
    packed_reader,
    start_packed,
    stdout_path,
)
from .worker import last_progress


# dense positions and normals in the proxy's own space
def proxy_space(dense, proxy):
    matrix = numpy.linalg.inv(proxy["matrix"]) @ numpy.asarray(
        dense["matrix"], dtype=numpy.float64
    )
    positions = numpy.asarray(dense["positions"], dtype=numpy.float64).reshape(-1, 3)
    positions = positions @ matrix[:3, :3].T + matrix[:3, 3]
    normals = numpy.asarray(dense["normals"], dtype=numpy.float64).reshape(-1, 3)
    normals = normals @ numpy.linalg.inv(matrix[:3, :3])
    return positions, normals


# also returns each new vertex's old index
def dense_subset(dense, faces):
    sizes = numpy.asarray(dense["face_sizes"], dtype=numpy.int64)
    starts = numpy.cumsum(sizes) - sizes
    faces = numpy.asarray(faces, dtype=numpy.int64)
    kept_sizes = sizes[faces]
    kept_starts = numpy.cumsum(kept_sizes) - kept_sizes
    offsets = numpy.arange(int(kept_sizes.sum())) - numpy.repeat(
        kept_starts, kept_sizes
    )
    corners = numpy.asarray(dense["corners"], dtype=numpy.int64)
    corners = corners[numpy.repeat(starts[faces], kept_sizes) + offsets]
    used, local = numpy.unique(corners, return_inverse=True)
    subset = {
        "positions": numpy.asarray(dense["positions"], dtype=numpy.float64)[used],
        "normals": numpy.asarray(dense["normals"], dtype=numpy.float64)[used],
        "matrix": dense["matrix"],
        "corners": local,
        "face_sizes": kept_sizes,
    }
    return subset, used


class TransferRun:
    def __init__(self, command, workdir, dense, proxy):
        positions, normals = proxy_space(dense, proxy)
        corners = numpy.asarray(dense["corners"], dtype=INTEGER)
        proxy_positions = numpy.asarray(proxy["positions"], dtype=FLOAT).reshape(-1, 3)
        proxy_sizes, proxy_uvs = face_arrays(proxy["corner_uvs"])
        proxy_corners = numpy.fromiter(
            itertools.chain.from_iterable(proxy["faces"]),
            INTEGER,
            int(proxy_sizes.sum()),
        )
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
            proxy_corners,
            proxy_uvs,
        ]

        # the seams as (low, high) pairs, and one uv per dense corner
        def read(data):
            take = packed_reader(data)
            (seam_count,) = take(INTEGER, 1).tolist()
            uvs = take(FLOAT, 2 * len(corners)).reshape(-1, 2)
            pairs = take(INTEGER, 2 * seam_count).reshape(-1, 2).tolist()
            return {(a, b) for a, b in pairs}, uvs

        self._run = start_packed(command, workdir, "proxy-transfer", arrays, read)

    def done(self):
        return self._run.poll() is not None

    @property
    def progress(self):
        return last_progress(stdout_path(self._run.workdir))

    def result(self):
        return self._run.result()

    def close(self):
        self._run.stop()
