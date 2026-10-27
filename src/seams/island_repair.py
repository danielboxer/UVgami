import itertools
import tempfile
from collections import namedtuple
from pathlib import Path

import numpy

from .engine_run import EngineRun, launch, poll_run

INPUT_NAME = "input.bin"
OUTPUT_NAME = "output.bin"
INTEGER = numpy.dtype("<i8")
FLOAT = numpy.dtype("<f8")

# the moved_ arrays hold one entry per moved corner, cuts holds the new seams
# islands are the face groups with the cuts as seams, in order of their lowest face
SplitMoves = namedtuple(
    "SplitMoves", "moved_faces moved_corners moved_uvs cuts islands"
)
# targets maps (face, corner) to its uv, a pinned plan has targets on its boundary only
RectifyPlan = namedtuple("RectifyPlan", "group targets pinned")


def _integers(values):
    return numpy.asarray(values, dtype=INTEGER)


def _face_arrays(uvs):
    sizes = numpy.fromiter((len(face) for face in uvs), INTEGER, len(uvs))
    flat = itertools.chain.from_iterable(itertools.chain.from_iterable(uvs))
    return sizes, numpy.fromiter(flat, FLOAT, 2 * int(sizes.sum()))


def _island_numbers(groups, face_count):
    island = numpy.empty(face_count, dtype=INTEGER)
    for index, group in enumerate(groups):
        island[group] = index
    return island


# arrays one after another, each length known from the ones before it
def _reader(data):
    offset = 0

    def take(dtype, count):
        nonlocal offset
        values = numpy.frombuffer(data, dtype, count, offset)
        offset += values.nbytes
        return values

    return take


def _groups(island):
    order = numpy.argsort(island, kind="stable")
    bounds = numpy.cumsum(numpy.bincount(island))[:-1]
    return [group.tolist() for group in numpy.split(order, bounds)]


# the arrays go in as the command's input, read gets its output's bytes
def _run(command, workdir, subcommand, arrays, read):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(dir=workdir))
    with (folder / INPUT_NAME).open("wb") as file:
        for array in arrays:
            file.write(array.tobytes())
    paths = [folder / INPUT_NAME, folder / OUTPUT_NAME]
    args = [str(arg) for arg in [*command, subcommand, *paths]]
    run = EngineRun(
        launch(folder, args, None),
        folder,
        paths[1],
        0,
        lambda path, _: read(path.read_bytes()),
    )
    return poll_run(run, None)


# ranges are (start, stop) faces, each one engine output with its own uv scale
def split_moves(command, workdir, verts, faces, uvs, ranges):
    sizes, flat_uvs = _face_arrays(uvs)
    corners = numpy.fromiter(
        itertools.chain.from_iterable(faces), INTEGER, int(sizes.sum())
    )
    arrays = [
        _integers([len(verts), len(faces), len(ranges)]),
        numpy.asarray(verts, dtype=FLOAT),
        sizes,
        corners,
        flat_uvs,
        _integers(ranges).reshape(-1),
    ]

    def read(data):
        take = _reader(data)
        move_count, cut_count = take(INTEGER, 2).tolist()
        moved_faces = take(INTEGER, move_count)
        moved_corners = take(INTEGER, move_count)
        moved_uvs = take(FLOAT, 2 * move_count).reshape(-1, 2)
        pairs = take(INTEGER, 2 * cut_count).reshape(-1, 2).tolist()
        islands = _groups(take(INTEGER, len(faces)))
        cuts = {(a, b) for a, b in pairs}
        return SplitMoves(moved_faces, moved_corners, moved_uvs, cuts, islands)

    return _run(command, workdir, "split-moves", arrays, read)


# groups lists every face once
def rectify_plans(command, workdir, uvs, groups):
    sizes, flat_uvs = _face_arrays(uvs)
    arrays = [
        _integers([len(uvs)]),
        sizes,
        flat_uvs,
        _island_numbers(groups, len(uvs)),
    ]

    def read(data):
        take = _reader(data)
        (count,) = take(INTEGER, 1).tolist()
        rows = take(INTEGER, 4 * count).reshape(-1, 4).tolist()
        targets = take(FLOAT, 2 * count).reshape(-1, 2).tolist()
        plans = {}
        for (island, pinned, face, corner), uv in zip(rows, targets):
            if island not in plans:
                plans[island] = RectifyPlan(groups[island], {}, bool(pinned))
            plans[island].targets[face, corner] = tuple(uv)
        return list(plans.values())

    return _run(command, workdir, "rectify", arrays, read)
