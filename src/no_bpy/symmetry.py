from collections import namedtuple

import numpy

from .engine_run import (
    EngineError,
    done_text,
    new_run_folder,
    poll_run,
    start_run,
    write_obj,
)
from .mesh import face_starts, faces_from_arrays

INPUT_NAME = "mirror.obj"
LABELS_NAME = "labels.txt"
UVS_NAME = "uvs.txt"
OUTPUT_NAME = "output.txt"

MirrorAxis = namedtuple("MirrorAxis", "axis exact crosses partners")


def _read_integers(folder):
    return numpy.array((folder / OUTPUT_NAME).read_text().split(), dtype=numpy.int64)


def _plane_flags(center, axis=None, tolerance=None):
    flags = ["--center", ",".join(repr(float(value)) for value in center)]
    if axis is not None:
        flags += ["--axis", axis]
    if tolerance is not None:
        flags += ["--tolerance", repr(float(tolerance))]
    return flags


def _start(
    command, workdir, subcommand, positions, faces, flags, read, labels=None, uvs=None
):
    folder = new_run_folder(workdir)
    input_path = folder / INPUT_NAME
    write_obj(input_path, numpy.asarray(positions, dtype=numpy.float64).tolist(), faces)
    paths = [input_path]
    if labels is not None:
        paths.append(folder / LABELS_NAME)
        paths[-1].write_text("".join(f"{label}\n" for label in labels.tolist()))
    if uvs is not None:
        paths.append(folder / UVS_NAME)
        paths[-1].write_text("".join(f"{u!r} {v!r}\n" for u, v in uvs.tolist()))
    paths.append(folder / OUTPUT_NAME)
    return start_run(
        folder,
        [*command.args, subcommand, *paths, *flags],
        command.env,
        folder,
        len(faces),
        lambda path, _: read(path),
    )


def faces_mirror(corners, totals, partners):
    starts = face_starts(totals)
    for size in numpy.unique(totals):
        rows = corners[starts[totals == size][:, None] + numpy.arange(size)]
        mirrored = partners[rows]
        if (mirrored < 0).any():
            return False
        own = numpy.unique(numpy.sort(rows, axis=1), axis=0)
        combined = numpy.concatenate([own, numpy.sort(mirrored, axis=1)])
        if len(numpy.unique(combined, axis=0)) != len(own):
            return False
    return True


def partner_map(partner):
    return {v: int(m) for v, m in enumerate(partner.tolist()) if m >= 0}


def face_partners(corners, totals, vertex_partners):
    starts = face_starts(totals)
    partners = numpy.full(len(totals), -1, dtype=numpy.int64)
    for size in numpy.unique(totals).tolist():
        faces = numpy.flatnonzero(totals == size)
        rows = corners[starts[faces][:, None] + numpy.arange(size)]
        face_of_vertices = {
            tuple(row): face
            for row, face in zip(numpy.sort(rows, axis=1).tolist(), faces.tolist())
        }
        mirrored = vertex_partners[rows]
        complete = (mirrored >= 0).all(axis=1)
        mirrored_rows = numpy.sort(mirrored[complete], axis=1).tolist()
        for face, row in zip(faces[complete].tolist(), mirrored_rows):
            partners[face] = face_of_vertices.get(tuple(row), -1)
    return partners


def region_partners(labels, face_partners):
    mirrored = face_partners >= 0
    pairs = numpy.stack([labels[mirrored], labels[face_partners[mirrored]]], axis=1)
    pairs, counts = numpy.unique(
        pairs[(pairs > 0).all(axis=1)], axis=0, return_counts=True
    )
    # dict keeps the last value per key
    return dict(pairs[numpy.argsort(counts, kind="stable")].tolist())


def start_symmetric_axis(
    command, workdir, positions, corners, totals, center, tolerance
):
    def read(folder):
        words = done_text(folder).split()
        if words == ["none"]:
            return None
        axis, exactness, crossing = words
        return MirrorAxis(
            axis, exactness == "exact", crossing == "crossing", _read_integers(folder)
        )

    faces = faces_from_arrays(corners, totals)
    flags = _plane_flags(center, tolerance=tolerance)
    return _start(command, workdir, "mirror-axis", positions, faces, flags, read)


def mutual_mirror_partners(command, workdir, positions, center, axis):
    def read(folder):
        done_text(folder)
        return partner_map(_read_integers(folder))

    flags = _plane_flags(center, axis=axis)
    run = _start(command, workdir, "mirror-partners", positions, [], flags, read)
    return poll_run(run, None)


MirroredRegions = namedtuple(
    "MirroredRegions", "kept_side labels border_edges region_count"
)


def symmetric_regions(
    command, workdir, positions, corners, totals, labels, axis, center, tolerance
):
    def read(folder):
        kept_side, region_count = map(int, done_text(folder).split())
        values = _read_integers(folder)
        edges = values[len(faces) :].reshape(-1, 2).tolist()
        return MirroredRegions(
            kept_side,
            values[: len(faces)],
            [tuple(edge) for edge in edges],
            region_count,
        )

    faces = faces_from_arrays(corners, totals)
    flags = _plane_flags(center, axis=axis, tolerance=tolerance)
    run = _start(
        command, workdir, "mirror-regions", positions, faces, flags, read, labels=labels
    )
    return poll_run(run, None)


class FaceSpansCharts(EngineError):
    pass


class FaceHasNoMirror(EngineError):
    pass


def mirrored_uvs(
    command, workdir, positions, corners, totals, uvs, center, axis, tolerance
):
    def read(folder):
        if done_text(folder) == "spans-charts":
            raise FaceSpansCharts(
                "a face's mirror image has its corners in two uv islands"
            )
        if done_text(folder) == "no-mirror":
            raise FaceHasNoMirror("a face on the negative side has no mirror face")
        values = numpy.loadtxt(folder / OUTPUT_NAME, dtype=numpy.float64, ndmin=2)
        if values.shape != uvs.shape:
            raise EngineError(
                f"the mirror engine wrote {values.shape} uvs for {uvs.shape}"
            )
        return values

    faces = faces_from_arrays(corners, totals)
    flags = _plane_flags(center, axis=axis, tolerance=tolerance)
    run = _start(command, workdir, "mirror-uvs", positions, faces, flags, read, uvs=uvs)
    return poll_run(run, None)
