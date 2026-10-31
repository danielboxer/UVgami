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
from .mesh import faces_from_arrays

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


def _run(
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
    run = start_run(
        folder,
        [*command.args, subcommand, *paths, *flags],
        command.env,
        folder,
        len(faces),
        lambda path, _: read(path),
    )
    return poll_run(run, None)


def partner_map(partner):
    return {v: int(m) for v, m in enumerate(partner.tolist()) if m >= 0}


def symmetric_axis(command, workdir, positions, corners, totals, center, tolerance):
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
    return _run(command, workdir, "mirror-axis", positions, faces, flags, read)


def mutual_mirror_partners(command, workdir, positions, center, axis):
    def read(folder):
        done_text(folder)
        return partner_map(_read_integers(folder))

    flags = _plane_flags(center, axis=axis)
    return _run(command, workdir, "mirror-partners", positions, [], flags, read)


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
    return _run(
        command, workdir, "mirror-regions", positions, faces, flags, read, labels=labels
    )


def mirrored_uvs(
    command, workdir, positions, corners, totals, uvs, center, axis, tolerance
):
    def read(folder):
        done_text(folder)
        values = numpy.loadtxt(folder / OUTPUT_NAME, dtype=numpy.float64, ndmin=2)
        if values.shape != uvs.shape:
            raise EngineError(
                f"the mirror engine wrote {values.shape} uvs for {uvs.shape}"
            )
        return values

    faces = faces_from_arrays(corners, totals)
    flags = _plane_flags(center, axis=axis, tolerance=tolerance)
    return _run(command, workdir, "mirror-uvs", positions, faces, flags, read, uvs=uvs)
