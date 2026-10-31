import collections
import os

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
from .worker import remove_folder

ENGINE_INPUT_STEM = "mesh"


# # compact copy of just these faces
# def submesh(verts, faces, subset):
#     vmap = {}
#     sub_verts = []
#     sub_faces = []
#     for f in subset:
#         face = []
#         for v in faces[f]:
#             idx = vmap.get(v)
#             if idx is None:
#                 idx = vmap[v] = len(sub_verts)
#                 sub_verts.append(verts[v])
#             face.append(idx)
#         sub_faces.append(tuple(face))
#     return sub_verts, sub_faces


SURFACE_MODE_CODES = {"ORGANIC": 1, "HARD": 2, "AUTO": 3}

SegmentRegions = collections.namedtuple(
    "SegmentRegions", "surface_modes border_edges region_count labels"
)

# exposure only breaks ties
VISIBILITY_SHARE = 0.1


def restriction_weights(painted, exposure):
    weights = {
        vertex: VISIBILITY_SHARE * share
        for vertex, share in (exposure or {}).items()
        if share > 0
    }
    for vertex, weight in (painted or {}).items():
        weights[vertex] = max(weights.get(vertex, 0.0), weight)
    return weights or None


HARDSURFACE_SEAMS_NAME = "seams.txt"
RESOLVED_MODES_NAME = "modes.txt"
HARDSURFACE_UVS_NAME = "uvs.txt"
HALVES_NAME = "halves"

HardSurfacePreseed = collections.namedtuple(
    "HardSurfacePreseed", "seams uvs flattened_faces"
)


def _read_hardsurface(uvs_path, face_count):
    seams_text = (uvs_path.parent / HARDSURFACE_SEAMS_NAME).read_text()
    seams = {
        tuple(int(vertex) for vertex in line.split())
        for line in seams_text.splitlines()
    }
    rows = [line.split() for line in uvs_path.read_text().splitlines()]
    if not rows:
        return None
    flattened = numpy.array([int(row[0]) for row in rows], dtype=numpy.int64)
    if flattened.max() >= face_count:
        raise EngineError(f"hard surface engine flattened face {flattened.max()}")
    loop_uvs = numpy.array(
        [float(value) for row in rows for value in row[1:]], dtype=numpy.float64
    ).reshape(-1, 2)
    return HardSurfacePreseed(seams, loop_uvs, flattened)


def _read_half_count(run_folder, face_count):
    return int(done_text(run_folder))


def _read_nothing(run_folder, face_count):
    return None


def _read_modes(path, face_count):
    modes = numpy.loadtxt(path, dtype=numpy.int64, ndmin=1)
    if len(modes) != face_count:
        raise EngineError(
            f"hard surface engine classified {len(modes)} faces, expected {face_count}"
        )
    return modes


def _start(folder, args, env, out_path, face_count, read):
    run_folder = new_run_folder(folder)
    return start_run(run_folder, args, env, out_path or run_folder, face_count, read)


def _write_weights(path, weights):
    if weights:
        path.write_text(
            ",".join(
                f"{vertex},{float(weight)!r}" for vertex, weight in weights.items()
            )
            + "\n"
        )


def write_modes(path, surface_modes):
    path.write_text(
        "".join(f"{mode}\n" for mode in numpy.asarray(surface_modes).tolist())
    )


def resolved_surface_modes(
    command, workdir, positions, faces, surface_modes, cancelled=None
):
    surface_modes = numpy.asarray(surface_modes, dtype=numpy.int64)
    if (surface_modes == SURFACE_MODE_CODES["ORGANIC"]).all():
        return surface_modes.copy()
    folder = new_run_folder(workdir)
    try:
        obj_path = folder / f"{ENGINE_INPUT_STEM}.obj"
        write_obj(obj_path, positions, faces)
        write_modes(folder / f"{ENGINE_INPUT_STEM}_modes", surface_modes)
        modes_path = folder / RESOLVED_MODES_NAME
        args = [*command.args, "classify", obj_path, modes_path]
        run = _start(folder, args, command.env, modes_path, len(faces), _read_modes)
        return poll_run(run, cancelled)
    finally:
        remove_folder(folder)


OptcutsRun = collections.namedtuple("OptcutsRun", "engine_command env engine_args")


def preseed_job(
    command,
    optcuts,
    workdir,
    positions,
    corners,
    totals,
    weights,
    exposure,
    plane,
    surface_modes,
    angle,
    split_target,
    cancelled=None,
):
    faces = faces_from_arrays(corners, totals)
    folder = new_run_folder(workdir)
    try:
        obj_path = folder / f"{ENGINE_INPUT_STEM}.obj"
        write_obj(obj_path, positions.tolist(), faces)
        _write_weights(
            folder / f"{ENGINE_INPUT_STEM}_weights",
            restriction_weights(weights, exposure),
        )
        write_modes(folder / f"{ENGINE_INPUT_STEM}_modes", surface_modes)
        plane_args = []
        if plane is not None:
            axis, partners = plane
            (folder / f"{ENGINE_INPUT_STEM}_partners").write_text(
                "".join(f"{vertex} {image}\n" for vertex, image in partners.items())
            )
            _write_weights(folder / f"{ENGINE_INPUT_STEM}_painted", weights)
            halves = folder / HALVES_NAME
            halves_args = [
                *command.args,
                "halves",
                obj_path,
                halves,
                "--plane-axis",
                axis,
            ]
            count = poll_run(
                _start(
                    folder, halves_args, command.env, None, len(faces), _read_half_count
                ),
                cancelled,
            )
            for index in range(count):
                engine_args = [
                    *optcuts.engine_command,
                    "-i",
                    halves / f"half{index}.obj",
                    "-o",
                    f"{halves / 'out'}{os.sep}",
                    *optcuts.engine_args,
                ]
                run = _start(folder, engine_args, optcuts.env, None, 0, _read_nothing)
                poll_run(run, cancelled)
            plane_args = ["--plane-axis", axis, "--halves", halves]
        uvs_path = folder / HARDSURFACE_UVS_NAME
        args = [
            *command.args,
            "preseed",
            obj_path,
            folder / HARDSURFACE_SEAMS_NAME,
            uvs_path,
            "--angle",
            angle,
            "--split-target",
            repr(float(split_target)),
            *plane_args,
        ]
        run = _start(folder, args, command.env, uvs_path, len(faces), _read_hardsurface)
        return poll_run(run, cancelled)
    finally:
        remove_folder(folder)


def classify_job(
    hardsurface_command,
    workdir,
    positions,
    corners,
    totals,
    surface_modes,
    cancelled=None,
):
    faces = faces_from_arrays(corners, totals)
    return resolved_surface_modes(
        hardsurface_command,
        workdir,
        positions.tolist(),
        faces,
        surface_modes,
        cancelled,
    )
