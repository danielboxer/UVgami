import collections
import os
import tempfile
from pathlib import Path

import numpy

from .cuts import restriction_weights
from .engine_run import EngineError, EngineRun, launch, poll_run, stdout_path, write_obj
from .mesh import faces_from_arrays
from .worker import remove_folder

# the stem of the engine's input obj, its sidecars and its output obj
ENGINE_INPUT_STEM = "mesh"


# compact copy of just these faces
def submesh(verts, faces, subset):
    vmap = {}
    sub_verts = []
    sub_faces = []
    for f in subset:
        face = []
        for v in faces[f]:
            idx = vmap.get(v)
            if idx is None:
                idx = vmap[v] = len(sub_verts)
                sub_verts.append(verts[v])
            face.append(idx)
        sub_faces.append(tuple(face))
    return sub_verts, sub_faces


SURFACE_MODE_CODES = {"ORGANIC": 1, "HARD": 2, "AUTO": 3}


HARDSURFACE_SEAMS_NAME = "seams.txt"
HARD_FACES_NAME = "hard_faces.txt"
HARDSURFACE_UVS_NAME = "uvs.txt"
HALVES_NAME = "halves"
DONE_PREFIX = "done: "


# a flattened face's index then its corner uvs on each line, None when nothing flattened
def _read_hardsurface(uvs_path, face_count):
    seams_text = (uvs_path.parent / HARDSURFACE_SEAMS_NAME).read_text()
    seams = {tuple(int(v) for v in line.split()) for line in seams_text.splitlines()}
    rows = [line.split() for line in uvs_path.read_text().splitlines()]
    if not rows:
        return None
    flattened = numpy.array([int(row[0]) for row in rows], dtype=numpy.int64)
    if flattened.max() >= face_count:
        raise EngineError(f"hard surface engine flattened face {flattened.max()}")
    loop_uvs = numpy.array(
        [float(value) for row in rows for value in row[1:]], dtype=numpy.float64
    ).reshape(-1, 2)
    return seams, loop_uvs, flattened


def _read_half_count(run_folder, face_count):
    done = [
        line
        for line in stdout_path(run_folder).read_text().splitlines()
        if line.startswith(DONE_PREFIX)
    ]
    if len(done) != 1:
        raise EngineError("hard surface engine printed no half part count")
    return int(done[0][len(DONE_PREFIX) :])


def _read_nothing(run_folder, face_count):
    return None


def _read_hard_faces(path, face_count):
    flags = numpy.loadtxt(path, dtype=numpy.int64, ndmin=1)
    if len(flags) != face_count:
        raise EngineError(
            f"hard surface engine classified {len(flags)} faces, expected {face_count}"
        )
    return flags != 0


# each run keeps its output files in a folder of its own, removed once read
def _start(folder, args, env, out_path, face_count, read):
    run_folder = Path(tempfile.mkdtemp(dir=folder))
    process = launch(run_folder, [str(arg) for arg in args], env)
    return EngineRun(process, run_folder, out_path or run_folder, face_count, read)


def _write_weights(path, weights):
    if weights:
        path.write_text(
            ",".join(f"{v},{float(w)!r}" for v, w in weights.items()) + "\n"
        )


def _write_modes(path, surface_modes):
    path.write_text(
        "".join(f"{mode}\n" for mode in numpy.asarray(surface_modes).tolist())
    )


# a loose part takes its first face's mode, Auto parts get the classifier's pick
def hard_faces(command, workdir, verts, faces, surface_modes, cancelled=None):
    if (numpy.asarray(surface_modes) == SURFACE_MODE_CODES["ORGANIC"]).all():
        return numpy.zeros(len(faces), dtype=bool)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(dir=workdir))
    try:
        obj_path = folder / f"{ENGINE_INPUT_STEM}.obj"
        write_obj(obj_path, verts, faces)
        _write_modes(folder / f"{ENGINE_INPUT_STEM}_modes", surface_modes)
        flags_path = folder / HARD_FACES_NAME
        args = [*command, "classify", obj_path, flags_path]
        run = _start(folder, args, None, flags_path, len(faces), _read_hard_faces)
        return poll_run(run, cancelled)
    finally:
        remove_folder(folder)


# optcuts' command, environment and arguments for the organic half's unwraps
OptcutsRun = collections.namedtuple("OptcutsRun", "engine_command env engine_args")


# with a plane, optcuts unwraps the organic half's parts between the exe's two runs
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
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(dir=workdir))
    try:
        obj_path = folder / f"{ENGINE_INPUT_STEM}.obj"
        write_obj(obj_path, positions.tolist(), faces)
        _write_weights(
            folder / f"{ENGINE_INPUT_STEM}_weights",
            restriction_weights(weights, exposure),
        )
        _write_modes(folder / f"{ENGINE_INPUT_STEM}_modes", surface_modes)
        plane_args = []
        if plane is not None:
            axis, partners = plane
            (folder / f"{ENGINE_INPUT_STEM}_partners").write_text(
                "".join(f"{v} {image}\n" for v, image in partners.items())
            )
            _write_weights(folder / f"{ENGINE_INPUT_STEM}_painted", weights)
            halves = folder / HALVES_NAME
            halves_args = [*command, "halves", obj_path, halves, "--plane-axis", axis]
            count = poll_run(
                _start(folder, halves_args, None, None, len(faces), _read_half_count),
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
            *command,
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
        run = _start(folder, args, None, uvs_path, len(faces), _read_hardsurface)
        return poll_run(run, cancelled)
    finally:
        remove_folder(folder)
