import functools

import numpy

from .ops.guides import exposure_weights, seam_restriction_weights
from .seams import EngineError, OptcutsRun, preseed_job
from .seams.mesh import edge_keys, index_ranges
from .seams.worker import WorkerProcess
from .utils.mesh import (
    face_vertex_arrays,
    loop_starts,
    loop_totals,
    loop_uvs,
    set_loop_uvs,
    vertex_position_array,
)
from .utils.paths import get_extension_dir_path, get_preferences
from .utils.uvgami_tools import tools_command


def optcuts_run(engine_args):
    # imported here: engines imports this module back
    from .engines import get_engine

    engine = get_engine("OPTCUTS")
    engine_ctx, error = engine.validate(get_preferences())
    if error is not None:
        raise EngineError(error)
    return OptcutsRun(
        engine.command(engine_ctx), engine.build_env(engine_ctx), engine_args
    )


# the same group the engine reads, higher repels seams
def seam_restrictions(obj):
    weights = {v: w for v, w in seam_restriction_weights(obj).items() if w}
    return weights or None


# (low, high) index pairs packed into one integer each
def _packed(pairs):
    return numpy.fromiter(
        ((a << 32) | b for a, b in pairs), dtype=numpy.int64, count=len(pairs)
    )


def _edge_keys(mesh):
    pairs = numpy.empty(len(mesh.edges) * 2, dtype=numpy.int64)
    mesh.edges.foreach_get("vertices", pairs)
    pairs = pairs.reshape(-1, 2)
    return edge_keys(pairs[:, 0], pairs[:, 1])


# edges given as (low, high) index pairs
def apply_seams(mesh, seams):
    mesh.edges.foreach_set("use_seam", numpy.isin(_edge_keys(mesh), _packed(seams)))


# like apply_seams, but the edges of these faces keep their marks
def apply_seams_except_faces(mesh, faces, seams):
    loop_edges = numpy.empty(len(mesh.loops), dtype=numpy.int64)
    mesh.loops.foreach_get("edge_index", loop_edges)
    kept_faces = numpy.zeros(len(mesh.polygons), dtype=bool)
    kept_faces[list(faces)] = True
    kept = numpy.zeros(len(mesh.edges), dtype=bool)
    kept[loop_edges[numpy.repeat(kept_faces, loop_totals(mesh))]] = True
    flags = numpy.isin(_edge_keys(mesh), _packed(seams))
    flags[kept] = seam_flags(mesh)[kept]
    mesh.edges.foreach_set("use_seam", flags)


# every other edge's mark is left as it was
def apply_interior_seams(mesh, interior, seams):
    keys = _edge_keys(mesh)
    flags = seam_flags(mesh)
    inside = numpy.isin(keys, _packed(interior))
    flags[inside] = numpy.isin(keys[inside], _packed(seams))
    mesh.edges.foreach_set("use_seam", flags)


def seam_flags(mesh):
    flags = numpy.empty(len(mesh.edges), dtype=bool)
    mesh.edges.foreach_get("use_seam", flags)
    return flags


# def marked_seams(mesh):
#     pairs = numpy.empty(len(mesh.edges) * 2, dtype=numpy.int64)
#     mesh.edges.foreach_get("vertices", pairs)
#     pairs = numpy.sort(pairs.reshape(-1, 2)[seam_flags(mesh)], axis=1)
#     return {(a, b) for a, b in pairs.tolist()}


# uvs has the loops of these faces, in the order faces lists them
def apply_loop_uvs(mesh, uvs, faces):
    faces = numpy.asarray(faces, dtype=numpy.int64)
    totals = numpy.asarray(loop_totals(mesh), dtype=numpy.int64)
    coords = loop_uvs(mesh)
    coords[index_ranges(loop_starts(mesh)[faces], totals[faces])] = uvs
    set_loop_uvs(mesh, coords)


# compute is bpy-free so a worker thread can run it, the exposure ray cast is not
def preseed_work(obj, angle, split_target, engine_args, weights, surface_modes, plane):
    mesh = obj.data
    worker = WorkerProcess(
        preseed_job,
        tools_command(),
        optcuts_run(engine_args),
        get_extension_dir_path() / "preseed",
        vertex_position_array(mesh),
        *face_vertex_arrays(mesh),
        weights,
        exposure_weights(obj),
        plane,
        surface_modes,
        angle,
        split_target,
    )
    return worker, functools.partial(_apply_preseed, mesh)


def _apply_preseed(mesh, result):
    if result is None:
        return False
    seams, uvs, flattened = result
    apply_seams(mesh, seams)
    if not mesh.uv_layers:
        mesh.uv_layers.new()
    apply_loop_uvs(mesh, uvs, flattened)
    return True
