import functools

import numpy

from .ops.guides import SEAM_RESTRICTIONS_GROUP, exposure_weights
from .seams import EngineError, preseed_job
from .seams.worker import WorkerProcess
from .utils.mesh import (
    face_vertex_arrays,
    loop_starts,
    loop_totals,
    loop_uvs,
    set_loop_uvs,
    vertex_position_array,
)
from .utils.paths import get_extension_dir_path


# the same group the engine reads, higher repels seams
def seam_restrictions(obj):
    group = obj.vertex_groups.get(SEAM_RESTRICTIONS_GROUP)
    if group is None:
        return None
    weights = {}
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group == group.index and g.weight:
                weights[v.index] = g.weight
                break
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
    return (pairs.min(axis=1) << 32) | pairs.max(axis=1)


# edges given as (low, high) index pairs
def apply_seams(mesh, seams):
    mesh.edges.foreach_set("use_seam", numpy.isin(_edge_keys(mesh), _packed(seams)))


# like apply_seams, but the edges of these faces keep their marks
def apply_seams_except_faces(mesh, faces, seams):
    loop_total = numpy.empty(len(mesh.polygons), dtype=numpy.int64)
    mesh.polygons.foreach_get("loop_total", loop_total)
    loop_edges = numpy.empty(len(mesh.loops), dtype=numpy.int64)
    mesh.loops.foreach_get("edge_index", loop_edges)
    kept_faces = numpy.zeros(len(mesh.polygons), dtype=bool)
    kept_faces[list(faces)] = True
    kept = numpy.zeros(len(mesh.edges), dtype=bool)
    kept[loop_edges[numpy.repeat(kept_faces, loop_total)]] = True
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


def apply_face_uvs(mesh, uvs, only):
    coords = loop_uvs(mesh)
    starts = loop_starts(mesh)
    for f in only:
        start = int(starts[f])
        coords[start : start + len(uvs[f])] = uvs[f]
    set_loop_uvs(mesh, coords)


# uvs has the loops of these faces in face order
def apply_loop_uvs(mesh, uvs, faces):
    flattened = numpy.zeros(len(mesh.polygons), dtype=bool)
    flattened[faces] = True
    coords = loop_uvs(mesh)
    coords[numpy.repeat(flattened, loop_totals(mesh))] = uvs
    set_loop_uvs(mesh, coords)


def hardsurface_command():
    # imported here: engines imports this module back
    from .engines.bundled import resolve_bundled_command

    try:
        return resolve_bundled_command("hardsurface", "Hard surface")
    except RuntimeError as error:
        raise EngineError(str(error)) from error


# compute is bpy-free so a worker thread can run it, the exposure ray cast is not
def preseed_work(obj, angle, split_target, weights, surface_modes):
    mesh = obj.data
    worker = WorkerProcess(
        preseed_job,
        hardsurface_command(),
        get_extension_dir_path() / "preseed",
        vertex_position_array(mesh),
        *face_vertex_arrays(mesh),
        weights,
        exposure_weights(obj),
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
