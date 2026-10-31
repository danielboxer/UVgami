import functools

import numpy

from .no_bpy import OptcutsRun, preseed_job
from .no_bpy.island_repair import finish_uvs
from .no_bpy.mesh import corners_of_faces, undirected_edge_keys
from .no_bpy.worker import WorkerProcess
from .ops.guides import exposure_weights
from .utils.mesh import (
    edge_vertex_pairs,
    face_uvs,
    loop_starts,
    loop_totals,
    loop_uvs,
    mesh_arrays,
    set_loop_uvs,
    vertex_positions,
)
from .utils.paths import get_extension_dir_path
from .utils.uvgami_tools import tools_command


def optcuts_run(engine_args):
    from .engines import ENGINES

    engine = ENGINES["OPTCUTS"]
    engine_ctx = engine.validate()
    return OptcutsRun(
        engine.command_prefix(engine_ctx), engine.build_env(engine_ctx), engine_args
    )


def _packed(pairs):
    array = numpy.array(list(pairs), dtype=numpy.int64).reshape(-1, 2)
    return undirected_edge_keys(array[:, 0], array[:, 1])


def _edge_keys(mesh):
    pairs = edge_vertex_pairs(mesh)
    return undirected_edge_keys(pairs[:, 0], pairs[:, 1])


def apply_seams(mesh, seams):
    mesh.edges.foreach_set("use_seam", numpy.isin(_edge_keys(mesh), _packed(seams)))


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


def apply_loop_uvs(mesh, uvs, faces):
    faces = numpy.asarray(faces, dtype=numpy.int64)
    coords = loop_uvs(mesh)
    coords[corners_of_faces(loop_starts(mesh), loop_totals(mesh), faces)] = uvs
    set_loop_uvs(mesh, coords)


def preseed_work(obj, angle, split_target, engine_args, weights, surface_modes, plane):
    mesh = obj.data
    worker = WorkerProcess(
        preseed_job,
        tools_command(),
        optcuts_run(engine_args),
        get_extension_dir_path() / "preseed",
        *mesh_arrays(mesh),
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
    apply_seams(mesh, result.seams)
    if not mesh.uv_layers:
        mesh.uv_layers.new()
    apply_loop_uvs(mesh, result.uvs, result.flattened_faces)
    return True


def finish_hard_surface(obj, faces, ranges):
    mesh = obj.data
    finished = finish_uvs(
        tools_command(),
        get_extension_dir_path() / "repair",
        vertex_positions(mesh),
        faces,
        face_uvs(mesh),
        ranges,
    )
    if not len(finished.faces):
        return
    coords = loop_uvs(mesh)
    coords[loop_starts(mesh)[finished.faces] + finished.corners] = finished.uvs
    set_loop_uvs(mesh, coords)
