import numpy

from .engines.optcuts.flags import PRIORITY_VALUES
from .guides import painted_weight_by_vertex
from .hard_surface import optcuts_run
from .no_bpy.mesh import face_starts
from .no_bpy.painted_islands import painted_flatten_job, painted_outline_job
from .no_bpy.worker import WorkerProcess
from .utils.mesh import (
    activate_uv_layer,
    loop_uvs,
    loop_vertices_and_face_sizes,
    mesh_arrays,
    painted_faces,
    set_loop_uvs,
)
from .utils.paths import get_extension_dir_path
from .utils.uvgami_tools import tools_command

# blender's weight paint turns from blue to red at 0.5
PAINTED_WEIGHT = 0.5


def weight_painted_faces(obj):
    weights = numpy.zeros(len(obj.data.vertices))
    for vertex, weight in painted_weight_by_vertex(obj).items():
        weights[vertex] = weight
    corners, totals = loop_vertices_and_face_sizes(obj.data)
    corner_sums = numpy.add.reduceat(weights[corners], face_starts(totals))
    return corner_sums / totals >= PAINTED_WEIGHT


def outline_work(obj, painted, symmetry):
    plane = None
    if symmetry is not None and symmetry.kind in ("BISECT", "WHOLE"):
        plane = (symmetry.center, symmetry.axis)
    return WorkerProcess(
        painted_outline_job,
        tools_command(),
        get_extension_dir_path() / "paint",
        *mesh_arrays(obj.data),
        painted,
        plane,
    )


def parts_without_painted(parts, mesh, painted):
    corners, totals = loop_vertices_and_face_sizes(mesh)
    face_labels = parts.vertex_labels[corners[face_starts(totals)]]
    counts = parts.triangle_counts.copy()
    counts[numpy.isin(parts.part_labels, face_labels[painted])] = 0
    return parts._replace(triangle_counts=counts)


def flatten_work(obj, painted, priority):
    return WorkerProcess(
        painted_flatten_job,
        tools_command(),
        optcuts_run(["-u", PRIORITY_VALUES[priority]]),
        get_extension_dir_path() / "paint",
        *mesh_arrays(obj.data),
        painted,
    )


def apply_painted_uvs(mesh, painted, corner_uvs):
    activate_uv_layer(mesh)
    uvs = loop_uvs(mesh)
    uvs[numpy.repeat(painted, loop_vertices_and_face_sizes(mesh)[1])] = corner_uvs
    set_loop_uvs(mesh, uvs)


def piece_is_painted(piece):
    painted = painted_faces(piece.data)
    if painted.any() and not painted.all():
        raise RuntimeError(f"{piece.name} mixes painted and unpainted faces")
    return bool(painted.all())
