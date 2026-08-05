"""Unwrap a decimated copy, then cut the original along its seams.

The engine only ever sees the proxy, and so does the repair, which is what
makes this fast: every cut is decided on a few thousand triangles instead of
the whole mesh. What comes out is a cut network, redrawn on the original by
snapping each cut edge to a path of real edges, and the dense mesh is then
flattened and packed once. Texel density follows the original, not the proxy,
because the original is really unwrapped.

Chart labels cannot carry a cut. Most of what the engine makes is a slit
inside one chart, which separates nothing and so has no boundary to label,
and a single chart output is nothing but slit."""

import collections

import bmesh
import bpy
from mathutils.kdtree import KDTree

from .hard_surface import (
    apply_face_uvs,
    apply_seams,
    build_seam_uvs,
    flatten_engine,
    marked_seams,
    seam_restrictions,
)
from .seams import check_manifold, face_edges, snap_paths
from .utils.mesh import new_bmesh, set_bmesh

# candidates to pick a facing match from when snapping a cut vertex
NEAREST_VERTS = 8


def make_proxy(obj, target_faces):
    """Decimate obj in place to roughly target_faces triangles.

    Collapsing leaves vertices with no face behind, which the engine reads as
    non-manifold vertices and refuses, so they go before the mesh is used."""
    triangles = sum(len(poly.vertices) - 2 for poly in obj.data.polygons)
    if triangles <= target_faces:
        return False
    bpy.context.view_layer.objects.active = obj
    modifier = obj.modifiers.new("UVgami Proxy", "DECIMATE")
    modifier.ratio = target_faces / triangles
    bpy.ops.object.modifier_apply(modifier=modifier.name)

    bm = new_bmesh(obj)
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    set_bmesh(bm, obj)
    return True


def cut_edges(output):
    """Proxy edges the uv map is torn across, as vertex index pairs.

    An edge already on the mesh boundary is a cut the original has of its own,
    so only interior tears count."""
    data = output.data
    uv = data.uv_layers.active.data
    faces = [tuple(poly.vertices) for poly in data.polygons]
    uvs = [[tuple(uv[i].uv) for i in poly.loop_indices] for poly in data.polygons]

    def corner(f, v):
        return uvs[f][faces[f].index(v)]

    torn = set()
    for (u, v), owners in face_edges(faces).items():
        if len(owners) != 2:
            continue
        f, g = owners
        if corner(f, u) != corner(g, u) or corner(f, v) != corner(g, v):
            torn.add((u, v))
    return torn


def vertex_map(input_mesh, output):
    """Each proxy vertex to the nearest input vertex facing the same way.

    Thin walls put the far side of the wall nearest, and a cut snapped
    through the wall would seam both sides at once."""
    data = input_mesh.data
    matrix = input_mesh.matrix_world
    rotation = matrix.to_3x3().inverted_safe().transposed()
    kd = KDTree(len(data.vertices))
    for i, vertex in enumerate(data.vertices):
        kd.insert(matrix @ vertex.co, i)
    kd.balance()
    normals = [(rotation @ vertex.normal).normalized() for vertex in data.vertices]

    out_matrix = output.matrix_world
    out_rotation = out_matrix.to_3x3().inverted_safe().transposed()
    mapped = []
    for vertex in output.data.vertices:
        facing = out_rotation @ vertex.normal
        found = kd.find_n(out_matrix @ vertex.co, NEAREST_VERTS)
        best = found[0][1]
        for _, index, _ in found:
            if normals[index].dot(facing) > 0:
                best = index
                break
        mapped.append(best)
    return mapped


def proxy_weights(input_mesh, mapped):
    """The painted restrictions on the proxy's own vertices, read off the
    original through the same map the cuts are snapped back with. The engine
    got this paint decimated with the proxy, so both rounds of cutting steer
    by it."""
    if not bpy.context.scene.uvgami.avoid_seams:
        return None
    weights = seam_restrictions(input_mesh)
    if weights is None:
        return None
    moved = {i: weights[v] for i, v in enumerate(mapped) if v in weights}
    return moved or None


def repair_proxy(output, weights):
    """Add the cuts an island needs to flatten, on the proxy.

    What ruins an island is its shape, which the proxy has too, so the repair
    belongs here where a round costs a second instead of a minute."""
    apply_seams(output.data, cut_edges(output))
    build_seam_uvs(output, marked="ONLY", weights=weights)
    return marked_seams(output.data)


def transfer_cuts(input_mesh, output):
    """Seam the original along the proxy's cuts and unwrap it there."""
    mapped = vertex_map(input_mesh, output)
    cuts = repair_proxy(output, proxy_weights(input_mesh, mapped))

    data = input_mesh.data
    verts = [tuple(vertex.co) for vertex in data.vertices]
    adjacent = collections.defaultdict(set)
    for edge in data.edges:
        a, b = edge.vertices
        adjacent[a].add(b)
        adjacent[b].add(a)
    seams = snap_paths(verts, adjacent, mapped, cuts)

    apply_seams(data, seams)
    if not data.uv_layers:
        data.uv_layers.new()
    # the proxy already settled which cuts the shape needs, so the dense mesh
    # only has to flatten and pack once
    faces = [tuple(poly.vertices) for poly in data.polygons]
    check_manifold(faces)
    uvs = flatten_engine().flatten(verts, faces, seams)
    apply_face_uvs(data, uvs)
