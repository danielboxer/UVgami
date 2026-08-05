from .ops.guides import SEAM_RESTRICTIONS_GROUP
from .seams import (
    CREASE_ANGLE,
    FlattenEngine,
    FlattenError,
    is_hard_surface,
    preseed_uvs,
    vertex_components,
)


def flatten_engine():
    """Client for the optcuts flatten mode, resolved like an unwrap run."""
    # imported here: engines imports this module back
    from .engines import get_engine
    from .utils.paths import get_extension_dir_path, get_preferences

    path, error = get_engine("OPTCUTS").validate(get_preferences())
    if error is not None:
        raise FlattenError(error)
    return FlattenEngine(path, get_extension_dir_path() / "preseed")


def seam_restrictions(obj):
    """Per-vertex weights from the painted guide, the same group the engine
    reads. Higher repels seams."""
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


def hard_faces(verts, faces, marks, marked="NONE"):
    """Face indices of the loose parts worth preseeding, for auto mode.

    Each part classifies on its own geometry. A part carrying marked seams
    counts as hard whenever marks are in use: the user placed seams there
    deliberately. With marked ONLY detection never runs, so the marked parts
    are the whole hard set."""
    marked_verts = {v for edge in marks for v in edge} if marked != "NONE" else set()
    hard = set()
    for comp in vertex_components(faces):
        if (marked_verts and marked_verts & {v for fi in comp for v in faces[fi]}) or (
            marked != "ONLY" and is_hard_surface(verts, [faces[fi] for fi in comp])
        ):
            hard.update(comp)
    return hard


def auto_hard_faces(obj, marked="NONE"):
    mesh = obj.data
    verts = [tuple(v.co) for v in mesh.vertices]
    faces = [tuple(p.vertices) for p in mesh.polygons]
    return hard_faces(verts, faces, marked_seams(mesh), marked)


def apply_seams(mesh, seams):
    for edge in mesh.edges:
        a, b = edge.vertices
        edge.use_seam = ((a, b) if a < b else (b, a)) in seams


def marked_seams(mesh):
    seams = set()
    for edge in mesh.edges:
        if edge.use_seam:
            a, b = edge.vertices
            seams.add((a, b) if a < b else (b, a))
    return seams


def apply_face_uvs(mesh, uvs, only=None):
    """Write per-face corner uvs into the active layer."""
    layer = mesh.uv_layers.active.data
    if only is None:
        flat = [c for face in uvs for uv in face for c in uv]
        layer.foreach_set("uv", flat)
    else:
        for f in only:
            for li, uv in zip(mesh.polygons[f].loop_indices, uvs[f]):
                layer[li].uv = uv


def preseed_work(obj, angle, marked="NONE", weights=None, auto=False):
    """build_seam_uvs split for a worker thread: compute is bpy-free and safe
    off the main thread, apply(compute()) writes the result back. compute
    returns None when there is nothing to preseed (no hard parts in auto, or
    an empty seam set on a closed mesh), which apply passes through as
    False."""
    mesh = obj.data
    verts = [tuple(v.co) for v in mesh.vertices]
    faces = [tuple(p.vertices) for p in mesh.polygons]
    marks = marked_seams(mesh) if (marked != "NONE" or auto) else frozenset()
    engine = flatten_engine()

    def compute():
        only = None
        if auto:
            only = hard_faces(verts, faces, marks, marked)
            if not only:
                return None
            if len(only) == len(faces):
                only = None
        result = preseed_uvs(engine, verts, faces, angle, marked, weights, only, marks)
        if result is None:
            return None
        seams, uvs = result
        return seams, uvs, only

    def apply(result):
        if result is None:
            return False
        seams, uvs, only = result
        apply_seams(mesh, seams)
        if not mesh.uv_layers:
            mesh.uv_layers.new()
        apply_face_uvs(mesh, uvs, sorted(only) if only is not None else None)
        return True

    return compute, apply


def build_seam_uvs(obj, angle=CREASE_ANGLE, marked="NONE", weights=None, only=None):
    """Seam the strip-merged feature boundaries, then flatten into the uv map.

    marked says what the mesh's own seam marks do. ONLY takes them as the whole
    starting set with no detection, for hand-edited marks after a Mark Seams.
    ADD detects as usual and forces the marks on top, cutting from the
    partition on so no merge pass can dissolve one: a detected seam running
    beside a marked one gives way instead of leaving a ribbon between the two.
    An island the flatten ruins ships as-is: the engine rejects it and recuts
    it with its own search, so hand-placed seams on healthy islands come
    through untouched.

    weights are the painted restrictions, which steer every cut this module
    places, the same paint the engine reads for the cuts it makes itself.

    only restricts everything to those face indices, whole loose parts in
    auto mode: detection and the flatten skip the other parts, whose uvs are
    left exactly as they were.

    The solve runs outside Blender, in seams.preseed against the engine's
    flatten mode. This function only reads the mesh into arrays and writes
    the result back. Returns False when the seam set came out empty on a
    closed mesh, leaving the mesh untouched."""
    mesh = obj.data
    verts = [tuple(v.co) for v in mesh.vertices]
    faces = [tuple(p.vertices) for p in mesh.polygons]
    result = preseed_uvs(
        flatten_engine(),
        verts,
        faces,
        angle,
        marked,
        weights,
        only,
        marked_seams(mesh) if marked != "NONE" else frozenset(),
    )
    if result is None:
        return False
    seams, uvs = result
    apply_seams(mesh, seams)
    if not mesh.uv_layers:
        mesh.uv_layers.new()
    apply_face_uvs(mesh, uvs, sorted(only) if only is not None else None)
    return True
