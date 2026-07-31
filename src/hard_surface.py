import math

import bpy

from .ops.guides import SEAM_RESTRICTIONS_GROUP
from .strips import (
    CREASE_ANGLE,
    face_edges,
    is_hard_surface,
    island_groups,
    island_ruined,
    seam_edges,
    signed_area,
    split_islands,
    vertex_components,
)

# slim iterations for re-unwrapping a folded island alone. 50 flattens the
# long folded strips, but applied globally it breaks other models, so only
# islands that already came out ruined get it (blender's default is 10)
REPAIR_ITERATIONS = 50
# repair rounds: a split piece can come out of its own unwrap still ruined,
# so repair and split repeat until clean, this many times at most. Each round
# halves the stubborn pieces, so this is plenty (pipe_wrench needs 4), and a
# clean model exits on round one
REPAIR_ROUNDS = 6


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


def auto_hard_faces(obj, marked="NONE"):
    """Face indices of the loose parts worth preseeding, for auto mode.

    Each part classifies on its own geometry. A part carrying marked seams
    counts as hard whenever marks are in use: the user placed seams there
    deliberately. With marked ONLY detection never runs, so the marked parts
    are the whole hard set."""
    mesh = obj.data
    verts = [tuple(v.co) for v in mesh.vertices]
    faces = [tuple(p.vertices) for p in mesh.polygons]
    marked_verts = (
        {v for edge in marked_seams(mesh) for v in edge} if marked != "NONE" else set()
    )
    hard = set()
    for comp in vertex_components(faces):
        if marked_verts and marked_verts & {v for fi in comp for v in faces[fi]}:
            hard.update(comp)
        elif marked != "ONLY" and is_hard_surface(verts, [faces[fi] for fi in comp]):
            hard.update(comp)
    return hard


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


def unwrap(obj, only=None, iterations=10):
    """Unwrap the whole mesh, or with `only` just those face indices."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    if only is None:
        bpy.ops.mesh.select_all(action="SELECT")
    else:
        bpy.ops.mesh.select_mode(type="FACE")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")
        for fi in only:
            obj.data.polygons[fi].select = True
        bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.uv.unwrap(method="MINIMUM_STRETCH", margin=0.001, iterations=iterations)
    bpy.ops.object.mode_set(mode="OBJECT")


def uv_density(mesh, only=None):
    """Uv area per unit surface area, the map's texel density."""
    uv = mesh.uv_layers.active.data
    polys = mesh.polygons if only is None else [mesh.polygons[i] for i in only]
    total_uv = total_3d = 0.0
    for poly in polys:
        total_uv += abs(signed_area([tuple(uv[i].uv) for i in poly.loop_indices]))
        total_3d += poly.area
    return total_uv / total_3d if total_3d else 1.0


def island_center(uvs, group):
    points = [uv for fi in group for uv in uvs[fi]]
    xs = [u for u, _ in points]
    ys = [v for _, v in points]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def restore_island(mesh, group, density, center):
    """Put a freshly unwrapped island back at the map's scale and position.

    A selection unwrap packs just the selection into the unit square, so the
    island comes back oversized somewhere else. Scaling it to the map's texel
    density instead of repacking the atlas keeps every measure that reads uv
    lengths against the atlas meaningful, and costs nothing: repacking a whole
    model to move a few islands is most of the repair loop's runtime. Islands
    may overlap until the single pack at the end, which is harmless because
    every test in the loop reads one island at a time."""
    uv = mesh.uv_layers.active.data
    loops = [li for fi in group for li in mesh.polygons[fi].loop_indices]
    area_uv = sum(
        abs(signed_area([tuple(uv[i].uv) for i in mesh.polygons[fi].loop_indices]))
        for fi in group
    )
    area_3d = sum(mesh.polygons[fi].area for fi in group)
    if area_uv <= 0 or area_3d <= 0:
        return
    scale = math.sqrt(density * area_3d / area_uv)
    xs = [uv[li].uv[0] for li in loops]
    ys = [uv[li].uv[1] for li in loops]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    for li in loops:
        u, v = uv[li].uv
        uv[li].uv = (center[0] + (u - cx) * scale, center[1] + (v - cy) * scale)


def normalize_islands(obj, only=None):
    """Restore relative island scale from 3D area and repack the atlas.

    Runs once after the repair loop, not per round. With `only`, just those
    faces normalize and pack, so an auto-mode run leaves the other loose
    parts' uvs alone."""
    tools = bpy.context.scene.tool_settings
    sync = tools.use_uv_select_sync
    tools.use_uv_select_sync = True
    bpy.ops.object.mode_set(mode="EDIT")
    if only is None:
        bpy.ops.mesh.select_all(action="SELECT")
    else:
        bpy.ops.mesh.select_mode(type="FACE")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")
        for fi in only:
            obj.data.polygons[fi].select = True
        bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.uv.average_islands_scale()
    bpy.ops.uv.pack_islands(margin=0.001)
    bpy.ops.object.mode_set(mode="OBJECT")
    tools.use_uv_select_sync = sync


def face_uvs(mesh):
    """Per-face loop uvs from the active layer, in face vertex order."""
    uv = mesh.uv_layers.active.data
    return [[tuple(uv[i].uv) for i in poly.loop_indices] for poly in mesh.polygons]


def build_seam_uvs(obj, angle=CREASE_ANGLE, marked="NONE", weights=None, only=None):
    """Seam the strip-merged feature boundaries, then unwrap into the uv map.

    marked says what the mesh's own seam marks do. ONLY takes them as the whole
    starting set with no detection, for hand-edited marks after a Seams Unwrap.
    ADD detects as usual and forces the marks on top, cutting from the
    partition on so no merge pass can dissolve one: a detected seam running
    beside a marked one gives way instead of leaving a ribbon between the two.
    The repair loop still adds cuts either way, but only inside islands the
    engine would reject, so hand-placed seams on healthy islands come through
    untouched.

    weights are the painted restrictions, which steer every cut this module
    places, the same paint the engine reads for the cuts it makes itself.

    only restricts everything to those face indices, whole loose parts in
    auto mode: detection, the unwrap, the repair loop and the final pack all
    skip the other parts, whose uvs are left exactly as they were.

    The layout matters, not just which edges are split: optcuts keeps the input
    map only when no triangle is flipped, no boundary crosses another and every
    chart is a disk, and it redoes the whole layout with Tutte otherwise.
    strips.py cuts the charts to disks, and MINIMUM_STRETCH (SLIM) is what
    keeps the flattening flip-free, ANGLE_BASED folds narrow charts. A cut-open
    tube still unrolls into one strip as long as its ring and SLIM folds the
    longest of those, so a ruined island is first re-unwrapped alone at more
    iterations, and only islands still ruined after that are cut across and
    unwrapped once more. A split piece can itself come out long and folded,
    so the repair repeats up to REPAIR_ROUNDS times. Clean islands are left
    alone unless they are strips too long to pack, which split for the
    atlas's sake; on a clean compact model nothing here runs at all.
    """
    mesh = obj.data
    verts = [tuple(v.co) for v in mesh.vertices]
    faces = [tuple(p.vertices) for p in mesh.polygons]
    if marked == "ONLY":
        seams = marked_seams(mesh)
    else:
        forced = marked_seams(mesh) if marked == "ADD" else None
        detect = faces if only is None else [faces[i] for i in sorted(only)]
        seams = seam_edges(verts, detect, angle, weights=weights, forced=forced)
        apply_seams(mesh, seams)

    if not obj.data.uv_layers:
        obj.data.uv_layers.new()
    unwrap(obj, sorted(only) if only is not None else None)

    edges = face_edges(faces)
    density = uv_density(mesh, only)
    repaired = False

    def redo_islands(targets, iterations=10):
        """Unwrap these islands again, each landing back at its own scale."""
        uvs = face_uvs(mesh)
        centers = [island_center(uvs, group) for group in targets]
        unwrap(obj, [f for group in targets for f in group], iterations)
        for group, center in zip(targets, centers):
            restore_island(mesh, group, density, center)

    # islands untouched by a round can't have changed, so later rounds only
    # rescan what was redone or cut; None means the first round checks all
    dirty = None
    for _ in range(REPAIR_ROUNDS):
        groups = island_groups(faces, seams, edges)
        if only is not None:
            # loose parts are disjoint, so an island is entirely in or out
            groups = [g for g in groups if g[0] in only]
        if dirty is not None:
            groups = [g for g in groups if dirty & set(g)]
        uvs = face_uvs(mesh)
        ruined = [g for g in groups if island_ruined(g, faces, uvs, edges, seams)]
        if ruined:
            repaired = True
            redo_islands(ruined, REPAIR_ITERATIONS)

        # cuts ruined islands, and clean strips too long to pack
        extra = split_islands(verts, faces, seams, face_uvs(mesh), weights, groups)
        if not extra:
            break
        repaired = True
        seams |= extra
        apply_seams(mesh, seams)
        touched = {f for e in extra for f in edges[e]}
        # the split pieces of one old island go back where it was, together
        redo_islands([g for g in groups if touched & set(g)])
        dirty = touched | {f for g in ruined for f in g}

    if repaired:
        normalize_islands(obj, only)
