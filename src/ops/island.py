import bmesh
import bpy

from ..engines import get_engine
from ..job import AreaUVs, IslandUVs
from ..logger import logger
from ..manager import manager
from ..seams import (
    REPAIR_ITERATIONS,
    face_edges,
    island_layout,
    pair,
    signed_area,
    uv_island_groups,
)
from ..unwrap import Unwrap
from ..utils.io import export_obj
from ..utils.mesh import new_bmesh, set_bmesh, triangulate
from ..utils.paths import get_extension_dir_path, get_preferences


def face_uvs(mesh):
    """Per-face loop uvs from the active layer, in face vertex order."""
    uv = mesh.uv_layers.active.data
    return [[tuple(uv[i].uv) for i in poly.loop_indices] for poly in mesh.polygons]


def unwrap(obj, only, iterations):
    """Blender's minimum stretch unwrap over just these faces. The area fixes
    stay on bpy.ops because they pin uvs, which the engine's flatten mode has
    no channel for."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="FACE")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for fi in only:
        obj.data.polygons[fi].select = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.uv.unwrap(method="MINIMUM_STRETCH", margin=0.001, iterations=iterations)
    bpy.ops.object.mode_set(mode="OBJECT")


def selected_faces(mesh):
    """The faces picked in the uv editor.

    With uv sync off the editor only draws faces that are selected in the 3d
    view, and it leaves stale uv flags on the ones it isn't drawing, so a face
    counts only when the mesh and the uv selection agree."""
    selected = {p.index for p in mesh.polygons if p.select}
    if bpy.context.scene.tool_settings.use_uv_select_sync:
        return selected
    uv_select = mesh.attributes.get(".uv_select_face")
    if uv_select is None:
        return set()
    return {fi for fi in selected if uv_select.data[fi].value}


def queue_fix(obj, job, name, path, vertex_count, props):
    """Queue an exported patch on the manager with the job that puts the
    engine's result back into obj. A fix carries none of the material or
    vertex group state a full unwrap does, the input mesh keeps its own."""
    manager.input[job] = obj
    manager.add(
        Unwrap(
            name=name,
            input_name=name,
            path=path,
            guide_path=None,
            edge_path=None,
            jobs=(None, None, None, None, job),
            origin=obj.matrix_world.translation,
            materials=[],
            added_edges=[],
            vertex_count=vertex_count,
            material_indices=[],
            vertex_groups={},
            shade_smooth=False,
            maintain_mode=props.maintain_mode,
        )
    )


def target_islands(obj):
    """Uv islands under the selected faces, each with its uv bounds and the uv
    area it covers."""
    mesh = obj.data
    if not mesh.uv_layers.active:
        return None, "Mesh has no uv map"
    selected = selected_faces(mesh)
    if not selected:
        return None, "Select the faces of the islands to fix"

    faces = [tuple(p.vertices) for p in mesh.polygons]
    uvs = face_uvs(mesh)
    targets = []
    for group in uv_island_groups(faces, uvs, face_edges(faces)):
        if selected.isdisjoint(group):
            continue
        points = [uv for fi in group for uv in uvs[fi]]
        xs = [u for u, _ in points]
        ys = [v for _, v in points]
        area = sum(abs(signed_area(uvs[fi])) for fi in group)
        targets.append((group, (min(xs), min(ys), max(xs), max(ys)), area))
    return targets, None


def queue_island(obj, group, bbox, area, k, input_path, props):
    """Export one island as its own mesh and queue it on the manager with a
    IslandUVs job that will put the result back in place."""
    mesh = obj.data
    used = sorted({v for fi in group for v in mesh.polygons[fi].vertices})
    local = {v: i for i, v in enumerate(used)}
    island_mesh = bpy.data.meshes.new("uvgami_island")
    island_mesh.from_pydata(
        [mesh.vertices[v].co.copy() for v in used],
        [],
        [[local[v] for v in mesh.polygons[fi].vertices] for fi in group],
    )
    temp = bpy.data.objects.new("uvgami_island", island_mesh)
    bpy.context.scene.collection.objects.link(temp)
    temp.matrix_world = obj.matrix_world.copy()

    bm = new_bmesh(temp)
    if any(len(f.verts) > 3 for f in bm.faces):
        triangulate(bm)
        set_bmesh(bm, temp)
    else:
        bm.free()

    name = f"{obj.name}_island_{k}"
    path = input_path / f"{bpy.path.clean_name(name)}.obj"
    while path.is_file():
        path = path.parent / f"{path.stem}1.obj"
    export_obj(temp, path, False)
    vertex_count = len(temp.data.vertices)
    bpy.data.objects.remove(temp, do_unlink=True)
    bpy.data.meshes.remove(island_mesh)

    queue_fix(obj, IslandUVs(list(group), bbox, area), name, path, vertex_count, props)


def islands_connected(mesh, targets):
    """True when the islands form one connected set through shared mesh edges,
    the only places the engine can weld."""
    parent = list(range(len(targets)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    edge_owner = {}
    for i, (group, _, _) in enumerate(targets):
        for fi in group:
            for key in mesh.polygons[fi].edge_keys:
                j = edge_owner.setdefault(key, i)
                if j != i:
                    parent[find(i)] = find(j)
    return len({find(i) for i in range(len(targets))}) == 1


def queue_combine(obj, targets, input_path, props):
    """Export the selected islands as one mesh with their uvs and a _stitch
    sidecar, and queue it with an IslandUVs job over all their faces so the
    merged result comes back into the islands' combined uv bounds."""
    mesh = obj.data
    layer = mesh.uv_layers.active
    faces = [fi for group, _, _ in targets for fi in group]
    used = sorted({v for fi in faces for v in mesh.polygons[fi].vertices})
    local = {v: i for i, v in enumerate(used)}
    combine_mesh = bpy.data.meshes.new("uvgami_combine")
    combine_mesh.from_pydata(
        [mesh.vertices[v].co.copy() for v in used],
        [],
        [[local[v] for v in mesh.polygons[fi].vertices] for fi in faces],
    )
    uv = combine_mesh.uv_layers.new()
    # optcuts treats a mirrored island as inverted and overlapping islands as
    # self-intersecting, and stitch mode forbids both, so export a mirrored-
    # back side-by-side layout. the result comes back into the combined bbox
    areas = []
    for group, _, _ in targets:
        total = 0.0
        for fi in group:
            poly = mesh.polygons[fi]
            pts = [
                tuple(layer.uv[poly.loop_start + c].vector)
                for c in range(poly.loop_total)
            ]
            total += signed_area(pts)
        areas.append(total)
    layout = island_layout([b for _, b, _ in targets], areas)
    transform = {fi: t for (group, _, _), t in zip(targets, layout) for fi in group}
    loop = 0
    for fi in faces:
        poly = mesh.polygons[fi]
        flip, du, dv = transform[fi]
        for c in range(poly.loop_total):
            u, v = layer.uv[poly.loop_start + c].vector
            if flip is not None:
                u = flip - u
            uv.uv[loop].vector = (u + du, v + dv)
            loop += 1
    temp = bpy.data.objects.new("uvgami_combine", combine_mesh)
    bpy.context.scene.collection.objects.link(temp)
    temp.matrix_world = obj.matrix_world.copy()

    bm = new_bmesh(temp)
    if any(len(f.verts) > 3 for f in bm.faces):
        triangulate(bm)
        set_bmesh(bm, temp)
    else:
        bm.free()

    name = f"{obj.name}_combine"
    path = input_path / f"{bpy.path.clean_name(name)}.obj"
    while path.is_file():
        path = path.parent / f"{path.stem}1.obj"
    export_obj(temp, path, True)
    (path.parent / f"{path.stem}_stitch").touch()
    vertex_count = len(temp.data.vertices)
    bpy.data.objects.remove(temp, do_unlink=True)
    bpy.data.meshes.remove(combine_mesh)

    bbox = (
        min(b[0] for _, b, _ in targets),
        min(b[1] for _, b, _ in targets),
        max(b[2] for _, b, _ in targets),
        max(b[3] for _, b, _ in targets),
    )
    area = sum(a for _, _, a in targets)
    queue_fix(obj, IslandUVs(faces, bbox, area), name, path, vertex_count, props)


def target_areas(obj, rings):
    """Selected areas per island, grown by face rings inside the island, each
    with the border verts that must stay pinned. A disconnected selection
    gives one area per connected piece."""
    mesh = obj.data
    if not mesh.uv_layers.active:
        return None, 0, 0, "Mesh has no uv map"
    selected = selected_faces(mesh)
    if not selected:
        return None, 0, 0, "Select the faces of the area to fix"

    faces = [tuple(p.vertices) for p in mesh.polygons]
    edges = face_edges(faces)
    targets = []
    whole = ring_skipped = 0
    for group in uv_island_groups(faces, face_uvs(mesh), edges):
        group_set = set(group)
        patch = selected & group_set
        if not patch:
            continue
        # grow so the fix blends out instead of stopping at the selection edge
        for _ in range(rings):
            ring_verts = {v for fi in patch for v in faces[fi]}
            grown = {
                fi for fi in group_set - patch if ring_verts.intersection(faces[fi])
            }
            if not grown:
                break
            patch |= grown
        patch |= enclosed_faces(faces, edges, group_set, patch)
        if patch == group_set:
            # a fully covered island has no border to hold
            whole += 1
            continue
        for comp in face_components(faces, edges, patch):
            comp_verts = {v for fi in comp for v in faces[fi]}
            comp_edges = {
                pair(faces[fi][i], faces[fi][(i + 1) % len(faces[fi])])
                for fi in comp
                for i in range(len(faces[fi]))
            }
            if len(comp_verts) - len(comp_edges) + len(comp) != 1:
                # the area rings a real hole, no selection can make it a
                # disk, only breaking the ring can
                ring_skipped += 1
                continue
            outside = group_set - comp
            border = comp_verts & {v for fi in outside for v in faces[fi]}
            targets.append((sorted(comp), border))
    return targets, whole, ring_skipped, None


def face_components(faces, edges, patch):
    """The patch split into edge-connected pieces."""
    unvisited = set(patch)
    components = []
    while unvisited:
        seed = unvisited.pop()
        component = {seed}
        stack = [seed]
        while stack:
            fi = stack.pop()
            face = faces[fi]
            for i in range(len(face)):
                for nb in edges[pair(face[i], face[(i + 1) % len(face)])]:
                    if nb in unvisited:
                        unvisited.discard(nb)
                        component.add(nb)
                        stack.append(nb)
        components.append(component)
    return components


def enclosed_faces(faces, edges, group_set, patch):
    """Island faces the patch encircles. A ring selection strands them, and
    the engine can only keep a disk, so they must join the patch."""
    boundary = {
        e
        for e, owners in edges.items()
        if sum(1 for fi in owners if fi in group_set) == 1
    }
    unvisited = group_set - patch
    enclosed = set()
    while unvisited:
        seed = unvisited.pop()
        component = {seed}
        stack = [seed]
        touches_boundary = False
        while stack:
            fi = stack.pop()
            face = faces[fi]
            for i in range(len(face)):
                e = pair(face[i], face[(i + 1) % len(face)])
                if e in boundary:
                    touches_boundary = True
                for nb in edges[e]:
                    if nb in unvisited:
                        unvisited.discard(nb)
                        component.add(nb)
                        stack.append(nb)
        if not touches_boundary:
            enclosed |= component
    return enclosed


def has_flipped(mesh, patch):
    """Any patch face with a backwards or degenerate uv corner. Checked per
    fan triangle: a twisted quad hides a flipped triangle behind a positive
    polygon area. Nonconvex faces can read as flipped, which only costs an
    unneeded repair pass."""
    layer = mesh.uv_layers.active
    for fi in patch:
        poly = mesh.polygons[fi]
        pts = [tuple(layer.uv[li].vector) for li in poly.loop_indices]
        for i in range(1, len(pts) - 1):
            if signed_area([pts[0], pts[i], pts[i + 1]]) <= 0:
                return True
    return False


def spans_own_seam(mesh, patch):
    """A patch vert carrying two different uvs: the island wraps around and
    borders its own cut edge here. queue_area's export mirrors vt indices to
    v indices, one uv per vert, so such a patch can't be represented and
    would weld the seam's two sides."""
    layer = mesh.uv_layers.active
    uv_of = {}
    for fi in patch:
        poly = mesh.polygons[fi]
        for c, v in enumerate(poly.vertices):
            uv = tuple(layer.uv[poly.loop_start + c].vector)
            if uv_of.setdefault(v, uv) != uv:
                return True
    return False


def repair_flipped(obj, patch, border):
    """Blender's minimum stretch unwrap over the patch with the border
    pinned, turning a flipped area into a valid map the engine can keep."""
    bm = new_bmesh(obj)
    uvl = bm.loops.layers.uv.active
    bm.faces.ensure_lookup_table()
    pinned = []
    for fi in patch:
        for c, loop in enumerate(bm.faces[fi].loops):
            if loop.vert.index in border and not loop[uvl].pin_uv:
                loop[uvl].pin_uv = True
                pinned.append((fi, c))
    set_bmesh(bm, obj)

    unwrap(obj, patch, REPAIR_ITERATIONS)

    bm = new_bmesh(obj)
    uvl = bm.loops.layers.uv.active
    bm.faces.ensure_lookup_table()
    for fi, c in pinned:
        bm.faces[fi].loops[c][uvl].pin_uv = False
    set_bmesh(bm, obj)


def queue_area(obj, patch, border, k, input_path, props, nocut, snapshot):
    """Export one patch with its uvs and pinned border and queue it on the
    manager with an AreaUVs job that will put the result back in place.

    The obj is written by hand: inside one island every vert has one uv, so
    vt indices can mirror v indices exactly, which is what makes the pinned
    indices in the _fixed sidecar and the engine's vertex indices line up."""
    mesh = obj.data
    layer = mesh.uv_layers.active
    used = sorted({v for fi in patch for v in mesh.polygons[fi].vertices})
    local = {v: i for i, v in enumerate(used)}

    uv_of = {}
    pins = []
    for fi in patch:
        poly = mesh.polygons[fi]
        for c, v in enumerate(poly.vertices):
            uv = tuple(layer.uv[poly.loop_start + c].vector)
            uv_of[v] = uv
            if v in border:
                pins.append((fi, c, uv))

    area_mesh = bpy.data.meshes.new("uvgami_area")
    area_mesh.from_pydata(
        [mesh.vertices[v].co.copy() for v in used],
        [],
        [[local[v] for v in mesh.polygons[fi].vertices] for fi in patch],
    )
    bm = bmesh.new()
    bm.from_mesh(area_mesh)
    if any(len(f.verts) > 3 for f in bm.faces):
        triangulate(bm)
        bm.to_mesh(area_mesh)
    bm.free()

    name = f"{obj.name}_area_{k}"
    path = input_path / f"{bpy.path.clean_name(name)}.obj"
    while path.is_file():
        path = path.parent / f"{path.stem}1.obj"

    matrix = obj.matrix_world
    with path.open("w") as f:
        for v in area_mesh.vertices:
            x, y, z = matrix @ v.co
            f.write(f"v {x} {y} {z}\n")
        for v in area_mesh.vertices:
            u, w = uv_of[used[v.index]]
            f.write(f"vt {u} {w}\n")
        for poly in area_mesh.polygons:
            corners = " ".join(f"{v + 1}/{v + 1}" for v in poly.vertices)
            f.write(f"f {corners}\n")
    with (path.parent / f"{path.stem}_fixed").open("w") as f:
        f.write(",".join(str(local[v]) for v in sorted(border)))
        if nocut:
            f.write("\nnocut")

    vertex_count = len(area_mesh.vertices)
    bpy.data.meshes.remove(area_mesh)

    queue_fix(obj, AreaUVs(patch, pins, snapshot), name, path, vertex_count, props)


def validate_engine(op):
    """The engine for a fix run, or None with the error already reported.
    Optcuts is the only engine that can pin a border or stitch islands, so
    these operators ignore the engine chosen in the 3d panel."""
    engine = get_engine("OPTCUTS")
    if manager.is_active and manager.engine is not engine:
        op.report(
            {"ERROR"},
            "Finish or cancel the current unwrap first",
        )
        return None, None
    engine_ctx, error = engine.validate(get_preferences())
    if error is not None:
        op.report({"ERROR"}, error)
        return None, None
    return engine, engine_ctx


def queue_targets(engine, engine_ctx, count, queue_one):
    """Prepare the io folders, queue each target, and start or extend the
    manager session."""
    input_path = get_extension_dir_path() / "input"
    input_path.mkdir(exist_ok=True)
    output_path = input_path.parent / "output"
    output_path.mkdir(exist_ok=True)
    if not manager.is_active:
        for file in input_path.iterdir():
            file.unlink()
        for file in output_path.iterdir():
            file.unlink()

    queued = len(manager._queue)
    try:
        for k in range(count):
            queue_one(k, input_path)
    except Exception:
        # drop the partial batch or it silently runs with the next session
        while len(manager._queue) > queued:
            manager._queue.pop()
        raise

    if not manager.is_active:
        logger.new_info()
        manager.engine = engine
        manager.engine_ctx = engine_ctx
        # these operators are run from the uv editor, so the bar belongs there
        manager.start(uv_editor=True)
    else:
        manager.starting_count += count


class UVGAMI_OT_unwrap_island(bpy.types.Operator):
    bl_idname = "uvgami.unwrap_island"
    bl_label = "Unwrap Island"
    bl_description = "Re-unwrap the island under the selected face(s)"
    bl_options = {"UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH"

    def execute(self, context):
        props = context.scene.uvgami
        engine, engine_ctx = validate_engine(self)
        if engine is None:
            return {"CANCELLED"}

        obj = context.view_layer.objects.active
        bpy.ops.object.mode_set(mode="OBJECT")
        try:
            targets, error = target_islands(obj)
            if error:
                self.report({"ERROR"}, error)
                return {"CANCELLED"}

            def queue_one(k, input_path):
                group, bbox, area = targets[k]
                queue_island(obj, group, bbox, area, k + 1, input_path, props)

            queue_targets(engine, engine_ctx, len(targets), queue_one)
        finally:
            obj.select_set(True)
            context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode="EDIT")

        self.report({"INFO"}, f"Unwrapping {len(targets)} island(s)")
        return {"FINISHED"}


class UVGAMI_OT_combine_islands(bpy.types.Operator):
    bl_idname = "uvgami.combine_islands"
    bl_label = "Combine Islands"
    bl_description = (
        "Select a face on two islands to merge them. The islands have to share a seam"
    )
    bl_options = {"UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH"

    def execute(self, context):
        props = context.scene.uvgami
        engine, engine_ctx = validate_engine(self)
        if engine is None:
            return {"CANCELLED"}

        obj = context.view_layer.objects.active
        bpy.ops.object.mode_set(mode="OBJECT")
        try:
            targets, error = target_islands(obj)
            if error:
                self.report({"ERROR"}, error)
                return {"CANCELLED"}
            if len(targets) < 2:
                self.report({"ERROR"}, "Select faces on at least two islands")
                return {"CANCELLED"}
            if not islands_connected(obj.data, targets):
                self.report(
                    {"ERROR"}, "The selected islands don't all share mesh edges"
                )
                return {"CANCELLED"}

            def queue_one(k, input_path):
                queue_combine(obj, targets, input_path, props)

            queue_targets(engine, engine_ctx, 1, queue_one)
        finally:
            obj.select_set(True)
            context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode="EDIT")

        self.report({"INFO"}, f"Combining {len(targets)} islands")
        return {"FINISHED"}


class AreaOperator:
    """Shared body of Recut Area and Relax Area. A plain mixin: registering a
    subclass of a registered operator unregisters the parent."""

    bl_options = {"UNDO"}
    nocut = False

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH"

    def execute(self, context):
        props = context.scene.uvgami
        engine, engine_ctx = validate_engine(self)
        if engine is None:
            return {"CANCELLED"}

        obj = context.view_layer.objects.active
        bpy.ops.object.mode_set(mode="OBJECT")
        try:
            targets, whole, rings, error = target_areas(obj, props.area_expand)
            if error:
                self.report({"ERROR"}, error)
                return {"CANCELLED"}
            seam_skipped = 0
            for target in list(targets):
                if spans_own_seam(obj.data, target[0]):
                    targets.remove(target)
                    seam_skipped += 1
            if not targets:
                if whole:
                    error = "The whole island is selected, use Unwrap Island instead"
                elif seam_skipped:
                    error = (
                        "The area spans the island's own seam,"
                        " deselect the faces on one side"
                    )
                else:
                    error = "The area rings a hole, deselect a face to break the ring"
                self.report({"ERROR"}, error)
                return {"CANCELLED"}

            # snapshot each patch so a failed run can put the map back, the
            # flipped pre-repair below changes it before the engine even runs
            layer = obj.data.uv_layers.active
            snapshots = []
            for patch, border in targets:
                snapshot = []
                for fi in patch:
                    poly = obj.data.polygons[fi]
                    for c in range(poly.loop_total):
                        snapshot.append(
                            (fi, c, tuple(layer.uv[poly.loop_start + c].vector))
                        )
                snapshots.append(snapshot)
                # a flipped area would make the engine reject the map,
                # pre-repair it with Blender's pinned unwrap so the border
                # still holds
                if has_flipped(obj.data, patch):
                    repair_flipped(obj, patch, border)

            def queue_one(k, input_path):
                patch, border = targets[k]
                queue_area(
                    obj,
                    patch,
                    border,
                    k + 1,
                    input_path,
                    props,
                    self.nocut,
                    snapshots[k],
                )

            queue_targets(engine, engine_ctx, len(targets), queue_one)
        finally:
            obj.select_set(True)
            context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode="EDIT")

        notes = []
        if whole:
            notes.append(f"{whole} whole island(s) skipped")
        if rings:
            notes.append(f"{rings} area(s) around holes skipped")
        if seam_skipped:
            notes.append(f"{seam_skipped} area(s) spanning their seam skipped")
        skipped = f", {', '.join(notes)}" if notes else ""
        self.report({"INFO"}, f"Fixing {len(targets)} area(s){skipped}")
        return {"FINISHED"}


class UVGAMI_OT_recut_area(AreaOperator, bpy.types.Operator):
    bl_idname = "uvgami.recut_area"
    bl_label = "Recut Area"
    bl_description = "Re-unwrap the selected faces with cuts if necessary"


class UVGAMI_OT_relax_area(AreaOperator, bpy.types.Operator):
    bl_idname = "uvgami.relax_area"
    bl_label = "Relax Area"
    bl_description = "Relax the selected faces to reduce stretching"
    nocut = True
