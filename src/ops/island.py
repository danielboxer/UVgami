from collections import namedtuple

import bmesh
import bpy

from ..engines import ENGINES
from ..job import AreaUVs, IslandUVs
from ..logger import logger
from ..manager import manager
from ..no_bpy import (
    EngineError,
    faces_by_edge,
    fan_areas,
    find_root,
    islands_from_uvs,
    signed_area,
    sorted_pair,
)
from ..proxy import ProxyIslandUVs, make_proxy, triangle_count
from ..ui.props import describe_settings, fix_settings
from ..unwrap import ExportData, Unwrap
from ..utils.io import export_obj
from ..utils.mesh import (
    UV_ROUNDING_DIGITS,
    face_uvs,
    face_vertices,
    new_bmesh,
    set_bmesh,
    triangulate_mesh,
)
from ..utils.paths import unused_input_path

IslandTarget = namedtuple("IslandTarget", "faces bbox area")
AreaTargets = namedtuple("AreaTargets", "targets whole_islands hole_rings")


class FixError(Exception):
    pass


# blender's default of 10 leaves folds
REPAIR_ITERATIONS = 50


def minimum_stretch_unwrap(temp, obj):
    # multi-object edit pulls a selected input mesh in and clears its faces
    obj.select_set(False)
    bpy.context.view_layer.objects.active = temp
    temp.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="FACE")
    bpy.ops.mesh.select_all(action="SELECT")
    # blender's unwrap keeps pinned uvs in place, optcuts' flatten mode has no pins
    bpy.ops.uv.unwrap(
        method="MINIMUM_STRETCH", margin=0.001, iterations=REPAIR_ITERATIONS
    )
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(True)


def selected_faces(mesh):
    selected = {polygon.index for polygon in mesh.polygons if polygon.select}
    if bpy.context.scene.tool_settings.use_uv_select_sync:
        return selected
    # with uv sync off, faces the uv editor hides can have stale uv flags
    uv_select = mesh.attributes.get(".uv_select_face")
    if uv_select is None:
        return set()
    return {face for face in selected if uv_select.data[face].value}


def queue_fix(obj, job, name, path, vertex_count):
    unwrap = Unwrap(
        name=name,
        input_name=name,
        path=path,
        transfer_uvs_job=job,
    )
    unwrap.export = ExportData(
        origin=obj.matrix_world.translation.copy(), vertex_count=vertex_count
    )
    manager.add(unwrap)


def target_islands(obj):
    mesh = obj.data
    if not mesh.uv_layers.active:
        raise FixError("Mesh has no uv map")
    selected = selected_faces(mesh)
    if not selected:
        raise FixError("Select the faces of the islands to fix")

    faces = face_vertices(mesh)
    uvs = face_uvs(mesh)
    targets = []
    for group in islands_from_uvs(faces, uvs, faces_by_edge(faces)):
        if selected.isdisjoint(group):
            continue
        points = [uv for face in group for uv in uvs[face]]
        xs = [u for u, _ in points]
        ys = [v for _, v in points]
        area = sum(abs(signed_area(uvs[face])) for face in group)
        targets.append(IslandTarget(group, (min(xs), min(ys), max(xs), max(ys)), area))
    return targets


def face_subset_object(obj, faces, name):
    mesh = obj.data
    used = sorted({vertex for face in faces for vertex in mesh.polygons[face].vertices})
    local = {vertex: index for index, vertex in enumerate(used)}
    subset_mesh = bpy.data.meshes.new(name)
    subset_mesh.from_pydata(
        [mesh.vertices[vertex].co.copy() for vertex in used],
        [],
        [[local[vertex] for vertex in mesh.polygons[face].vertices] for face in faces],
    )
    temp = bpy.data.objects.new(name, subset_mesh)
    bpy.context.scene.collection.objects.link(temp)
    temp.matrix_world = obj.matrix_world.copy()
    return temp, used


def remove_temp(temp):
    mesh = temp.data
    bpy.data.objects.remove(temp, do_unlink=True)
    bpy.data.meshes.remove(mesh)


def write_fixed_sidecar(path, pinned, *, nocut):
    # _fixed files have the pin list on the first line
    with (path.parent / f"{path.stem}_fixed").open("w") as file:
        file.write(",".join(str(vertex) for vertex in pinned))
        if nocut:
            # nocut leaves the cuts as they are and only reduces stretch
            file.write("\nnocut")


def queue_island(obj, group, bbox, area, number, input_path, props):
    temp, _ = face_subset_object(obj, group, "uvgami_island")
    triangulate_mesh(temp.data)

    proxied = props.use_proxy and make_proxy(temp, props.proxy_faces)

    name = f"{obj.name}_island_{number}"
    path = unused_input_path(input_path, name, manager.claimed_paths())
    export_obj(temp, path, export_uv=False)
    vertex_count = len(temp.data.vertices)
    remove_temp(temp)

    job = (
        ProxyIslandUVs(obj, list(group), bbox, area)
        if proxied
        else IslandUVs(obj, list(group), bbox, area)
    )
    queue_fix(obj, job, name, path, vertex_count)


def edge_splits_uv(edge, uv_layer):
    uv_of = {}
    for loop in edge.link_loops:
        for corner in (loop, loop.link_loop_next):
            uv = tuple(
                round(value, UV_ROUNDING_DIGITS) for value in corner[uv_layer].uv
            )
            if uv_of.setdefault(corner.vert.index, uv) != uv:
                return True
    return False


def folded_faces(bm, uv_layer):
    total = 0.0
    fans = []
    for face in bm.faces:
        areas = fan_areas([loop[uv_layer].uv for loop in face.loops])
        total += sum(areas)
        fans.append((face, areas))
    orientation = 1 if total >= 0 else -1
    return [
        face for face, areas in fans if any(area * orientation < 0 for area in areas)
    ]


def repair_flipped_island(obj, temp):
    bm = new_bmesh(temp)
    uv_layer = bm.loops.layers.uv.active

    flipped = folded_faces(bm, uv_layer)
    if not flipped:
        bm.free()
        return

    for edge in bm.edges:
        edge.seam = edge_splits_uv(edge, uv_layer)

    free = {vertex for face in flipped for vertex in face.verts}
    for _ in range(2):
        grown = [face for face in bm.faces if not free.isdisjoint(face.verts)]
        free |= {vertex for face in grown for vertex in face.verts}
    for face in bm.faces:
        for loop in face.loops:
            loop[uv_layer].pin_uv = loop.vert not in free
    set_bmesh(bm, temp)

    minimum_stretch_unwrap(temp, obj)


def island_copy(obj, group):
    mesh = obj.data
    layer = mesh.uv_layers.active
    temp, _ = face_subset_object(obj, group, "uvgami_island")
    island_layer = temp.data.uv_layers.new()
    corner = 0
    for face in group:
        polygon = mesh.polygons[face]
        for offset in range(polygon.loop_total):
            island_layer.uv[corner].vector = layer.uv[
                polygon.loop_start + offset
            ].vector
            corner += 1
    return temp


def queue_nocut(obj, temp, group, bbox, area, number, input_path):
    layer = temp.data.uv_layers.active
    total = 0.0
    for polygon in temp.data.polygons:
        points = [
            tuple(layer.uv[polygon.loop_start + offset].vector)
            for offset in range(polygon.loop_total)
        ]
        total += signed_area(points)
    mirrored = total < 0
    if mirrored:
        for corner in layer.uv:
            u, w = corner.vector
            corner.vector = (-u, w)

    triangulate_mesh(temp.data)

    repair_flipped_island(obj, temp)

    name = f"{obj.name}_island_{number}"
    path = unused_input_path(input_path, name, manager.claimed_paths())
    export_obj(temp, path, export_uv=True)
    write_fixed_sidecar(path, [], nocut=True)
    vertex_count = len(temp.data.vertices)
    remove_temp(temp)

    job = IslandUVs(obj, list(group), bbox, area, mirrored=mirrored)
    queue_fix(obj, job, name, path, vertex_count)


def queue_relax(obj, group, bbox, area, number, input_path, props):
    queue_nocut(obj, island_copy(obj, group), group, bbox, area, number, input_path)


def euler_characteristic(temp, cut_edges):
    bm = new_bmesh(temp)
    bm.edges.ensure_lookup_table()
    bmesh.ops.split_edges(bm, edges=[bm.edges[index] for index in cut_edges])
    characteristic = len(bm.verts) - len(bm.edges) + len(bm.faces)
    bm.free()
    return characteristic


def weld_shared_seams(temp, island_of, island_count):
    bm = new_bmesh(temp)
    uv_layer = bm.loops.layers.uv.active
    splits = []
    seams = []
    for edge in bm.edges:
        if not edge_splits_uv(edge, uv_layer):
            continue
        splits.append(edge.index)
        shared = len({island_of[face.index] for face in edge.link_faces}) > 1
        edge.seam = not shared
        if not shared:
            seams.append(edge.index)
    set_bmesh(bm, temp)

    expected = euler_characteristic(temp, splits) - (island_count - 1)
    if euler_characteristic(temp, seams) < expected:
        raise FixError(
            "The islands touch along more than one seam,"
            " so they can't be one flat island"
        )


def islands_connected(mesh, targets):
    parent = list(range(len(targets)))
    edge_owner = {}
    for i, target in enumerate(targets):
        for face in target.faces:
            for key in mesh.polygons[face].edge_keys:
                j = edge_owner.setdefault(key, i)
                if j != i:
                    parent[find_root(parent, i)] = find_root(parent, j)
    return len({find_root(parent, i) for i in range(len(targets))}) == 1


def target_areas(obj, rings):
    mesh = obj.data
    if not mesh.uv_layers.active:
        raise FixError("Mesh has no uv map")
    selected = selected_faces(mesh)
    if not selected:
        raise FixError("Select the faces of the area to fix")

    faces = face_vertices(mesh)
    uvs = face_uvs(mesh)
    edges = faces_by_edge(faces)
    targets = []
    whole = ring_skipped = 0
    for group in islands_from_uvs(faces, uvs, edges):
        group_set = set(group)
        island_selected = selected & group_set
        if not island_selected:
            continue
        taken = set(island_selected)
        emitted = set()
        absorbed = set()
        for seed in sorted(
            face_components(faces, uvs, edges, island_selected), key=min
        ):
            patch = set(seed) - absorbed
            if not patch:
                continue
            uv_of = {
                faces[face][i]: uvs[face][i]
                for face in patch
                for i in range(len(faces[face]))
            }
            for _ in range(rings):
                ring = set(uv_of.items())
                grew = False
                for face in sorted(group_set - taken):
                    corners = list(zip(faces[face], uvs[face]))
                    if ring.isdisjoint(corners):
                        continue
                    if any(uv_of.get(vertex, uv) != uv for vertex, uv in corners):
                        continue
                    patch.add(face)
                    taken.add(face)
                    uv_of.update(corners)
                    grew = True
                if not grew:
                    break
            enclosed = enclosed_faces(faces, edges, group_set, patch) - emitted
            patch |= enclosed
            taken |= enclosed
            absorbed |= enclosed
            if patch == group_set:
                whole += 1
                continue
            emitted |= patch
            for component in face_components(faces, uvs, edges, patch):
                component_vertices = {
                    vertex for face in component for vertex in faces[face]
                }
                component_edges = {
                    sorted_pair(faces[face][i], faces[face][(i + 1) % len(faces[face])])
                    for face in component
                    for i in range(len(faces[face]))
                }
                euler = len(component_vertices) - len(component_edges) + len(component)
                if euler != 1:
                    ring_skipped += 1
                    continue
                outside = group_set - component
                border = component_vertices & {
                    vertex for face in outside for vertex in faces[face]
                }
                targets.append((sorted(component), border))
    return AreaTargets(targets, whole, ring_skipped)


def face_components(faces, uvs, edges, patch):
    def corner_uv(face, vertex):
        return uvs[face][faces[face].index(vertex)]

    unvisited = set(patch)
    components = []
    while unvisited:
        seed = unvisited.pop()
        component = {seed}
        stack = [seed]
        while stack:
            face = stack.pop()
            corners = faces[face]
            for i in range(len(corners)):
                tail, head = corners[i], corners[(i + 1) % len(corners)]
                for neighbor in edges[sorted_pair(tail, head)]:
                    if (
                        neighbor in unvisited
                        and corner_uv(neighbor, tail) == corner_uv(face, tail)
                        and corner_uv(neighbor, head) == corner_uv(face, head)
                    ):
                        unvisited.discard(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
        components.append(component)
    return components


def enclosed_faces(faces, edges, group_set, patch):
    boundary = {
        edge
        for edge, owners in edges.items()
        if sum(1 for face in owners if face in group_set) == 1
    }
    unvisited = group_set - patch
    stranded = []
    while unvisited:
        seed = unvisited.pop()
        component = {seed}
        stack = [seed]
        touches_boundary = False
        while stack:
            face = stack.pop()
            corners = faces[face]
            for i in range(len(corners)):
                edge = sorted_pair(corners[i], corners[(i + 1) % len(corners)])
                if edge in boundary:
                    touches_boundary = True
                for neighbor in edges[edge]:
                    if neighbor in unvisited:
                        unvisited.discard(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
        # optcuts input must be a disk
        if not touches_boundary:
            stranded.append(component)
    if not boundary and stranded:
        stranded.remove(max(stranded, key=len))
    enclosed = set()
    for component in stranded:
        enclosed |= component
    return enclosed


def spans_own_seam(mesh, patch):
    layer = mesh.uv_layers.active
    uv_of = {}
    for face in patch:
        polygon = mesh.polygons[face]
        for offset, vertex in enumerate(polygon.vertices):
            vector = layer.uv[polygon.loop_start + offset].vector
            uv = tuple(round(value, UV_ROUNDING_DIGITS) for value in vector)
            if uv_of.setdefault(vertex, uv) != uv:
                return True
    return False


def repair_flipped(obj, temp, used, border):
    bm = new_bmesh(temp)
    uv_layer = bm.loops.layers.uv.active
    for face in bm.faces:
        for loop in face.loops:
            loop[uv_layer].pin_uv = used[loop.vert.index] in border
    set_bmesh(bm, temp)

    minimum_stretch_unwrap(temp, obj)


def queue_area(obj, patch, border, number, input_path, *, nocut):
    mesh = obj.data
    layer = mesh.uv_layers.active

    uv_of = {}
    pins = []
    for face in patch:
        polygon = mesh.polygons[face]
        for offset, vertex in enumerate(polygon.vertices):
            uv = tuple(layer.uv[polygon.loop_start + offset].vector)
            uv_of[vertex] = uv
            if vertex in border:
                pins.append((face, offset, uv))

    face_uv_lists = [
        [uv_of[vertex] for vertex in mesh.polygons[face].vertices] for face in patch
    ]
    mirrored = sum(signed_area(points) for points in face_uv_lists) < 0
    if mirrored:
        face_uv_lists = [[(-u, w) for u, w in points] for points in face_uv_lists]

    temp, used = face_subset_object(obj, patch, "uvgami_area")
    corner_uvs = [value for points in face_uv_lists for uv in points for value in uv]
    temp.data.uv_layers.new().data.foreach_set("uv", corner_uvs)

    # optcuts fails on a map with a flipped triangle
    if any(area <= 0 for points in face_uv_lists for area in fan_areas(points)):
        repair_flipped(obj, temp, used, border)

    triangulate_mesh(temp.data)

    name = f"{obj.name}_area_{number}"
    path = unused_input_path(input_path, name, manager.claimed_paths())
    export_obj(temp, path, export_uv=True)
    local = {vertex: index for index, vertex in enumerate(used)}
    write_fixed_sidecar(path, [local[vertex] for vertex in sorted(border)], nocut=nocut)
    vertex_count = len(temp.data.vertices)
    remove_temp(temp)

    job = AreaUVs(obj, patch, pins, mirrored=mirrored)
    queue_fix(obj, job, name, path, vertex_count)


def optcuts_for_fixes():
    # optcuts is the only engine that can pin a border or stitch islands
    engine = ENGINES["OPTCUTS"]
    if manager.is_active and manager.engine is not engine:
        raise FixError("Finish or cancel the current unwrap first")
    return engine, engine.validate()


def queue_targets(engine, engine_ctx, obj, count, queue_one):
    input_path = manager.prepare_input_dir()

    queued_before = set(manager.active)
    try:
        for index in range(count):
            queue_one(index, input_path)
    except Exception:
        for unwrap in manager.active:
            if unwrap not in queued_before:
                manager.remove_unwrap(unwrap)
        raise

    if not manager.is_active:
        props = bpy.context.scene.uvgami
        info = logger.new_info()
        info.engine = engine.describe()
        info.objects = [(obj.name, triangle_count(obj))]
        info.settings = describe_settings(props, fix_settings(props))
        manager.start(engine, engine_ctx, uv_editor=True)


class FixOperatorMixin:
    # registering a subclass of a registered operator unregisters the parent
    bl_options = {"UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH"

    def execute(self, context):
        try:
            engine, engine_ctx = optcuts_for_fixes()
            obj = context.view_layer.objects.active
            bpy.ops.object.mode_set(mode="OBJECT")
            try:
                return self.queue_selection(
                    obj, context.scene.uvgami, engine, engine_ctx
                )
            finally:
                obj.select_set(True)
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode="EDIT")
        except (FixError, EngineError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}


class IslandOperator(FixOperatorMixin):
    queue_target = None
    verb = ""

    def queue_selection(self, obj, props, engine, engine_ctx):
        targets = target_islands(obj)

        def queue_one(index, input_path):
            target = targets[index]
            self.queue_target(
                obj,
                target.faces,
                target.bbox,
                target.area,
                index + 1,
                input_path,
                props,
            )

        queue_targets(engine, engine_ctx, obj, len(targets), queue_one)
        self.report({"INFO"}, f"{self.verb} {len(targets)} island(s)")
        return {"FINISHED"}


class UVGAMI_OT_unwrap_island(IslandOperator, bpy.types.Operator):
    bl_idname = "uvgami.unwrap_island"
    bl_label = "Unwrap Island"
    bl_description = "Re-unwrap the island under the selected face(s)"
    queue_target = staticmethod(queue_island)
    verb = "Unwrapping"


class UVGAMI_OT_relax_island(IslandOperator, bpy.types.Operator):
    bl_idname = "uvgami.relax_island"
    bl_label = "Relax Island"
    bl_description = (
        "Relax the island under the selected faces to reduce stretching"
        " without changing its seams"
    )
    queue_target = staticmethod(queue_relax)
    verb = "Relaxing"


class UVGAMI_OT_combine_islands(FixOperatorMixin, bpy.types.Operator):
    bl_idname = "uvgami.combine_islands"
    bl_label = "Combine Islands"
    bl_description = "Select a face on two islands to re-unwrap them as one island"

    def queue_selection(self, obj, props, engine, engine_ctx):
        targets = target_islands(obj)
        if len(targets) < 2:
            raise FixError("Select faces on at least two islands")
        if not islands_connected(obj.data, targets):
            raise FixError(
                "The selected islands don't touch, select a face on each island between them"
            )

        group = sorted({face for target in targets for face in target.faces})
        boxes = [target.bbox for target in targets]
        bbox = (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )
        area = sum(target.area for target in targets)

        island_of = {
            face: index for index, target in enumerate(targets) for face in target.faces
        }
        temp = island_copy(obj, group)
        try:
            weld_shared_seams(temp, [island_of[face] for face in group], len(targets))
        except FixError:
            remove_temp(temp)
            raise
        minimum_stretch_unwrap(temp, obj)
        bm = new_bmesh(temp)
        folded = bool(folded_faces(bm, bm.loops.layers.uv.active))
        bm.free()

        if folded:
            # optcuts can't unfold a kept map
            remove_temp(temp)

            def queue_one(index, input_path):
                queue_island(obj, group, bbox, area, index + 1, input_path, props)

            message = "The combined island folds, unwrapping it from scratch instead"
            level = "WARNING"
        else:

            def queue_one(index, input_path):
                queue_nocut(obj, temp, group, bbox, area, index + 1, input_path)

            message = f"Combining {len(targets)} islands"
            level = "INFO"

        queue_targets(engine, engine_ctx, obj, 1, queue_one)
        self.report({level}, message)
        return {"FINISHED"}


class AreaOperator(FixOperatorMixin):
    nocut = False

    def queue_selection(self, obj, props, engine, engine_ctx):
        selection = target_areas(obj, props.area_expand)
        targets = [
            target
            for target in selection.targets
            if not spans_own_seam(obj.data, target[0])
        ]
        seam_skipped = len(selection.targets) - len(targets)
        if not targets:
            if selection.whole_islands:
                raise FixError(
                    "The whole island is selected, use Unwrap Island instead"
                )
            if seam_skipped:
                raise FixError(
                    "The area covers both sides of a seam,"
                    " deselect the faces on one side"
                )
            raise FixError("The area rings a hole, deselect a face to break the ring")

        def queue_one(index, input_path):
            patch, border = targets[index]
            queue_area(obj, patch, border, index + 1, input_path, nocut=self.nocut)

        queue_targets(engine, engine_ctx, obj, len(targets), queue_one)

        notes = []
        if selection.whole_islands:
            notes.append(f"{selection.whole_islands} whole island(s) skipped")
        if selection.hole_rings:
            notes.append(f"{selection.hole_rings} area(s) around holes skipped")
        if seam_skipped:
            notes.append(f"{seam_skipped} area(s) on both sides of a seam skipped")
        if notes:
            self.report({"WARNING"}, ", ".join(notes))
        return {"FINISHED"}


class UVGAMI_OT_unwrap_area(AreaOperator, bpy.types.Operator):
    bl_idname = "uvgami.unwrap_area"
    bl_label = "Unwrap Area"
    bl_description = "Re-unwrap the selected faces with cuts if necessary"


class UVGAMI_OT_relax_area(AreaOperator, bpy.types.Operator):
    bl_idname = "uvgami.relax_area"
    bl_label = "Relax Area"
    bl_description = "Relax the selected faces to reduce stretching"
    nocut = True
