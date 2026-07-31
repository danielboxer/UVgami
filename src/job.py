from collections import namedtuple

import bmesh
import bpy

from .logger import logger
from .objfile import merge_obj_files
from .proxy import transfer_cuts
from .seams import uv_area_fit
from .uv_transfer import plan_transfer
from .utils.mesh import check_exists, deselect_all, new_bmesh, set_bmesh

TransferOutcome = namedtuple("TransferOutcome", ["applied", "split_count", "detail"])


def output_mesh_data(output, output_uv):
    """World positions, polygons, per-face loop uvs and seams of an engine
    output object, in the plain form plan_transfer takes."""
    output_data = output.data
    output_matrix = output.matrix_world

    output_positions = [tuple(output_matrix @ v.co) for v in output_data.vertices]
    output_polygons = []
    output_uvs = []
    for poly in output_data.polygons:
        output_polygons.append(list(poly.vertices))
        output_uvs.append(
            [
                tuple(output_uv.uv[poly.loop_start + c].vector)
                for c in range(poly.loop_total)
            ]
        )

    output_seams = [
        (edge.vertices[0], edge.vertices[1])
        for edge in output_data.edges
        if edge.use_seam
    ]

    return output_positions, output_polygons, output_uvs, output_seams


class Job:
    def __init__(self, count):
        self.count = count
        self.unwrapped = []
        self.is_expanded = False

    def is_completed(self):
        return len(self.unwrapped) == self.count


class Preserve(Job):
    @staticmethod
    def _seam_edges(bm):
        """The mesh edges that carry a uv seam, found by rebuilding the uv
        layout as its own mesh: welding its coincident corners leaves the
        seams as the only boundary edges."""
        uvs = []
        uv_idcs = []
        mesh_verts = []
        uv_count = 0
        uv_layer = bm.loops.layers.uv.active

        for face in bm.faces:
            uv_i = []
            for loop in face.loops:
                uv = loop[uv_layer].uv
                uvs.append((uv.x, uv.y, 0))
                # all face points are added, duplicates are removed later
                # that means the index is new each time
                uv_i.append(uv_count)
                uv_count += 1
                # store the original mesh vertex so it can be accessed using the uvs
                mesh_verts.append(loop.vert)
            uv_idcs.append(uv_i)

        mesh_data = bpy.data.meshes.new("")
        mesh_data.from_pydata(uvs, [], uv_idcs)
        uvbm = bmesh.new()
        uvbm.from_mesh(mesh_data)

        uvvert_to_meshvert = {}
        for uv_v_idx, uv_v in enumerate(uvbm.verts):
            uvvert_to_meshvert[uv_v] = mesh_verts[uv_v_idx]

        # the faces will all be separate, so merging by distance joins them
        bmesh.ops.remove_doubles(uvbm, verts=uvbm.verts, dist=0.0001)

        seams = []
        for e in uvbm.edges:
            if e.is_boundary:
                m_v1 = uvvert_to_meshvert[e.verts[0]]
                m_v2 = uvvert_to_meshvert[e.verts[1]]
                for edge in m_v1.link_edges:
                    if edge.other_vert(m_v1) is m_v2:
                        seams.append(edge)
        uvbm.free()
        return seams

    def finish(self, unwrap, output, added_edges):
        # return mesh to original state
        bm = new_bmesh(output)

        e_dict = {}
        for edge in bm.edges:
            e_dict[(edge.verts[0].index, edge.verts[1].index)] = edge

        # check if the edges are set already
        if not added_edges:
            added_edges = unwrap.added_edges

        # partial keeps the seams, so only non-seam edges may dissolve
        seams = self._seam_edges(bm) if unwrap.maintain_mode == "PARTIAL" else ()

        dissolve_edges = []
        for e in added_edges:
            bm_edge = None

            if e in e_dict:
                bm_edge = e_dict[e]
            elif (e[1], e[0]) in e_dict:
                bm_edge = e_dict[(e[1], e[0])]
            else:
                # this shouldn't happen, edge not found
                if (
                    logger.get_latest().errors
                    and logger.get_latest().errors[-1]
                    == "    Error removing added edge"
                ):
                    # don't add duplicate errors
                    continue
                logger.add_data("errors", "Error removing added edge")
                # skip removing edge
                continue

            if bm_edge not in seams:
                dissolve_edges.append(bm_edge)

        bmesh.ops.dissolve_edges(bm, edges=dissolve_edges)
        set_bmesh(bm, output)


class Join(Job):
    def __init__(self, count):
        super().__init__(count)
        # set when a whole-group cancel is issued, so the exporter can drop the
        # group's still-unexported pieces
        self.cancel_requested = False

    def finish(self, unwrap):
        paths = [u.path.parents[1] / "output" / u.path.name for u in self.unwrapped]
        edge_path = unwrap.edge_path

        # the merged first obj is the file that will be imported
        path = merge_obj_files(paths)

        added_edges = []
        if unwrap.preserve_job is not None:
            # combine all added edges in the group
            v_count = 0
            for e_idx, edges in enumerate([u.added_edges for u in self.unwrapped]):
                for v1, v2 in edges:
                    added_edges.append((v1 + v_count, v2 + v_count))
                v_count += self.unwrapped[e_idx].vertex_count

            # combine all edge files
            unwraps = self.unwrapped
            edge_path = unwraps[0].edge_path
            v_count = unwraps[0].vertex_count
            e_paths = [u.edge_path for u in unwraps]
            with e_paths[0].open("a") as f:
                for e_idx, e_path in enumerate(e_paths[1:], 1):
                    with e_path.open() as f2:
                        for line in f2:
                            line = line.split()
                            f.write(
                                f"{int(line[0]) + v_count} {int(line[1]) + v_count}\n"
                            )
                    v_count += unwraps[e_idx].vertex_count

        return (path, edge_path, added_edges)


class HideInput(Job):
    def finish(self, input_mesh):
        if check_exists(input_mesh):
            input_mesh.hide_set(True)


class TransferUVs(Job):
    # whether the manager should repack the input mesh in place of the
    # deleted output at session end
    repack_input = True

    def finish(self, input_mesh, output):
        if not check_exists(input_mesh) or not check_exists(output):
            return TransferOutcome(False, 0, "input or output object missing")

        output_uv = output.data.uv_layers.active
        if output_uv is None:
            return TransferOutcome(False, 0, "output mesh has no uv layer")

        # exit edit mode to read and write mesh data, restore it no matter what
        old_active = bpy.context.view_layer.objects.active
        was_in_edit = input_mesh.mode == "EDIT"
        try:
            if was_in_edit:
                bpy.context.view_layer.objects.active = input_mesh
                bpy.ops.object.mode_set(mode="OBJECT")

            result = plan_transfer(*self._extract(input_mesh, output, output_uv))
            if not result.ok:
                return TransferOutcome(False, 0, f"{result.reason}: {result.detail}")

            self._apply(input_mesh, result)
        finally:
            if was_in_edit:
                bpy.context.view_layer.objects.active = input_mesh
                bpy.ops.object.mode_set(mode="EDIT")
            bpy.context.view_layer.objects.active = old_active

        # only tear down the output once the whole plan applied
        bpy.data.objects.remove(output, do_unlink=True)
        input_mesh.hide_set(False)
        return TransferOutcome(True, len(result.split_faces), "")

    def _extract(self, input_mesh, output, output_uv):
        input_data = input_mesh.data
        input_matrix = input_mesh.matrix_world

        input_positions = [tuple(input_matrix @ v.co) for v in input_data.vertices]
        input_polygons = [list(poly.vertices) for poly in input_data.polygons]

        return (input_positions, input_polygons) + output_mesh_data(output, output_uv)

    def _apply(self, input_mesh, plan):
        if plan.split_faces:
            self._apply_with_splits(input_mesh, plan)
            return

        input_data = input_mesh.data
        if not input_data.uv_layers:
            input_data.uv_layers.new(name="UVMap")
        input_uv = input_data.uv_layers.active

        loop_idx = 0
        for poly in input_data.polygons:
            for c in range(poly.loop_total):
                input_uv.uv[poly.loop_start + c].vector = plan.loop_uvs[loop_idx]
                loop_idx += 1

        for edge in input_data.edges:
            a, b = edge.vertices[0], edge.vertices[1]
            edge.use_seam = ((a, b) if a < b else (b, a)) in plan.seam_edges

        input_data.update()

    def _apply_with_splits(self, input_mesh, plan):
        """Same as _apply, but rebuilds the faces a uv cut runs through. Goes
        through bmesh because changing topology needs it."""
        bm = new_bmesh(input_mesh)
        uv_layer = bm.loops.layers.uv.verify()

        loop_idx = 0
        to_split = []
        for face_idx, face in enumerate(bm.faces):
            parts = plan.split_faces.get(face_idx)
            if parts is None:
                for loop in face.loops:
                    loop[uv_layer].uv = plan.loop_uvs[loop_idx]
                    loop_idx += 1
            else:
                loop_idx += len(face.loops)
                to_split.append((face, parts, face.material_index, face.smooth))

        # delete first so a new piece can never collide with the face it replaces
        bmesh.ops.delete(
            bm, geom=[face for face, _, _, _ in to_split], context="FACES_ONLY"
        )
        bm.verts.ensure_lookup_table()
        for _, parts, material_index, smooth in to_split:
            for verts, uvs in parts:
                new_face = bm.faces.new([bm.verts[v] for v in verts])
                new_face.material_index = material_index
                new_face.smooth = smooth
                for loop, uv in zip(new_face.loops, uvs):
                    loop[uv_layer].uv = uv

        for edge in bm.edges:
            a, b = edge.verts[0].index, edge.verts[1].index
            edge.seam = ((a, b) if a < b else (b, a)) in plan.seam_edges

        set_bmesh(bm, input_mesh)


class IslandUVs(TransferUVs):
    """Put an engine re-unwrap of one island back into the input mesh, scaled
    to the uv area it used to cover so the rest of the atlas stays put. Rides
    TransferUVs' position matching, fed only the island's faces."""

    repack_input = False

    def __init__(self, faces, bbox, area):
        super().__init__(1)
        self.faces = faces
        self.bbox = bbox
        self.area = area
        self.orig_vert = []
        self.loop_base = {}
        self.loop_counts = []

    def _extract(self, input_mesh, output, output_uv):
        data = input_mesh.data
        matrix = input_mesh.matrix_world

        used = sorted({v for fi in self.faces for v in data.polygons[fi].vertices})
        self.orig_vert = used
        local = {v: i for i, v in enumerate(used)}

        positions = [tuple(matrix @ data.vertices[v].co) for v in used]
        polygons = []
        self.loop_counts = []
        base = 0
        for fi in self.faces:
            poly = data.polygons[fi]
            self.loop_base[fi] = base
            self.loop_counts.append(poly.loop_total)
            base += poly.loop_total
            polygons.append([local[v] for v in poly.vertices])

        return (positions, polygons) + output_mesh_data(output, output_uv)

    def _fit(self, plan):
        """Scale the engine's layout back to the island's old uv area, centered
        on its old spot. The faces a cut split are read from their new parts,
        the loops they came from are dead."""
        polygons = []
        for i, count in enumerate(self.loop_counts):
            parts = plan.split_faces.get(i)
            if parts is None:
                base = self.loop_base[self.faces[i]]
                polygons.append([plan.loop_uvs[base + c] for c in range(count)])
            else:
                polygons.extend(part_uvs for _, part_uvs in parts)
        move = uv_area_fit(polygons, self.area, self.bbox)
        for k in plan.loop_uvs:
            plan.loop_uvs[k] = move(plan.loop_uvs[k])
        for fi, parts in plan.split_faces.items():
            plan.split_faces[fi] = [
                (verts, [move(uv) for uv in part_uvs]) for verts, part_uvs in parts
            ]

    def _apply(self, input_mesh, plan):
        self._fit(plan)
        data = input_mesh.data
        ov = self.orig_vert
        seams = {
            ((ov[a], ov[b]) if ov[a] < ov[b] else (ov[b], ov[a]))
            for a, b in plan.seam_edges
        }

        # only edges interior to the island get the plan's seams, the island
        # boundary and the rest of the mesh keep their marks
        owner_count = {}
        for fi in self.faces:
            poly = data.polygons[fi].vertices
            n = len(poly)
            for i in range(n):
                a, b = poly[i], poly[(i + 1) % n]
                key = (a, b) if a < b else (b, a)
                owner_count[key] = owner_count.get(key, 0) + 1
        interior = {key for key, count in owner_count.items() if count == 2}

        split_faces = {self.faces[fi]: parts for fi, parts in plan.split_faces.items()}
        if split_faces:
            self._apply_island_splits(input_mesh, plan, split_faces, seams, interior)
            return

        layer = data.uv_layers.active
        for fi in self.faces:
            poly = data.polygons[fi]
            base = self.loop_base[fi]
            for c in range(poly.loop_total):
                layer.uv[poly.loop_start + c].vector = plan.loop_uvs[base + c]

        for edge in data.edges:
            a, b = edge.vertices
            key = (a, b) if a < b else (b, a)
            if key in interior:
                edge.use_seam = key in seams

        data.update()

    def _apply_island_splits(self, input_mesh, plan, split_faces, seams, interior):
        """Same as _apply, but rebuilds the island faces a uv cut runs
        through, like TransferUVs._apply_with_splits."""
        bm = new_bmesh(input_mesh)
        uv_layer = bm.loops.layers.uv.verify()
        bm.faces.ensure_lookup_table()

        to_split = []
        for fi in self.faces:
            face = bm.faces[fi]
            parts = split_faces.get(fi)
            if parts is None:
                base = self.loop_base[fi]
                for c, loop in enumerate(face.loops):
                    loop[uv_layer].uv = plan.loop_uvs[base + c]
            else:
                to_split.append((face, parts, face.material_index, face.smooth))

        bmesh.ops.delete(
            bm, geom=[face for face, _, _, _ in to_split], context="FACES_ONLY"
        )
        bm.verts.ensure_lookup_table()
        ov = self.orig_vert
        for _, parts, material_index, smooth in to_split:
            for verts, uvs in parts:
                new_face = bm.faces.new([bm.verts[ov[v]] for v in verts])
                new_face.material_index = material_index
                new_face.smooth = smooth
                for loop, uv in zip(new_face.loops, uvs):
                    loop[uv_layer].uv = uv

        for edge in bm.edges:
            a, b = edge.verts[0].index, edge.verts[1].index
            key = (a, b) if a < b else (b, a)
            if key in seams:
                edge.seam = True
            elif key in interior:
                edge.seam = False

        set_bmesh(bm, input_mesh)


class AreaUVs(IslandUVs):
    """Put an engine fix of part of an island back into the input mesh. The
    engine held the patch border in place, so instead of a bbox fit the
    output is aligned by undoing its normalization through those pinned
    loops, and they snap back to their exact old uvs so the patch rejoins
    the island seamlessly."""

    def __init__(self, faces, pins, snapshot):
        super().__init__(faces, None, None)
        self.pins = pins  # (face index, corner, old uv)
        self.snapshot = snapshot  # every patch loop's old uv, for restore

    def restore(self, input_mesh):
        """Put the patch uvs back after a failed run. The flipped pre-repair
        changes the map before the engine even starts, so a failure must not
        leave that behind."""
        if not check_exists(input_mesh):
            return
        old_active = bpy.context.view_layer.objects.active
        was_in_edit = input_mesh.mode == "EDIT"
        try:
            if was_in_edit:
                bpy.context.view_layer.objects.active = input_mesh
                bpy.ops.object.mode_set(mode="OBJECT")
            data = input_mesh.data
            layer = data.uv_layers.active
            for fi, c, uv in self.snapshot:
                layer.uv[data.polygons[fi].loop_start + c].vector = uv
        finally:
            if was_in_edit:
                bpy.context.view_layer.objects.active = input_mesh
                bpy.ops.object.mode_set(mode="EDIT")
            bpy.context.view_layer.objects.active = old_active

    def _fit(self, plan):
        pairs = []
        for fi, corner, old in self.pins:
            new = plan.loop_uvs.get(self.loop_base[fi] + corner)
            if new is not None:
                pairs.append((old, new))
        if len(pairs) < 2:
            return

        # the output is the solved map scaled into the unit box, recover the
        # uniform scale and offset from the two most distant pins
        lo = min(pairs, key=lambda p: p[0][0])
        hi = max(pairs, key=lambda p: p[0][0])
        if hi[1][0] == lo[1][0]:
            lo = min(pairs, key=lambda p: p[0][1])
            hi = max(pairs, key=lambda p: p[0][1])
            scale = (hi[0][1] - lo[0][1]) / (hi[1][1] - lo[1][1])
        else:
            scale = (hi[0][0] - lo[0][0]) / (hi[1][0] - lo[1][0])
        du = lo[0][0] - scale * lo[1][0]
        dv = lo[0][1] - scale * lo[1][1]

        def move(uv):
            return (scale * uv[0] + du, scale * uv[1] + dv)

        for k in plan.loop_uvs:
            plan.loop_uvs[k] = move(plan.loop_uvs[k])
        for fi, parts in plan.split_faces.items():
            plan.split_faces[fi] = [
                (verts, [move(uv) for uv in part_uvs]) for verts, part_uvs in parts
            ]
        # pinned loops held to float noise, snap them so the border welds
        for fi, corner, old in self.pins:
            k = self.loop_base[fi] + corner
            if k in plan.loop_uvs:
                plan.loop_uvs[k] = old


class ProxyUVs(Job):
    """Cut the original along the unwrapped proxy's seams and unwrap it.

    Rides the transfer uvs slot: the engine ran on a decimated copy, so the
    original is the mesh that needs a uv map and the copy is thrown away."""

    repack_input = True

    def finish(self, input_mesh, output):
        if not check_exists(input_mesh) or not check_exists(output):
            return TransferOutcome(False, 0, "input or output object missing")
        if output.data.uv_layers.active is None:
            return TransferOutcome(False, 0, "output mesh has no uv layer")

        old_active = bpy.context.view_layer.objects.active
        old_selected = list(bpy.context.selected_objects)
        old_mode = old_active.mode if old_active is not None else "OBJECT"
        if old_mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        # the unwrap goes through bpy.ops, which would take every other
        # selected mesh into edit mode along with this one
        deselect_all()
        try:
            transfer_cuts(input_mesh, output)
        finally:
            deselect_all()
            for obj in old_selected:
                if check_exists(obj):
                    obj.select_set(True)
            if old_active is not None and check_exists(old_active):
                bpy.context.view_layer.objects.active = old_active
                if old_mode != "OBJECT":
                    bpy.ops.object.mode_set(mode=old_mode)

        bpy.data.objects.remove(output, do_unlink=True)
        input_mesh.hide_set(False)
        return TransferOutcome(True, 0, "")


class Symmetrise(Job):
    def __init__(self, count, axes, center, overlap):
        super().__init__(count)
        self.x = "X" in axes
        self.y = "Y" in axes
        self.z = "Z" in axes
        self.center = center
        self.overlap = overlap

    def finish(self, output):
        mirror = output.modifiers.new("Mirror", "MIRROR")
        mirror.use_axis = (self.x, self.y, self.z)
        empty = None
        # if the object origin is not at the center, the mirror axis will be wrong
        if self.center != output.matrix_world.to_translation():
            empty = bpy.data.objects.new("Empty", None)
            empty.location.x = self.center.x
            empty.location.y = self.center.y
            empty.location.z = self.center.z
            mirror.mirror_object = empty

        if not self.overlap:
            # separate islands
            mirror.use_mirror_u = True
            mirror.use_mirror_v = True

        old_active = bpy.context.view_layer.objects.active
        bpy.context.view_layer.objects.active = output
        bpy.ops.object.modifier_apply(modifier=mirror.name)
        bpy.context.view_layer.objects.active = old_active
        if empty is not None:
            bpy.data.objects.remove(empty, do_unlink=True)
