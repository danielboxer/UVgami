# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import re

import bmesh
import bpy

from .logger import logger
from .utils.mesh import check_exists, new_bmesh, set_bmesh


class Job:
    def __init__(self, count):
        self.count = count
        self.unwrapped = []
        self.is_expanded = False

    def is_completed(self):
        return len(self.unwrapped) == self.count


class Preserve(Job):
    def __init__(self, count):
        super().__init__(count)

    def finish(self, unwrap, output, added_edges):
        # return mesh to original state
        bm = new_bmesh(output)

        e_dict = {}
        for edge in bm.edges:
            e_dict[(edge.verts[0].index, edge.verts[1].index)] = edge

        # check if the edges are set already
        if not added_edges:
            added_edges = unwrap.added_edges

        if bpy.context.scene.uvgami.maintain_mode == "PARTIAL":
            # get seams so they can be avoided
            uvs = []
            uv_idcs = []
            uvvert_to_meshvert = {}
            mesh_verts = []
            uv_count = 0
            uv_layer = bm.loops.layers.uv.active

            # get uv data
            for face in bm.faces:
                uv_i = []
                for loop in face.loops:
                    # get uv coordinate
                    uv = loop[uv_layer].uv
                    uvs.append((uv.x, uv.y, 0))
                    # all face points are added, duplicates are removed later
                    # that means the index is new each time
                    uv_i.append(uv_count)
                    uv_count += 1
                    # store the original mesh vertex so it can be accessed using the uvs
                    mesh_verts.append(loop.vert)
                uv_idcs.append(uv_i)

            # make uv bmesh out of mesh data
            mesh_data = bpy.data.meshes.new("")
            mesh_data.from_pydata(uvs, [], uv_idcs)
            uvbm = bmesh.new()
            uvbm.from_mesh(mesh_data)

            # make lookup table between mesh and uv mesh
            for uv_v_idx, uv_v in enumerate(uvbm.verts):
                uvvert_to_meshvert[uv_v] = mesh_verts[uv_v_idx]

            # the faces will all be separate, so merging by distance joins them
            bmesh.ops.remove_doubles(uvbm, verts=uvbm.verts, dist=0.0001)

            # find boundary edges of uv bmesh which are seams of original bmesh
            seams = []
            for e in uvbm.edges:
                if e.is_boundary:
                    uv_v1 = e.verts[0]
                    uv_v2 = e.verts[1]

                    m_v1 = uvvert_to_meshvert[uv_v1]
                    m_v2 = uvvert_to_meshvert[uv_v2]

                    # get edge from vertices
                    for edge in m_v1.link_edges:
                        if edge.other_vert(m_v1) is m_v2:
                            seams.append(edge)
            uvbm.free()

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

            if bpy.context.scene.uvgami.maintain_mode == "PARTIAL":
                if bm_edge not in seams:
                    dissolve_edges.append(bm_edge)
            else:
                dissolve_edges.append(bm_edge)

        bmesh.ops.dissolve_edges(bm, edges=dissolve_edges)
        set_bmesh(bm, output)


class Join(Job):
    def __init__(self, count):
        super().__init__(count)

    def finish(self, unwrap):
        paths = [u.path.parents[1] / "output" / u.path.name for u in self.unwrapped]
        edge_path = unwrap.edge_path

        # set the current path to the first obj in the job
        # this is the file that will be imported
        path = paths[0]

        prev_v = 0
        prev_vt = 0
        # go through first obj file to get the starting size
        with paths[0].open() as f:
            for line in f:
                if line.startswith("v "):
                    prev_v += 1
                elif line.startswith("vt "):
                    prev_vt += 1
                elif line.startswith("f "):
                    break

        new_v = prev_v
        new_vt = prev_vt
        with paths[0].open("a") as f:
            # since there are multiple obj files combined, the size of the
            # previous ones must be added to the index numbers of the next
            for obj_path in paths[1:]:
                with obj_path.open() as f2:
                    for line in f2:
                        new_line = line
                        if line.startswith("v "):
                            new_v += 1
                        elif line.startswith("vt "):
                            new_vt += 1
                        elif line.startswith("f "):
                            line = line[2:]
                            line = re.split(r"[ /]", line)
                            new_line = "f "

                            count = 0
                            for num in line:
                                count += 1
                                if count == 1:
                                    new_line += str(int(num) + prev_v)
                                    new_line += "/"
                                elif count == 2:
                                    new_line += str(int(num) + prev_vt)
                                    new_line += " "
                                    count = 0
                            new_line += "\n"

                        f.write(new_line)
                    prev_v = new_v
                    prev_vt = new_vt

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


class Cleanup(Job):
    def __init__(self, count, action):
        super().__init__(count)
        self.action = action

    def finish(self, input_mesh):
        if check_exists(input_mesh):
            if self.action == "HIDE":
                input_mesh.hide_set(True)
            # deleting the object while editing it will crash blender
            elif self.action == "DELETE" and input_mesh.mode != "EDIT":
                bpy.data.objects.remove(input_mesh, do_unlink=True)


class TransferUVs(Job):
    def __init__(self, count):
        super().__init__(count)

    def finish(self, input_mesh, output):
        if not check_exists(input_mesh) or not check_exists(output):
            return False, False

        # need to exit edit mode to modify mesh data
        was_in_edit = input_mesh.mode == "EDIT"
        if was_in_edit:
            old_active = bpy.context.view_layer.objects.active
            bpy.context.view_layer.objects.active = input_mesh
            bpy.ops.object.mode_set(mode="OBJECT")

        input_data = input_mesh.data
        output_data = output.data

        # vertex count must match
        if len(input_data.vertices) != len(output_data.vertices):
            if was_in_edit:
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.context.view_layer.objects.active = old_active
            return False, False

        output_uv = output_data.uv_layers.active
        if output_uv is None:
            if was_in_edit:
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.context.view_layer.objects.active = old_active
            return False, False

        # get or create UV layer on input
        if not input_data.uv_layers:
            input_data.uv_layers.new(name="UVMap")
        input_uv = input_data.uv_layers.active

        topology_matched = len(input_data.polygons) == len(
            output_data.polygons
        ) and len(input_data.loops) == len(output_data.loops)

        # build output vert index -> input vert index mapping via world position
        out_to_in_vert = self._build_vertex_mapping(
            input_mesh, output, input_data, output_data
        )
        if out_to_in_vert is None:
            if was_in_edit:
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.context.view_layer.objects.active = old_active
            return False, False

        # build input face lookup: frozenset of vert indices -> poly
        input_face_lookup = {}
        for poly in input_data.polygons:
            face_key = frozenset(poly.vertices)
            input_face_lookup[face_key] = poly

        # for each output face, find matching input face and transfer UVs
        for out_poly in output_data.polygons:
            # map output face vertices to input vertex indices
            mapped_verts = frozenset(
                out_to_in_vert.get(v, -1) for v in out_poly.vertices
            )

            target_poly = None
            if mapped_verts in input_face_lookup:
                # exact face match
                target_poly = input_face_lookup[mapped_verts]
            else:
                # output face is a subset of an input face (triangulated quad/ngon)
                for face_key, poly in input_face_lookup.items():
                    if mapped_verts.issubset(face_key):
                        target_poly = poly
                        break

            if target_poly is None:
                continue

            # build vert -> loop index lookup for the input face
            in_vert_to_loop = {}
            for in_loop_idx in range(
                target_poly.loop_start,
                target_poly.loop_start + target_poly.loop_total,
            ):
                in_vert_to_loop[input_data.loops[in_loop_idx].vertex_index] = (
                    in_loop_idx
                )

            # transfer UV for each loop in the output face
            for out_loop_idx in range(
                out_poly.loop_start, out_poly.loop_start + out_poly.loop_total
            ):
                out_vert = output_data.loops[out_loop_idx].vertex_index
                in_vert = out_to_in_vert.get(out_vert)
                if in_vert is not None and in_vert in in_vert_to_loop:
                    in_loop_idx = in_vert_to_loop[in_vert]
                    input_uv.uv[in_loop_idx].vector = output_uv.uv[out_loop_idx].vector

        # transfer seams using position-based edge matching
        self._transfer_seams(input_data, output_data, out_to_in_vert)

        input_data.update()

        # remove output mesh
        bpy.data.objects.remove(output, do_unlink=True)
        # unhide input in case it was hidden
        input_mesh.hide_set(False)

        if was_in_edit:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.context.view_layer.objects.active = old_active

        return True, topology_matched

    def _build_vertex_mapping(self, input_mesh, output_mesh, input_data, output_data):
        precision = 4
        input_matrix = input_mesh.matrix_world
        output_matrix = output_mesh.matrix_world

        input_pos_to_idx = {}
        for v in input_data.vertices:
            world_co = input_matrix @ v.co
            key = (
                round(world_co.x, precision),
                round(world_co.y, precision),
                round(world_co.z, precision),
            )
            # handle multiple vertices at same position
            if key not in input_pos_to_idx:
                input_pos_to_idx[key] = []
            input_pos_to_idx[key].append(v.index)

        out_to_in = {}
        for v in output_data.vertices:
            world_co = output_matrix @ v.co
            key = (
                round(world_co.x, precision),
                round(world_co.y, precision),
                round(world_co.z, precision),
            )
            if key in input_pos_to_idx and input_pos_to_idx[key]:
                # pop from list to handle duplicate positions correctly
                out_to_in[v.index] = input_pos_to_idx[key].pop(0)

        # if we couldn't map all vertices, the transfer won't work
        if len(out_to_in) != len(output_data.vertices):
            return None
        return out_to_in

    def _transfer_seams(self, input_data, output_data, out_to_in_vert):
        seam_edges = set()
        for edge in output_data.edges:
            if edge.use_seam:
                v0 = out_to_in_vert.get(edge.vertices[0])
                v1 = out_to_in_vert.get(edge.vertices[1])
                if v0 is not None and v1 is not None:
                    seam_edges.add((min(v0, v1), max(v0, v1)))

        for edge in input_data.edges:
            edge_key = (
                min(edge.vertices[0], edge.vertices[1]),
                max(edge.vertices[0], edge.vertices[1]),
            )
            edge.use_seam = edge_key in seam_edges


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
