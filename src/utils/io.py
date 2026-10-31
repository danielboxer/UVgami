import re
from collections import namedtuple

import bpy
import numpy

from ..no_bpy.mesh import face_starts
from .mesh import loop_uvs, loop_vertices_and_face_sizes, world_positions


def export_obj(obj, path, *, export_uv, flip_mirrored=False, matrix=None):
    mesh = obj.data
    matrix = numpy.array(obj.matrix_world if matrix is None else matrix)
    positions = world_positions(mesh, matrix)
    loop_vertices, totals = loop_vertices_and_face_sizes(mesh)

    # mirror matrices turn faces inside out unless the corner order is reversed
    loop_order = None
    if flip_mirrored and numpy.linalg.det(matrix[:3, :3]) < 0:
        starts = face_starts(totals)
        within = numpy.arange(len(loop_vertices)) - numpy.repeat(starts, totals)
        loop_order = numpy.repeat(starts + totals - 1, totals) - within
        loop_vertices = loop_vertices[loop_order]

    layer = mesh.uv_layers.active
    export_uv = export_uv and layer is not None
    vertex_of_uv = None

    with path.open("w") as file:
        # optcuts' obj reader prints a warning for o lines but not g lines
        file.write(f"g {obj.name}\n")
        # one %-format over the whole array runs in C, f-strings are ten times slower
        file.write(
            ("v %.9f %.9f %.9f\n" * len(positions)) % tuple(positions.ravel().tolist())
        )

        if export_uv:
            uvs = loop_uvs(mesh)
            if loop_order is not None:
                uvs = uvs[loop_order]
            # optcuts welds corners that share a vt
            scaled = numpy.rint(uvs * 1e9).astype(numpy.int64)
            order = numpy.lexsort((scaled[:, 1], scaled[:, 0], loop_vertices))
            sorted_vertices = loop_vertices[order]
            sorted_u = scaled[order, 0]
            sorted_v = scaled[order, 1]
            first = numpy.zeros(len(order), dtype=bool)
            first[:1] = True
            first[1:] = (
                (sorted_vertices[1:] != sorted_vertices[:-1])
                | (sorted_u[1:] != sorted_u[:-1])
                | (sorted_v[1:] != sorted_v[:-1])
            )
            loop_uv_indices = numpy.empty(len(order), dtype=numpy.int64)
            loop_uv_indices[order] = numpy.cumsum(first) - 1
            vertex_of_uv = sorted_vertices[first]
            values = numpy.column_stack((sorted_u[first], sorted_v[first])) / 1e9
            # Blender's obj exporter rounds to 6 decimals, which flips tiny uv triangles
            file.write(
                ("vt %.9f %.9f\n" * len(values)) % tuple(values.ravel().tolist())
            )
            corners = numpy.column_stack((loop_vertices + 1, loop_uv_indices + 1))
            token = "%d/%d"
        else:
            corners = loop_vertices[:, None] + 1
            token = "%d"

        if len(totals) and (totals == totals[0]).all():
            size = int(totals[0])
            line_format = "f " + " ".join([token] * size) + "\n"
            file.write((line_format * len(totals)) % tuple(corners.ravel().tolist()))
        else:
            flat = corners.ravel().tolist()
            width = corners.shape[1]
            lines = []
            offset = 0
            for total in totals.tolist():
                line_format = "f " + " ".join([token] * total) + "\n"
                lines.append(line_format % tuple(flat[offset : offset + total * width]))
                offset += total * width
            file.writelines(lines)

    return vertex_of_uv


def _block(text, prefix):
    tag = "\n" + prefix + " "
    if text.startswith(prefix + " "):
        start = 0
    else:
        start = text.find(tag)
        if start == -1:
            return ""
        start += 1
    end = text.find("\n", max(text.rfind(tag), start) + 1)
    block = text[start:] + "\n" if end == -1 else text[start : end + 1]
    if block.count("\n") == block.count(tag) + 1:
        return block
    runs = re.findall(rf"(?m)^(?:{prefix} [^\n]*\n?)+", text)
    block = "".join(runs)
    return block if block.endswith("\n") else block + "\n"


def obj_numeric_columns(text, prefix, columns):
    block = _block(text, prefix)
    if not block:
        return numpy.empty((0, columns))
    lines = block.count("\n")
    tokens = block.replace(prefix + " ", " ").split()
    if len(tokens) % lines == 0 and len(tokens) // lines >= columns:
        stride = len(tokens) // lines
        return numpy.array(tokens, dtype=numpy.float64).reshape(-1, stride)[:, :columns]
    return numpy.array(
        [line.split()[1 : columns + 1] for line in block.splitlines()],
        dtype=numpy.float64,
    )


ObjFaces = namedtuple("ObjFaces", "totals loop_vertices loop_uv_indices")


def parse_obj_faces(text):
    block = _block(text, "f")
    empty = numpy.empty(0, dtype=numpy.int64)
    if not block:
        return ObjFaces(numpy.empty(0, dtype=numpy.int32), empty, empty)
    lines = block.count("\n")
    first_line = block.split("\n", 1)[0][2:].split()
    size = len(first_line)
    stride = first_line[0].count("/") + 1
    tokens = block.replace("f ", " ").replace("//", "/0/").replace("/", " ").split()
    if len(tokens) == lines * size * stride:
        numbers = numpy.array(tokens, dtype=numpy.int64).reshape(-1, stride)
        totals = numpy.full(lines, size, dtype=numpy.int32)
        loop_vertices = numbers[:, 0] - 1
        if stride > 1:
            loop_uv_indices = numpy.maximum(numbers[:, 1] - 1, 0)
        else:
            loop_uv_indices = numpy.zeros(len(numbers), dtype=numpy.int64)
        return ObjFaces(totals, loop_vertices, loop_uv_indices)

    totals = []
    loop_vertices = []
    loop_uv_indices = []
    for line in block.splitlines():
        corners = line.split()[1:]
        totals.append(len(corners))
        for corner in corners:
            parts = corner.split("/")
            loop_vertices.append(int(parts[0]) - 1)
            if len(parts) > 1 and parts[1]:
                loop_uv_indices.append(int(parts[1]) - 1)
            else:
                loop_uv_indices.append(0)
    return ObjFaces(
        numpy.array(totals, dtype=numpy.int32),
        numpy.array(loop_vertices, dtype=numpy.int64),
        numpy.array(loop_uv_indices, dtype=numpy.int64),
    )


def import_obj(path, name=""):
    text = path.read_text()

    if not name:
        name_match = re.search(r"(?m)^[og] (.*)$", text)
        name = name_match.group(1).strip() if name_match else path.stem

    positions = obj_numeric_columns(text, "v", 3)
    uvs = obj_numeric_columns(text, "vt", 2)
    faces = parse_obj_faces(text)
    totals = faces.totals

    # bpy.ops.wm.obj_import changes selection, the active object and undo
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(positions))
    mesh.vertices.foreach_set("co", positions.ravel())
    mesh.loops.add(len(faces.loop_vertices))
    mesh.loops.foreach_set("vertex_index", faces.loop_vertices)
    mesh.polygons.add(len(totals))
    mesh.polygons.foreach_set("loop_start", face_starts(totals))
    mesh.polygons.foreach_set("loop_total", totals)
    # some meshes have no sharp_face attribute
    mesh.polygons.foreach_set("use_smooth", numpy.zeros(len(totals), dtype=bool))
    if len(uvs):
        layer = mesh.uv_layers.new()
        layer.data.foreach_set("uv", uvs[faces.loop_uv_indices].ravel())
    mesh.update(calc_edges=True)
    mesh.validate()

    imported_obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(imported_obj)
    return imported_obj


def print_stdin(process, line):
    if process.poll() is not None:
        return False
    try:
        print(line, file=process.stdin, flush=True)
    except OSError:
        return False
    return True
