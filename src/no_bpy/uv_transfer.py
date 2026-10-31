import math
from dataclasses import dataclass, field

import numpy

from .mesh import faces_from_arrays, find_root, signed_area, torn_interior_edges


@dataclass
class TransferPlan:
    loop_uvs: dict
    split_faces: dict
    seam_edges: set
    untouched_faces: set = field(default_factory=set)


AMBIGUOUS_GEOMETRY = "ambiguous_geometry"


class TransferError(Exception):
    def __init__(self, reason, detail):
        super().__init__(reason, detail)
        self.reason = reason
        self.detail = detail

    def __str__(self):
        return f"{self.reason}: {self.detail}"


def _grid_key(position, cells_per_unit):
    return (
        int(math.floor(position[0] * cells_per_unit)),
        int(math.floor(position[1] * cells_per_unit)),
        int(math.floor(position[2] * cells_per_unit)),
    )


def _query_grid(grid, cells_per_unit, positions, position, squared_tolerance):
    base = _grid_key(position, cells_per_unit)
    found = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                bucket = grid.get((base[0] + dx, base[1] + dy, base[2] + dz))
                if not bucket:
                    continue
                for i in bucket:
                    offset = positions[i] - position
                    if float(offset @ offset) <= squared_tolerance:
                        found.append(i)
    return found


def _flatten_part(assign, positions):
    relative = positions[assign] - positions[assign[0]]
    normal = numpy.zeros(3)
    for i in range(1, len(relative) - 1):
        normal += numpy.cross(relative[i], relative[i + 1])
    normal_length = numpy.linalg.norm(normal)
    axis_u = relative[1]
    axis_u_length = numpy.linalg.norm(axis_u)
    if normal_length == 0 or axis_u_length == 0:
        return None
    axis_u = axis_u / axis_u_length
    axis_v = numpy.cross(normal / normal_length, axis_u)
    return [
        complex(float(offset @ axis_u), float(offset @ axis_v)) for offset in relative
    ]


def _uv_map_of(flat, uvs):
    d1, d2 = flat[1] - flat[0], flat[2] - flat[0]
    u0, u1, u2 = (complex(*uv) for uv in uvs[:3])
    e1, e2 = u1 - u0, u2 - u0
    det = d1 * d2.conjugate() - d2 * d1.conjugate()
    if det == 0:
        return None
    p = (e1 * d2.conjugate() - e2 * d1.conjugate()) / det
    q = (d1 * e2 - d2 * e1) / det
    return p, q, u0 - p * flat[0] - q * flat[0].conjugate()


def _weld_parts(parts, positions):
    anchor = max(range(len(parts)), key=lambda i: abs(signed_area(parts[i][1])))
    anchor_assign, anchor_uvs = parts[anchor]
    anchor_flat = _flatten_part(anchor_assign, positions)
    if anchor_flat is None:
        return None
    uv_map = _uv_map_of(anchor_flat, anchor_uvs)
    if uv_map is None:
        return None
    anchor_area = abs(signed_area([(z.real, z.imag) for z in anchor_flat]))
    if anchor_area == 0:
        return None
    tolerance = 0.01 * math.sqrt(anchor_area)

    placed = dict(zip(anchor_assign, anchor_flat))
    pending = [i for i in range(len(parts)) if i != anchor]
    while pending:
        progress = False
        for i in list(pending):
            assign, _ = parts[i]
            shared = [v for v in assign if v in placed]
            if len(shared) < 2:
                continue
            flat = _flatten_part(assign, positions)
            if flat is None:
                return None
            a, b = shared[:2]
            za, zb = flat[assign.index(a)], flat[assign.index(b)]
            if za == zb:
                return None
            s = (placed[a] - placed[b]) / (za - zb)
            t = placed[a] - s * za

            moved = [(v, s * z + t) for v, z in zip(assign, flat)]
            if any(v in placed and abs(placed[v] - z) > tolerance for v, z in moved):
                return None
            for v, z in moved:
                placed.setdefault(v, z)
            pending.remove(i)
            progress = True
        if not progress:
            return None

    p, q, r = uv_map
    corner_uvs = {v: uv for v, uv in zip(anchor_assign, anchor_uvs)}
    for v, z in placed.items():
        if v not in corner_uvs:
            w = p * z + q * z.conjugate() + r
            corner_uvs[v] = (w.real, w.imag)
    return corner_uvs, anchor


def _is_ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _is_collinear(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1]) == 0


class _OutputLayout:
    def __init__(self, output_polygons, output_uvs):
        self.parent = list(range(len(output_polygons)))
        corner_face = {}
        self.edges = {}
        total = 0.0
        count = 0
        for output_face, (vertices, uvs) in enumerate(zip(output_polygons, output_uvs)):
            corners = [(float(uv[0]), float(uv[1])) for uv in uvs]
            for vertex, p in zip(vertices, corners):
                other = corner_face.setdefault((vertex, p), output_face)
                if other != output_face:
                    self.parent[find_root(self.parent, output_face)] = find_root(
                        self.parent, other
                    )
            corner_count = len(corners)
            for i in range(corner_count):
                p, q = corners[i], corners[(i + 1) % corner_count]
                self.edges.setdefault(p, []).append((output_face, q))
                self.edges.setdefault(q, []).append((output_face, p))
                total += math.hypot(q[0] - p[0], q[1] - p[1])
                count += 1
        self.cell = total / count if total > 0 else 1.0
        self.grid = {}
        for p in self.edges:
            key = (math.floor(p[0] / self.cell), math.floor(p[1] / self.cell))
            self.grid.setdefault(key, []).append(p)

    def points_near(self, low_u, low_v, high_u, high_v):
        found = []
        for cell_u in range(
            math.floor(low_u / self.cell), math.floor(high_u / self.cell) + 1
        ):
            for cell_v in range(
                math.floor(low_v / self.cell), math.floor(high_v / self.cell) + 1
            ):
                found.extend(self.grid.get((cell_u, cell_v), ()))
        return found


def _weld_overlaps(parts, part_output_faces, anchor, welded_uvs, layout, repack):
    own = set(part_output_faces)
    island = find_root(layout.parent, part_output_faces[anchor])

    moved = []
    for i, (assign, _) in enumerate(parts):
        if i == anchor:
            continue
        corners = [welded_uvs[v] for v in assign]
        corner_count = len(corners)
        moved.extend(
            (corners[j], corners[(j + 1) % corner_count]) for j in range(corner_count)
        )

    us = [p[0] for edge in moved for p in edge]
    vs = [p[1] for edge in moved for p in edge]
    for p in layout.points_near(
        min(us) - layout.cell,
        min(vs) - layout.cell,
        max(us) + layout.cell,
        max(vs) + layout.cell,
    ):
        for output_face, q in layout.edges[p]:
            if output_face in own:
                continue
            if repack and find_root(layout.parent, output_face) != island:
                continue
            for m1, m2 in moved:
                if _is_collinear(p, m1, m2) or _is_collinear(q, m1, m2):
                    continue
                crosses_moved = _is_ccw(p, m1, m2) != _is_ccw(q, m1, m2)
                if crosses_moved and _is_ccw(p, q, m1) != _is_ccw(p, q, m2):
                    return True
    return False


SEAM_TOLERANCE = 1e-6


def transfer_exact(
    input_positions,
    input_polygons,
    output_positions,
    output_polygons,
    output_uvs,
    repack=True,
    partial=False,
):
    input_array = numpy.asarray(input_positions, dtype=float)
    output_array = numpy.asarray(output_positions, dtype=float)

    # engine obj output has 7 significant digits
    tolerance = (
        max(float(numpy.abs(input_array).max()) * 2e-6, 1e-5)
        if len(input_array)
        else 1e-5
    )
    squared_tolerance = tolerance * tolerance
    cells_per_unit = 1.0 / tolerance

    input_grid = {}
    for i in range(len(input_array)):
        input_grid.setdefault(_grid_key(input_array[i], cells_per_unit), []).append(i)

    faces_by_vertex = [[] for _ in range(len(input_array))]
    input_face_vertex_sets = []
    input_vertex_local = []
    input_loop_start = []
    loop_cursor = 0
    for face_index, polygon in enumerate(input_polygons):
        input_loop_start.append(loop_cursor)
        loop_cursor += len(polygon)
        input_face_vertex_sets.append(set(polygon))
        local = {}
        for corner, vertex in enumerate(polygon):
            faces_by_vertex[vertex].append(face_index)
            local[vertex] = corner
        input_vertex_local.append(local)

    # optcuts and xatlas output the input vertex indices unchanged, partuv merges doubles
    input_count = len(input_array)
    keeps_indices = 0 < input_count <= len(output_array) and bool(
        (
            ((output_array[:input_count] - input_array) ** 2).sum(axis=1)
            <= squared_tolerance
        ).all()
    )
    candidate_cache = {}

    def candidates(output_vertex):
        if keeps_indices and output_vertex < input_count:
            return [output_vertex]
        cached = candidate_cache.get(output_vertex)
        if cached is None:
            cached = _query_grid(
                input_grid,
                cells_per_unit,
                input_array,
                output_array[output_vertex],
                squared_tolerance,
            )
            candidate_cache[output_vertex] = cached
        return cached

    face_parts = [[] for _ in input_polygons]
    part_output_faces = [[] for _ in input_polygons]

    for output_face, output_vertices in enumerate(output_polygons):
        corner_candidates = []
        for output_vertex in output_vertices:
            matching_vertices = candidates(output_vertex)
            if not matching_vertices:
                raise TransferError(
                    "vertex_match",
                    f"output vertex {output_vertex} has no matching input vertex",
                )
            corner_candidates.append(matching_vertices)

        candidate_faces = set()
        for vertex in corner_candidates[0]:
            candidate_faces.update(faces_by_vertex[vertex])
        for corner in corner_candidates[1:]:
            allowed = set()
            for vertex in corner:
                allowed.update(faces_by_vertex[vertex])
            candidate_faces &= allowed
            if not candidate_faces:
                break
        if not candidate_faces:
            raise TransferError(
                "face_match", f"output face {output_face} maps to no input face"
            )

        valid = []
        for face in candidate_faces:
            face_vertex_set = input_face_vertex_sets[face]
            assign = []
            for corner in corner_candidates:
                matches = [vertex for vertex in corner if vertex in face_vertex_set]
                if len(matches) > 1:
                    raise TransferError(
                        AMBIGUOUS_GEOMETRY,
                        f"output face {output_face} corner matches multiple vertices"
                        " of one input face",
                    )
                assign.append(matches[0])
            valid.append((face, assign))

        if len(valid) > 1:
            raise TransferError(
                AMBIGUOUS_GEOMETRY,
                f"output face {output_face} matches multiple input faces",
            )
        face, assign = valid[0]
        face_uvs = output_uvs[output_face]
        face_parts[face].append(
            (assign, [(float(uv[0]), float(uv[1])) for uv in face_uvs])
        )
        part_output_faces[face].append(output_face)

    layout = None
    loop_uvs = {}
    split_faces = {}
    untouched = set()
    for face_index, polygon in enumerate(input_polygons):
        parts = face_parts[face_index]
        if partial and not parts:
            untouched.add(face_index)
            continue
        covered = {vertex for assign, _ in parts for vertex in assign}
        if covered != set(polygon):
            raise TransferError(
                "incomplete_coverage", f"input face {face_index} has corners with no uv"
            )

        corner_uvs = {}
        conflict = False
        for assign, uvs in parts:
            for input_vertex, uv in zip(assign, uvs):
                previous = corner_uvs.get(input_vertex)
                if previous is None:
                    corner_uvs[input_vertex] = uv
                elif abs(previous[0] - uv[0]) > 1e-6 or abs(previous[1] - uv[1]) > 1e-6:
                    conflict = True

        local = input_vertex_local[face_index]
        if not conflict:
            base = input_loop_start[face_index]
            for input_vertex, uv in corner_uvs.items():
                loop_uvs[base + local[input_vertex]] = uv
            continue

        if len({frozenset(assign) for assign, _ in parts}) != len(parts):
            raise TransferError(
                AMBIGUOUS_GEOMETRY,
                f"input face {face_index} maps to overlapping output faces",
            )

        welded = _weld_parts(parts, input_array)
        if welded is not None:
            welded_uvs, anchor = welded
            if layout is None:
                layout = _OutputLayout(output_polygons, output_uvs)
            if not _weld_overlaps(
                parts, part_output_faces[face_index], anchor, welded_uvs, layout, repack
            ):
                base = input_loop_start[face_index]
                for input_vertex, uv in welded_uvs.items():
                    loop_uvs[base + local[input_vertex]] = uv
                continue

        split_faces[face_index] = []
        for assign, uvs in parts:
            ordered = sorted(zip(assign, uvs), key=lambda pair: local[pair[0]])
            split_faces[face_index].append(
                ([vertex for vertex, _ in ordered], [uv for _, uv in ordered])
            )

    if len(untouched) == len(input_polygons):
        raise TransferError("incomplete_coverage", "no input face got uvs")

    faces = []
    for face_index, polygon in enumerate(input_polygons):
        if face_index in untouched:
            continue
        parts = split_faces.get(face_index)
        if parts is None:
            base = input_loop_start[face_index]
            faces.append(
                (polygon, [loop_uvs[base + corner] for corner in range(len(polygon))])
            )
        else:
            faces.extend(parts)
    seams = torn_interior_edges(
        [vertices for vertices, _ in faces], [uvs for _, uvs in faces], SEAM_TOLERANCE
    )
    return TransferPlan(loop_uvs, split_faces, seams, untouched)


def transfer_exact_job(
    input_positions,
    input_corners,
    input_totals,
    output_positions,
    output_corners,
    output_totals,
    output_loop_uvs,
    **options,
):
    return transfer_exact(
        input_positions,
        faces_from_arrays(input_corners, input_totals),
        output_positions,
        faces_from_arrays(output_corners, output_totals),
        faces_from_arrays(output_loop_uvs, output_totals),
        **options,
    )
