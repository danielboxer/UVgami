import collections
import itertools

import numpy

from .no_bpy.mesh import NEIGHBOR_CELLS, bounding_box_diagonal, transform_positions
from .no_bpy.mesh_reads import loop_vertices_and_face_sizes, vertex_position_array

MATCH_TOLERANCE = 1e-4


def rigid_fit(source, target):
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    left, _, right_transposed = numpy.linalg.svd(
        (source - source_center).T @ (target - target_center)
    )
    rotation = right_transposed.T @ left.T
    translation = target_center - rotation @ source_center
    moved = source @ rotation.T + translation
    error = numpy.sqrt(numpy.mean(numpy.sum((moved - target) ** 2, axis=1)))
    matrix = numpy.identity(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix, error


# duplicates coincide to float precision after pca alignment
MATCH_CELL = 1e-4

SPREAD_TOLERANCE = 1e-3

# pca axes in a plane are arbitrary when their eigenvalues are closer than this
DEGENERATE_GAP = 1e-2


class _Piece:
    def __init__(self, obj):
        mesh = obj.data
        self.obj = obj
        self.signature = (len(mesh.vertices), len(mesh.loops), len(mesh.polygons))
        self.corners, self.totals = loop_vertices_and_face_sizes(mesh)
        self.positions = transform_positions(
            vertex_position_array(mesh), obj.matrix_world
        )
        self.diagonal = bounding_box_diagonal(self.positions)
        centered = self.positions - self.positions.mean(axis=0)
        self.spread, axes = numpy.linalg.eigh(centered.T @ centered)
        self.local = centered @ axes
        self.grid = None


def _match_vertices(representative_local, grid, aligned, cell):
    permutation = numpy.empty(len(aligned), dtype=numpy.int64)
    used = numpy.zeros(len(representative_local), dtype=bool)
    keys = numpy.round(aligned / cell).astype(numpy.int64).tolist()
    for index, (x, y, z) in enumerate(keys):
        options = [i for i in grid.get((x, y, z), ()) if not used[i]]
        if not options:
            for dx, dy, dz in NEIGHBOR_CELLS:
                options += [
                    i for i in grid.get((x + dx, y + dy, z + dz), ()) if not used[i]
                ]
        if not options:
            return None
        distances = numpy.linalg.norm(
            representative_local[options] - aligned[index], axis=1
        )
        nearest = int(numpy.argmin(distances))
        if distances[nearest] > cell:
            return None
        permutation[index] = options[nearest]
        used[options[nearest]] = True
    return permutation


def _plane_alignments(representative, piece, plane, unique, cell):
    representative_radius = numpy.hypot(
        representative.local[:, plane[0]], representative.local[:, plane[1]]
    )
    anchor = int(representative_radius.argmax())
    anchor_angle = numpy.arctan2(
        representative.local[anchor, plane[1]], representative.local[anchor, plane[0]]
    )
    anchor_height = representative.local[anchor, unique]
    candidate_radius = numpy.hypot(piece.local[:, plane[0]], piece.local[:, plane[1]])
    for axis_sign in (1.0, -1.0):
        heights = piece.local[:, unique] * axis_sign
        anchors = numpy.nonzero(
            (numpy.abs(candidate_radius - representative_radius[anchor]) <= cell)
            & (numpy.abs(heights - anchor_height) <= cell)
        )[0]
        for mirror in (1.0, -1.0):
            for vertex in anchors.tolist():
                angle = numpy.arctan2(
                    piece.local[vertex, plane[1]] * mirror,
                    piece.local[vertex, plane[0]],
                )
                turn = anchor_angle - angle
                cosine, sine = numpy.cos(turn), numpy.sin(turn)
                first = piece.local[:, plane[0]]
                second = piece.local[:, plane[1]] * mirror
                aligned = piece.local.copy()
                aligned[:, unique] = heights
                aligned[:, plane[0]] = first * cosine - second * sine
                aligned[:, plane[1]] = first * sine + second * cosine
                yield aligned


def _alignments(representative, piece, cell):
    scale = representative.spread[2]
    low_gap = (representative.spread[1] - representative.spread[0]) / scale
    high_gap = (representative.spread[2] - representative.spread[1]) / scale
    if low_gap >= DEGENERATE_GAP and high_gap >= DEGENERATE_GAP:
        for signs in itertools.product((1.0, -1.0), repeat=3):
            yield piece.local * signs
    elif low_gap >= DEGENERATE_GAP or high_gap >= DEGENERATE_GAP:
        plane = (1, 2) if high_gap < DEGENERATE_GAP else (0, 1)
        unique = 0 if high_gap < DEGENERATE_GAP else 2
        yield from _plane_alignments(representative, piece, plane, unique, cell)
    else:
        for order in itertools.permutations((0, 1, 2)):
            for signs in itertools.product((1.0, -1.0), repeat=3):
                yield piece.local[:, order] * signs


def _face_keys(totals, corners):
    keys = []
    face_corners = corners.tolist()
    start = 0
    for total in totals.tolist():
        cycle = face_corners[start : start + total]
        keys.append(
            min(
                tuple(direction[shift:] + direction[:shift])
                for direction in (cycle, cycle[::-1])
                for shift in range(total)
            )
        )
        start += total
    return sorted(keys)


def _twin_matrix(representative, piece):
    tolerance = MATCH_TOLERANCE * piece.diagonal
    same_order = numpy.array_equal(
        representative.totals, piece.totals
    ) and numpy.array_equal(representative.corners, piece.corners)
    if same_order:
        matrix, error = rigid_fit(representative.positions, piece.positions)
        if error <= tolerance:
            return matrix, True
    spread_gap = numpy.abs(representative.spread - piece.spread).max()
    if spread_gap > SPREAD_TOLERANCE * piece.spread.max():
        return None
    if not numpy.array_equal(
        numpy.sort(representative.totals), numpy.sort(piece.totals)
    ):
        return None
    cell = MATCH_CELL * representative.diagonal
    if representative.grid is None:
        representative.grid = {}
        keys = numpy.round(representative.local / cell).astype(numpy.int64)
        for index, key in enumerate(map(tuple, keys.tolist())):
            representative.grid.setdefault(key, []).append(index)
    face_keys = _face_keys(piece.totals, piece.corners)
    for aligned in _alignments(representative, piece, cell):
        permutation = _match_vertices(
            representative.local, representative.grid, aligned, cell
        )
        if permutation is None:
            continue
        inverse = numpy.empty_like(permutation)
        inverse[permutation] = numpy.arange(len(permutation))
        moved_keys = _face_keys(representative.totals, inverse[representative.corners])
        if moved_keys != face_keys:
            continue
        matrix, error = rigid_fit(
            representative.positions[permutation], piece.positions
        )
        if error <= tolerance:
            return matrix, False
    return None


def _copyable_data(representative, obj):
    if list(representative.data.materials) != list(obj.data.materials):
        return False
    slots = {polygon.material_index for polygon in representative.data.polygons}
    other_slots = {polygon.material_index for polygon in obj.data.polygons}
    if len(slots) > 1 or slots != other_slots:
        return False
    smooth = {polygon.use_smooth for polygon in representative.data.polygons}
    other_smooth = {polygon.use_smooth for polygon in obj.data.polygons}
    return len(smooth) == 1 and smooth == other_smooth


Twin = collections.namedtuple("Twin", "representative matrix exact")


def find_twins(objects):
    representatives = {}
    twins = {}
    for obj in objects:
        piece = _Piece(obj)
        candidates = representatives.setdefault(piece.signature, [])
        for representative in candidates:
            match = _twin_matrix(representative, piece)
            if match is None:
                continue
            matrix, exact = match
            if not exact and not _copyable_data(representative.obj, obj):
                continue
            twins[obj] = Twin(representative.obj, matrix, exact)
            break
        else:
            candidates.append(piece)
    return twins


# with fewer decimals, distinct islands at nearly the same uvs count as one stack
STACK_DECIMALS = 9


def find_stacks(groups, uvs):
    by_key = {}
    for group in groups:
        key = tuple(
            sorted(
                (round(u, STACK_DECIMALS), round(v, STACK_DECIMALS))
                for face in group
                for u, v in uvs[face]
            )
        )
        by_key.setdefault(key, []).append(group)
    return [
        (members[0], members[1:]) for members in by_key.values() if len(members) > 1
    ]


def write_twin_output(source_path, target_path, matrix):
    rotation = numpy.array(matrix)[:3, :3]
    translation = numpy.array(matrix)[:3, 3]
    # reflections turn normals inward unless each face's winding is reversed
    mirrored = numpy.linalg.det(rotation) < 0
    lines = []
    for line in source_path.read_text().splitlines(keepends=True):
        if line.startswith("v "):
            position = rotation @ [float(x) for x in line.split()[1:4]] + translation
            lines.append("v {:.9f} {:.9f} {:.9f}\n".format(*position))
        elif mirrored and line.startswith("f "):
            tokens = line.split()
            lines.append(" ".join([tokens[0], *reversed(tokens[1:])]) + "\n")
        else:
            lines.append(line)
    target_path.write_text("".join(lines))
