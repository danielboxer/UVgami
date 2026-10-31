import collections
import itertools

import numpy


def sorted_pair(a, b):
    return (a, b) if a < b else (b, a)


def split_per_face(values, totals):
    faces = []
    start = 0
    for total in totals:
        faces.append(values[start : start + total])
        start += total
    return faces


def faces_from_arrays(values, totals):
    # faces cross to a worker as flat arrays, which pickle far faster than per-face lists
    return split_per_face(values.tolist(), totals.tolist())


def find_root(parent, element):
    while parent[element] != element:
        parent[element] = parent[parent[element]]
        element = parent[element]
    return element


def faces_by_edge(faces):
    edges = collections.defaultdict(list)
    for face_index, face in enumerate(faces):
        for i in range(len(face)):
            edges[sorted_pair(face[i], face[(i + 1) % len(face)])].append(face_index)
    return edges


def signed_area(points):
    total = 0.0
    for i in range(len(points)):
        a, b = points[i], points[(i + 1) % len(points)]
        total += a[0] * b[1] - b[0] * a[1]
    return total / 2


def fan_areas(points):
    return [
        signed_area([points[0], points[i], points[i + 1]])
        for i in range(1, len(points) - 1)
    ]


# above a collapsed map's float noise and below any real packed map's area
COLLAPSED_UV_AREA = 1e-8


def uvs_collapsed_to_points(polygons):
    return sum(abs(signed_area(polygon)) for polygon in polygons) < COLLAPSED_UV_AREA


def islands_between_seams(faces, seams, edges):
    parent = list(range(len(faces)))

    for key, owners in edges.items():
        if len(owners) == 2 and key not in seams:
            a, b = find_root(parent, owners[0]), find_root(parent, owners[1])
            if a != b:
                parent[a] = b

    members = collections.defaultdict(list)
    for face_index in range(len(faces)):
        members[find_root(parent, face_index)].append(face_index)
    return list(members.values())


def undirected_edge_keys(tail, head):
    tail = numpy.asarray(tail, dtype=numpy.int64)
    head = numpy.asarray(head, dtype=numpy.int64)
    return (numpy.minimum(tail, head) << 32) | numpy.maximum(tail, head)


def face_starts(totals):
    return numpy.cumsum(totals) - totals


def next_corners_in_face(face_sizes):
    sizes = numpy.asarray(face_sizes, dtype=numpy.int64)
    starts = face_starts(sizes)
    face_of = numpy.repeat(numpy.arange(len(sizes)), sizes)
    local = numpy.arange(len(face_of)) - starts[face_of]
    return starts[face_of] + (local + 1) % sizes[face_of]


def corners_of_faces(starts, sizes, faces):
    kept_sizes = sizes[faces]
    kept_starts = face_starts(kept_sizes)
    return numpy.repeat(starts[faces] - kept_starts, kept_sizes) + numpy.arange(
        int(kept_sizes.sum())
    )


def lowest_index_part_labels(count, edges):
    # int64, keys built from these overflow a windows int32
    label = numpy.arange(count, dtype=numpy.int64)
    a = edges[:, 0]
    b = edges[:, 1]
    while a.size:
        # minimum.at is far slower than sorting for the reduceat
        targets = numpy.concatenate([a, b])
        sources = numpy.concatenate([b, a])
        order = numpy.argsort(targets, kind="stable")
        targets = targets[order]
        sources = sources[order]
        starts = numpy.flatnonzero(numpy.diff(targets, prepend=-1))
        grouped = targets[starts]
        low = label.copy()
        low[grouped] = numpy.minimum(
            low[grouped], numpy.minimum.reduceat(label[sources], starts)
        )
        while True:
            jumped = low[low]
            if numpy.array_equal(jumped, low):
                break
            low = jumped
        label = low
        a = label[a]
        b = label[b]
        keep = a != b
        a = a[keep]
        b = b[keep]
    return label


def edge_tears(tail_vertices, head_vertices, tail_uvs, head_uvs, tolerance):
    low_first = (tail_vertices < head_vertices)[:, None]
    at_low = numpy.where(low_first, tail_uvs, head_uvs)
    at_high = numpy.where(low_first, head_uvs, tail_uvs)
    keys = undirected_edge_keys(tail_vertices, head_vertices)
    unique, first, group = numpy.unique(keys, return_index=True, return_inverse=True)
    agrees = numpy.all(numpy.abs(at_low - at_low[first][group]) <= tolerance, axis=1)
    agrees &= numpy.all(numpy.abs(at_high - at_high[first][group]) <= tolerance, axis=1)
    torn = unique[numpy.unique(group[~agrees])]
    return {(int(key >> 32), int(key & 0xFFFFFFFF)) for key in torn.tolist()}


def torn_interior_edges(faces, corner_uvs, tolerance=0.0):
    sizes = [len(face) for face in faces]
    corners = numpy.fromiter(
        (v for face in faces for v in face), dtype=numpy.int64, count=sum(sizes)
    )
    uvs = numpy.array(
        [uv for face in corner_uvs for uv in face], dtype=numpy.float64
    ).reshape(-1, 2)
    following = next_corners_in_face(sizes)
    return edge_tears(corners, corners[following], uvs, uvs[following], tolerance)


def islands_from_uvs(faces, uvs, edges):
    return islands_between_seams(faces, torn_interior_edges(faces, uvs), edges)


def boxes_overlap(boxes):
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    for k, i in enumerate(order):
        for j in order[k + 1 :]:
            if boxes[j][0] >= boxes[i][2]:
                break
            if boxes[j][1] < boxes[i][3] and boxes[i][1] < boxes[j][3]:
                return True
    return False


IslandMove = collections.namedtuple("IslandMove", "mirror_sum shift_u shift_v")


def island_layout(boxes, areas):
    gap = 0.05 * max(x1 - x0 for x0, _, x1, _ in boxes)
    transforms = []
    cursor = 0.0
    for (x0, y0, x1, _), area in zip(boxes, areas):
        mirror_sum = x0 + x1 if area < 0 else None
        transforms.append(IslandMove(mirror_sum, cursor - x0, -y0))
        cursor += x1 - x0 + gap
    return transforms


def moved_island_uvs(loop_uvs, island_faces, moves, totals):
    face_count = len(totals)
    flips = numpy.full(face_count, numpy.nan)
    offsets = numpy.empty((face_count, 2))
    for faces, move in zip(island_faces, moves):
        if move.mirror_sum is not None:
            flips[faces] = move.mirror_sum
        offsets[faces] = (move.shift_u, move.shift_v)
    loop_flip = numpy.repeat(flips, totals)
    mirrored = ~numpy.isnan(loop_flip)
    moved = loop_uvs.copy()
    moved[mirrored, 0] = loop_flip[mirrored] - moved[mirrored, 0]
    return moved + numpy.repeat(offsets, totals, axis=0)


def scale_into_box_center(points, box, scale):
    us = [u for u, _ in points]
    vs = [v for _, v in points]
    center_u, center_v = (min(us) + max(us)) / 2, (min(vs) + max(vs)) / 2
    box_u, box_v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return lambda uv: (
        box_u + (uv[0] - center_u) * scale,
        box_v + (uv[1] - center_v) * scale,
    )


def fit_uvs_into_box(points, box):
    us = [u for u, _ in points]
    vs = [v for _, v in points]
    low_u, low_v, high_u, high_v = box
    width, height = max(us) - min(us), max(vs) - min(vs)
    scales = []
    if width > 0:
        scales.append((high_u - low_u) / width)
    if height > 0:
        scales.append((high_v - low_v) / height)
    scale = min(scales) if scales else 1.0
    return scale_into_box_center(points, box, scale)


def fit_uvs_keeping_area(polygons, area, box):
    new_area = sum(abs(signed_area(polygon)) for polygon in polygons)
    points = [uv for polygon in polygons for uv in polygon]
    if area <= 0 or new_area <= 0:
        return fit_uvs_into_box(points, box)
    return scale_into_box_center(points, box, (area / new_area) ** 0.5)


def transform_positions(positions, matrix):
    matrix = numpy.asarray(matrix)
    return positions @ matrix[:3, :3].T + matrix[:3, 3]


def bounding_box_diagonal(positions):
    return float(numpy.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))


NEIGHBOR_CELLS = [
    offsets
    for offsets in itertools.product((0, -1, 1), repeat=3)
    if offsets != (0, 0, 0)
]
