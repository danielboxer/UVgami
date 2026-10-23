import collections
import itertools
import math

import numpy

# low on purpose: a bevel splits one crease into several small turns
LOW_ANGLE = 10


def cross(u, v):
    return [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]


def norm(v):
    x, y, z = v
    return math.sqrt(x * x + y * y + z * z)


def pair(a, b):
    return (a, b) if a < b else (b, a)


# each polygon covers a contiguous run of loops, so the totals alone place it
def split_per_face(values, totals):
    faces = []
    start = 0
    for total in totals:
        faces.append(values[start : start + total])
        start += total
    return faces


# arrays pickle far faster than the per-face lists
def faces_from_arrays(values, totals):
    return split_per_face(values.tolist(), totals.tolist())


# parent maps each element to its parent, itself for a root
def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


# a face's edges as sorted vertex index pairs
def face_keys(face):
    return [pair(face[i], face[(i + 1) % len(face)]) for i in range(len(face))]


# edge -> owning faces, keyed by sorted vertex index pair
def face_edges(faces):
    edges = collections.defaultdict(list)
    for fi, face in enumerate(faces):
        for key in face_keys(face):
            edges[key].append(fi)
    return edges


# degrees the surface turns across an edge
def turn_angle(weighted, owners):
    ax, ay, az = weighted[owners[0]]
    bx, by, bz = weighted[owners[1]]
    # two roots, not one over the product: folding them flips the odd seam
    scale = math.sqrt(ax * ax + ay * ay + az * az) * math.sqrt(
        bx * bx + by * by + bz * bz
    )
    if not scale:
        return 0.0
    dot = (ax * bx + ay * by + az * bz) / scale
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


# scaled by twice the area of the first three corners
def weighted_normals(verts, faces):
    weighted = []
    for face in faces:
        a, b, c = (verts[i] for i in face[:3])
        weighted.append(
            cross([b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)])
        )
    return weighted


# by fan, the same fan the flatten solves
def polygon_area(verts, face):
    x0, y0, z0 = verts[face[0]]
    total = 0.0
    for i in range(1, len(face) - 1):
        ax, ay, az = verts[face[i]]
        bx, by, bz = verts[face[i + 1]]
        ux, uy, uz = ax - x0, ay - y0, az - z0
        vx, vy, vz = bx - x0, by - y0, bz - z0
        cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        total += math.sqrt(cx * cx + cy * cy + cz * cz) / 2.0
    return total


def signed_area(pts):
    total = 0.0
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        total += a[0] * b[1] - b[0] * a[1]
    return total / 2


# between a collapsed map's float noise and any real packed map's area
COLLAPSED_UV_AREA = 1e-8


# a uv map crushed to points, a failed flatten's signature
def uvs_collapsed(polygons):
    return sum(abs(signed_area(pts)) for pts in polygons) < COLLAPSED_UV_AREA


# joined by interior edges not on a seam
def island_groups(faces, seams, edges):
    parent = list(range(len(faces)))

    for key, owners in edges.items():
        if len(owners) == 2 and key not in seams:
            a, b = find(parent, owners[0]), find(parent, owners[1])
            if a != b:
                parent[a] = b

    members = collections.defaultdict(list)
    for fi in range(len(faces)):
        members[find(parent, fi)].append(fi)
    return list(members.values())


# one integer per undirected edge, the same for both directions
def edge_keys(tail, head):
    tail = numpy.asarray(tail, dtype=numpy.int64)
    head = numpy.asarray(head, dtype=numpy.int64)
    return (numpy.minimum(tail, head) << 32) | numpy.maximum(tail, head)


# faces are laid out one after another
def following_corners(face_sizes):
    sizes = numpy.asarray(face_sizes, dtype=numpy.int64)
    starts = numpy.cumsum(sizes) - sizes
    face_of = numpy.repeat(numpy.arange(len(sizes)), sizes)
    local = numpy.arange(len(face_of)) - starts[face_of]
    return starts[face_of] + (local + 1) % sizes[face_of]


# the corner starting the same edge in the other face, -1 on a boundary or non-manifold edge
def edge_twins(corners, following):
    keys = edge_keys(corners, corners[following])
    order = numpy.argsort(keys, kind="stable")
    _, first, counts = numpy.unique(keys[order], return_index=True, return_counts=True)
    paired = first[counts == 2]
    twin = numpy.full(len(corners), -1, dtype=numpy.int64)
    a, b = order[paired], order[paired + 1]
    twin[a] = b
    twin[b] = a
    return twin


# the ranges starts[i] to starts[i] + sizes[i], one after another
def index_ranges(starts, sizes):
    offsets = numpy.cumsum(sizes) - sizes
    return numpy.repeat(starts - offsets, sizes) + numpy.arange(int(sizes.sum()))


# a part's label is its lowest index, never a dense range
def connected_labels(count, edges):
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
        # low[low] shortcuts label chains
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


# edges given as corner rows, the vertex and uv at each end
def edge_tears(tail, head, tail_uv, head_uv, tolerance):
    low_first = (tail < head)[:, None]
    at_low = numpy.where(low_first, tail_uv, head_uv)
    at_high = numpy.where(low_first, head_uv, tail_uv)
    keys = edge_keys(tail, head)
    unique, first, group = numpy.unique(keys, return_index=True, return_inverse=True)
    agrees = numpy.all(numpy.abs(at_low - at_low[first][group]) <= tolerance, axis=1)
    agrees &= numpy.all(numpy.abs(at_high - at_high[first][group]) <= tolerance, axis=1)
    torn = unique[numpy.unique(group[~agrees])]
    return {(int(key >> 32), int(key & 0xFFFFFFFF)) for key in torn.tolist()}


# (low, high) pairs of edges whose faces don't share their corner uvs, boundary edges never count
def uv_tears(faces, corner_uvs, tolerance=0.0):
    sizes = [len(face) for face in faces]
    corners = numpy.fromiter(
        (v for face in faces for v in face), dtype=numpy.int64, count=sum(sizes)
    )
    uvs = numpy.array(
        [uv for face in corner_uvs for uv in face], dtype=numpy.float64
    ).reshape(-1, 2)
    following = following_corners(sizes)
    return edge_tears(corners, corners[following], uvs, uvs[following], tolerance)


# follows the uv map itself, so it needs no seam marks
def uv_island_groups(faces, uvs, edges):
    return island_groups(faces, uv_tears(faces, uvs), edges)


# boxes can touch without the boundaries crossing
def islands_overlap(boxes):
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    for k, i in enumerate(order):
        for j in order[k + 1 :]:
            if boxes[j][0] >= boxes[i][2]:
                break
            if boxes[j][1] < boxes[i][3] and boxes[i][1] < boxes[j][3]:
                return True
    return False


# applied as u -> flip - u when flip is not None, then add (du, dv)
def island_layout(boxes, areas):
    gap = 0.05 * max(x1 - x0 for x0, _, x1, _ in boxes)
    transforms = []
    cursor = 0.0
    for (x0, y0, x1, _), area in zip(boxes, areas):
        flip = x0 + x1 if area < 0 else None
        transforms.append((flip, cursor - x0, -y0))
        cursor += x1 - x0 + gap
    return transforms


# keeps a repaired island inside the spot its old layout occupied
def uv_fit(points, bbox):
    xs = [u for u, _ in points]
    ys = [v for _, v in points]
    x0, y0, x1, y1 = bbox
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    scales = []
    if w > 0:
        scales.append((x1 - x0) / w)
    if h > 0:
        scales.append((y1 - y0) / h)
    s = min(scales) if scales else 1.0
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    ox, oy = (x0 + x1) / 2, (y0 + y1) / 2
    return lambda uv: (ox + (uv[0] - cx) * s, oy + (uv[1] - cy) * s)


# keeps the island's texel density, which uv_fit loses
def uv_area_fit(polygons, area, bbox):
    new_area = sum(abs(signed_area(p)) for p in polygons)
    points = [uv for p in polygons for uv in p]
    if area <= 0 or new_area <= 0:
        return uv_fit(points, bbox)
    s = (area / new_area) ** 0.5
    xs = [u for u, _ in points]
    ys = [v for _, v in points]
    x0, y0, x1, y1 = bbox
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    ox, oy = (x0 + x1) / 2, (y0 + y1) / 2
    return lambda uv: (ox + (uv[0] - cx) * s, oy + (uv[1] - cy) * s)


# positions moved by a 4x4 matrix
def transformed(positions, matrix):
    matrix = numpy.asarray(matrix)
    return positions @ matrix[:3, :3].T + matrix[:3, 3]


# the 26 cells around a grid cell
NEIGHBOR_CELLS = [
    offsets
    for offsets in itertools.product((0, -1, 1), repeat=3)
    if offsets != (0, 0, 0)
]
