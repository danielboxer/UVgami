"""Mesh primitives the rest of the package shares: edge maps, turn
angles, island grouping, and uv fitting. No seam logic."""

import collections
import math


# the package's base angle in degrees: edges turning less than this read as
# flat. it is the partition angle, low on purpose, over-segmenting is what
# makes region width meaningful, and the merges reassemble the pieces
LOW_ANGLE = 10


def cross(u, v):
    return [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def diagonal(verts):
    lo = [min(v[i] for v in verts) for i in range(3)]
    hi = [max(v[i] for v in verts) for i in range(3)]
    return norm([hi[i] - lo[i] for i in range(3)])


def pair(a, b):
    return (a, b) if a < b else (b, a)


def face_keys(face):
    """A face's edges as sorted vertex index pairs."""
    return [pair(face[i], face[(i + 1) % len(face)]) for i in range(len(face))]


def face_edges(faces):
    """Edge -> owning faces, keyed by sorted vertex index pair."""
    edges = collections.defaultdict(list)
    for fi, face in enumerate(faces):
        for key in face_keys(face):
            edges[key].append(fi)
    return edges


def turn_angle(weighted, owners):
    """Degrees the surface turns across an edge, from its two face normals."""
    na, nb = weighted[owners[0]], weighted[owners[1]]
    scale = norm(na) * norm(nb)
    if not scale:
        return 0.0
    dot = sum(x * y for x, y in zip(na, nb)) / scale
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def build(verts, faces):
    """Per-face weighted normals and areas, plus edge -> owning faces."""
    weighted, areas = [], []
    for face in faces:
        a, b, c = (verts[i] for i in face[:3])
        n = cross([b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)])
        weighted.append(n)
        areas.append(norm(n) / 2)
    return weighted, areas, face_edges(faces)


def signed_area(pts):
    total = 0.0
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        total += a[0] * b[1] - b[0] * a[1]
    return total / 2


def island_groups(faces, seams, edges):
    """Faces grouped into uv islands: joined by interior edges not on a seam."""
    parent = list(range(len(faces)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for key, owners in edges.items():
        if len(owners) == 2 and key not in seams:
            a, b = find(owners[0]), find(owners[1])
            if a != b:
                parent[a] = b

    members = collections.defaultdict(list)
    for fi in range(len(faces)):
        members[find(fi)].append(fi)
    return list(members.values())


def uv_island_groups(faces, uvs, edges):
    """Faces grouped into uv islands: joined by interior edges whose corner
    uvs agree on both faces, so the grouping follows the uv map itself and
    needs no seam marks."""
    parent = list(range(len(faces)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def corner_uv(f, v):
        return uvs[f][faces[f].index(v)]

    for (u, v), owners in edges.items():
        if len(owners) != 2:
            continue
        f, g = owners
        if corner_uv(f, u) == corner_uv(g, u) and corner_uv(f, v) == corner_uv(g, v):
            a, b = find(f), find(g)
            if a != b:
                parent[a] = b

    members = collections.defaultdict(list)
    for fi in range(len(faces)):
        members[find(fi)].append(fi)
    return list(members.values())


def vertex_components(faces):
    """Faces grouped into loose parts: joined by any shared vertex, the same
    connectivity mesh.separate(type="LOOSE") splits on."""
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for face in faces:
        for v in face:
            parent.setdefault(v, v)
        for v in face[1:]:
            ra, rb = find(face[0]), find(v)
            if ra != rb:
                parent[ra] = rb

    members = collections.defaultdict(list)
    for fi, face in enumerate(faces):
        members[find(face[0])].append(fi)
    return list(members.values())


def uv_fit(points, bbox):
    """Mapping that scales the points uniformly into the bbox, centered.
    Keeps a repaired island inside the spot its old layout occupied."""
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


def uv_area_fit(polygons, area, bbox):
    """Mapping that scales the polygons to cover the uv area the island had,
    centered on its old bbox. Keeps the island's texel density, which a bbox
    fit loses whenever the new layout packs to a different shape."""
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
