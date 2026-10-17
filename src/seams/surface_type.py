import math

import numpy

TINY_AREA_SHARE = 0.01
TINY_FACES = 50
# normals around a face within about 0.01 degrees of each other only happen on cad meshes
FLAT_SPREAD = 2e-4
FLAT_SHARE = 0.03
CURVE_RATIO = 0.25
# neighbourhoods stop at edges turning more than this
RING_CREASE = math.radians(30)
MIRROR_SCORE = 0.8
# a mirrored sample this close to the surface, as a share of the part's diagonal, matches
MATCH_TOLERANCE = 0.01
SURFACE_SAMPLES = 40000
QUERY_SAMPLES = 4000
MINIMUM_QUERIES = 300
# mirror plane normals are tried this far apart around each principal axis
PLANE_STEP = math.radians(15)


def _triangles(faces):
    used = sorted({v for face in faces for v in face})
    index = {v: i for i, v in enumerate(used)}
    triangles = numpy.array(
        [
            [index[face[0]], index[face[i]], index[face[i + 1]]]
            for face in faces
            for i in range(1, len(face) - 1)
        ],
        dtype=numpy.int64,
    ).reshape(-1, 3)
    return used, triangles


def _neighbours(triangles, normals):
    count = len(triangles)
    edges = numpy.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    keys = edges[:, 0] * (int(triangles.max()) + 1) + edges[:, 1]
    owners = numpy.repeat(numpy.arange(count, dtype=numpy.int64), 3)
    order = numpy.argsort(keys, kind="stable")
    keys, owners = keys[order], owners[order]
    starts = numpy.flatnonzero(numpy.diff(keys, prepend=-1))
    sizes = numpy.diff(numpy.append(starts, len(keys)))
    first = starts[sizes == 2]
    a, b = owners[first], owners[first + 1]
    smooth = numpy.einsum("ij,ij->i", normals[a], normals[b]) > math.cos(RING_CREASE)
    a, b = a[smooth], b[smooth]
    table = numpy.full((count + 1, 3), count, dtype=numpy.int64)
    sources = numpy.concatenate([a, b])
    targets = numpy.concatenate([b, a])
    order = numpy.argsort(sources, kind="stable")
    sources, targets = sources[order], targets[order]
    group_start = numpy.flatnonzero(numpy.diff(sources, prepend=-1))
    slot = numpy.arange(len(sources)) - numpy.repeat(
        group_start, numpy.diff(numpy.append(group_start, len(sources)))
    )
    table[sources, slot] = targets
    return table


def _two_rings(table):
    count = len(table) - 1
    own = numpy.arange(count, dtype=numpy.int64)[:, None]
    ring = numpy.concatenate(
        [own, table[:count], table[table[:count]].reshape(count, 9)], axis=1
    )
    ring.sort(axis=1)
    repeated = numpy.zeros(ring.shape, dtype=bool)
    repeated[:, 1:] = ring[:, 1:] == ring[:, :-1]
    return ring, (ring < count) & ~repeated


def _normal_spread(normals, areas, ring, kept):
    padded_areas = numpy.append(areas, 0.0)
    padded_normals = numpy.vstack([normals, numpy.zeros(3)])
    weight = numpy.zeros(len(areas))
    first_moment = numpy.zeros((len(areas), 3))
    second_moment = numpy.zeros((len(areas), 3, 3))
    for column in range(ring.shape[1]):
        member = ring[:, column]
        area = padded_areas[member] * kept[:, column]
        normal = padded_normals[member]
        weight += area
        first_moment += normal * area[:, None]
        second_moment += (
            numpy.einsum("fi,fj->fij", normal, normal) * area[:, None, None]
        )
    mean = first_moment / weight[:, None]
    covariance = second_moment / weight[:, None, None] - numpy.einsum(
        "fi,fj->fij", mean, mean
    )
    values = numpy.sqrt(numpy.maximum(numpy.linalg.eigvalsh(covariance), 0))
    return values[:, 2], values[:, 1]


def _weighted_median(values, weights):
    order = numpy.argsort(values)
    cumulative = numpy.cumsum(weights[order])
    return values[order][numpy.searchsorted(cumulative, cumulative[-1] / 2)]


def _sample_surface(corners, areas, count, rng):
    face = rng.choice(len(areas), count, p=areas / areas.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    picked = corners[face]
    return (
        picked[:, 0]
        + u[:, None] * (picked[:, 1] - picked[:, 0])
        + v[:, None] * (picked[:, 2] - picked[:, 0])
    )


def _plane_normals(axes):
    normals = []
    for spin in range(3):
        a, b = axes[(spin + 1) % 3], axes[(spin + 2) % 3]
        for angle in numpy.arange(0, math.pi, PLANE_STEP):
            normal = math.cos(angle) * a + math.sin(angle) * b
            # the three principal planes each come up in two of the sweeps
            if all(abs(normal @ seen) < 0.999 for seen in normals):
                normals.append(normal)
    return normals


_NEIGHBOUR_OFFSETS = numpy.array(numpy.meshgrid(*[[-1, 0, 1]] * 3)).reshape(3, -1).T


class _OccupiedCells:
    def __init__(self, points, tolerance):
        self.cell = tolerance / 2
        self.low = points.min(axis=0) - 2 * self.cell
        self.size = numpy.ceil(
            (points.max(axis=0) + 2 * self.cell - self.low) / self.cell
        )
        self.size = self.size.astype(numpy.int64) + 1
        self.keys = numpy.unique(self._keys(self._cells(points)))
        self.neighbour_steps = self._keys(_NEIGHBOUR_OFFSETS)

    def _cells(self, points):
        return numpy.floor((points - self.low) / self.cell).astype(numpy.int64)

    def _keys(self, cells):
        return cells[..., 0] + self.size[0] * (
            cells[..., 1] + self.size[1] * cells[..., 2]
        )

    def matched_shares(self, points):
        cells = self._cells(points)
        inside = numpy.all((cells >= 1) & (cells < self.size - 1), axis=-1)
        keys = self._keys(cells)[..., None] + self.neighbour_steps
        found = numpy.searchsorted(self.keys, keys)
        found = numpy.minimum(found, len(self.keys) - 1)
        matched = (self.keys[found] == keys).any(axis=-1) & inside
        return matched.mean(axis=1)


def _second_mirror_score(corners, areas, centroids, face_count):
    rng = numpy.random.default_rng(0)
    centre = (centroids * areas[:, None]).sum(axis=0) / areas.sum()
    offset = centroids - centre
    _, axes = numpy.linalg.eigh((offset * areas[:, None]).T @ offset / areas.sum())
    surface = _sample_surface(corners, areas, SURFACE_SAMPLES, rng) - centre
    queries = min(QUERY_SAMPLES, max(MINIMUM_QUERIES, 3 * face_count))
    points = _sample_surface(corners, areas, queries, rng) - centre
    diagonal = numpy.linalg.norm(centroids.max(axis=0) - centroids.min(axis=0))
    cells = _OccupiedCells(surface, MATCH_TOLERANCE * diagonal)
    normals = numpy.array(_plane_normals(axes.T))
    mirrored = points - 2 * (points @ normals.T).T[:, :, None] * normals[:, None, :]
    return numpy.sort(cells.matched_shares(mirrored))[-2]


# hard means a crease mesh, a revolved or boxy part, or one too small to judge
def is_hard_surface(verts, faces, model_area=None):
    if len(faces) < TINY_FACES:
        return True
    used, triangles = _triangles(faces)
    corners = numpy.asarray([verts[v] for v in used], dtype=numpy.float64)[triangles]
    cross = numpy.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    length = numpy.linalg.norm(cross, axis=1)
    areas = length / 2
    total = areas.sum()
    if model_area and total / model_area < TINY_AREA_SHARE:
        return True
    keep = length > 0
    if not keep.any():
        return True
    triangles, corners = triangles[keep], corners[keep]
    areas, normals = areas[keep], cross[keep] / length[keep, None]
    ring, kept = _two_rings(_neighbours(triangles, normals))
    first, second = _normal_spread(normals, areas, ring, kept)
    flat = first < FLAT_SPREAD
    if areas[flat].sum() / total >= FLAT_SHARE:
        return True
    if not (~flat).any():
        return True
    ratio = second[~flat] / first[~flat]
    if _weighted_median(ratio, areas[~flat]) < CURVE_RATIO:
        return True
    centroids = corners.mean(axis=1)
    return (
        _second_mirror_score(corners, areas, centroids, len(triangles)) >= MIRROR_SCORE
    )
