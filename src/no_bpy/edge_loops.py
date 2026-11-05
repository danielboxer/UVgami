import math
from collections import namedtuple

import numpy

from .mesh import (
    face_starts,
    faces_from_arrays,
    lowest_index_part_labels,
    next_corners_in_face,
    undirected_edge_keys,
)

Strip = namedtuple("Strip", "faces crossed_edges")
Slice = namedtuple("Slice", "strip plane_distance length")
COARSE_TILT_DEGREES = range(0, 180, 30)
FINE_TILT_DEGREES = 10


def _unit(vector):
    length = numpy.linalg.norm(vector)
    return vector / length if length > 0 else vector


def _unit_rows(vectors):
    lengths = numpy.linalg.norm(vectors, axis=1, keepdims=True)
    return numpy.divide(
        vectors, lengths, out=numpy.zeros_like(vectors), where=lengths > 0
    )


def _faces_of_edges(edge_of_corner, face_of_corner, face_counts):
    by_edge = numpy.argsort(edge_of_corner, kind="stable")
    group_starts = face_starts(face_counts)
    first_faces = face_of_corner[by_edge[group_starts]]
    second_faces = numpy.where(
        face_counts > 1,
        face_of_corner[by_edge[numpy.minimum(group_starts + 1, len(by_edge) - 1)]],
        -1,
    )
    return first_faces, second_faces


class EdgeLoops:
    def __init__(self, positions, corners, totals):
        positions = numpy.asarray(positions, dtype=numpy.float64)
        corners = numpy.asarray(corners, dtype=numpy.int64)
        totals = numpy.asarray(totals, dtype=numpy.int64)
        vertex_count = len(positions)
        heads = corners[next_corners_in_face(totals)]
        face_of_corner = numpy.repeat(numpy.arange(len(totals)), totals)
        self.positions = positions
        self.corners = corners
        self.totals = totals
        self.face_of_corner = face_of_corner

        keys, edge_of_corner, face_counts = numpy.unique(
            undirected_edge_keys(corners, heads),
            return_inverse=True,
            return_counts=True,
        )
        edges = numpy.stack([keys >> 32, keys & 0xFFFFFFFF], axis=1)
        first_faces, second_faces = _faces_of_edges(
            edge_of_corner, face_of_corner, face_counts
        )

        edge_count = len(edges)
        sources = numpy.concatenate([edges[:, 0], edges[:, 1]])
        targets = numpy.concatenate([edges[:, 1], edges[:, 0]])
        order = numpy.argsort(sources, kind="stable")
        sources, targets = sources[order], targets[order]
        entry_edges = numpy.concatenate([numpy.arange(edge_count)] * 2)[order]

        face_normals = numpy.zeros((len(totals), 3))
        numpy.add.at(
            face_normals,
            face_of_corner,
            numpy.cross(positions[corners], positions[heads]),
        )
        # the summed corner cross products of a planar polygon are twice its area
        self.face_areas = numpy.linalg.norm(face_normals, axis=1) / 2
        vertex_normals = numpy.zeros((vertex_count, 3))
        numpy.add.at(vertex_normals, corners, face_normals[face_of_corner])
        self.vertex_normals = _unit_rows(vertex_normals)

        boundary = numpy.zeros(vertex_count, dtype=bool)
        boundary[edges[face_counts != 2].ravel()] = True
        self.boundary = boundary
        not_quad = numpy.bincount(
            corners[totals[face_of_corner] != 4], minlength=vertex_count
        )
        valence = numpy.bincount(sources, minlength=vertex_count)

        self._start = numpy.searchsorted(
            sources, numpy.arange(vertex_count + 1)
        ).tolist()
        self._target = targets.tolist()
        self._faces = list(
            zip(first_faces[entry_edges].tolist(), second_faces[entry_edges].tolist())
        )
        self._quad = ((valence == 4) & ~boundary & (not_quad == 0)).tolist()

    def _entry(self, tail, head):
        for entry in range(self._start[tail], self._start[tail + 1]):
            if self._target[entry] == head:
                return entry
        raise ValueError(f"no edge between vertices {tail} and {head}")

    def _opposite(self, entry):
        vertex = self._target[entry]
        arriving_faces = set(self._faces[entry])
        for candidate in range(self._start[vertex], self._start[vertex + 1]):
            if arriving_faces.isdisjoint(self._faces[candidate]):
                return candidate
        raise ValueError(f"vertex {vertex} has no edge across from the arriving one")

    def _quad_walk(self, allowed, tail, head, walked):
        entry = self._entry(tail, head)
        vertices = [tail, head]
        while self._quad[self._target[entry]]:
            entry = self._opposite(entry)
            vertex = self._target[entry]
            if vertex == tail:
                return vertices, True
            stop = (
                vertex in vertices
                or not allowed[vertex]
                or (min(vertices[-1], vertex), max(vertices[-1], vertex)) in walked
            )
            if stop:
                return vertices, False
            vertices.append(vertex)
        return vertices, False

    def quad_loops(self, region):
        allowed = region.interior
        walked = set()
        loops = []
        for tail, head in region.vertices[region.edges].tolist():
            if not (allowed[tail] and allowed[head]) or (tail, head) in walked:
                continue
            vertices, closed = self._quad_walk(allowed, tail, head, walked)
            following = vertices[1:] + vertices[:1] if closed else vertices[1:]
            walked.update(
                (min(start, end), max(start, end))
                for start, end in zip(vertices, following)
            )
            if closed:
                loops.append(vertices)
        return loops

    def sliced_loops(self, region):
        tails, heads = region.vertices[region.edges].T
        covered = numpy.zeros(len(region.edges), dtype=bool)
        loops = []
        for index, (tail, head) in enumerate(zip(tails.tolist(), heads.tolist())):
            if covered[index] or not (region.interior[tail] and region.interior[head]):
                continue
            loop = self.find_loop(region, tail, head)
            if loop is not None:
                loops.append(loop)
            covered[region.edges_near(loop or [tail, head])] = True
        return loops

    def find_loop(self, region, tail, head):
        allowed = region.interior
        if not (allowed[tail] and allowed[head]):
            return None
        quad_loop, closed = self._quad_walk(allowed, tail, head, set())
        if closed:
            return quad_loop
        if region.mostly_quads:
            return None
        surface_normal = _unit(self.vertex_normals[tail] + self.vertex_normals[head])
        across = _unit(
            numpy.cross(self.positions[head] - self.positions[tail], surface_normal)
        )
        return region.sliced_loop(tail, across, numpy.cross(surface_normal, across))


class LoopRegion:
    def __init__(self, edge_loops, in_region, face_pairs):
        self.faces = numpy.flatnonzero(in_region)
        in_region_corners = in_region[edge_loops.face_of_corner]
        self.vertices, corners = numpy.unique(
            edge_loops.corners[in_region_corners], return_inverse=True
        )
        outside = numpy.bincount(
            edge_loops.corners[~in_region_corners],
            minlength=len(edge_loops.boundary),
        )
        interior = (outside == 0) & ~edge_loops.boundary
        self.interior = interior.tolist()
        self.local_interior = interior[self.vertices].tolist()
        self.positions = edge_loops.positions[self.vertices]
        self.position_list = self.positions.tolist()

        totals = edge_loops.totals[self.faces]
        self.mostly_quads = bool((totals == 4).mean() > 0.5)
        self.corners = corners
        next_corner_vertices = corners[next_corners_in_face(totals)]
        self.corner_faces = numpy.repeat(numpy.arange(len(self.faces)), totals)
        keys, corner_edges, face_counts = numpy.unique(
            undirected_edge_keys(corners, next_corner_vertices),
            return_inverse=True,
            return_counts=True,
        )
        self.edges = numpy.stack([keys >> 32, keys & 0xFFFFFFFF], axis=1)
        self.face_edges = faces_from_arrays(corner_edges, totals)
        first_faces, second_faces = _faces_of_edges(
            corner_edges, self.corner_faces, face_counts
        )
        self.edge_first_faces = first_faces.tolist()
        self.edge_second_faces = numpy.where(
            face_counts == 2, second_faces, -1
        ).tolist()
        self.face_vertices = faces_from_arrays(corners, totals)
        self.vertex_faces = [[] for _ in range(len(self.vertices))]
        for face, ring in enumerate(self.face_vertices):
            for vertex in ring:
                self.vertex_faces[vertex].append(face)

        inside = in_region[face_pairs.first] & in_region[face_pairs.second]
        self.local_face_pairs = numpy.stack(
            [
                numpy.searchsorted(self.faces, face_pairs.first[inside]),
                numpy.searchsorted(self.faces, face_pairs.second[inside]),
            ],
            axis=1,
        )
        self.pair_edge_keys = undirected_edge_keys(
            face_pairs.edges[inside, 0], face_pairs.edges[inside, 1]
        )

    def sliced_loop(self, tail, across, along):
        tail = int(numpy.searchsorted(self.vertices, tail))
        slices = {}
        for tilt in COARSE_TILT_DEGREES:
            slices[tilt] = self._slice(tail, across, along, tilt)
        coarse_best = min(slices, key=lambda tilt: slices[tilt].length)
        for tilt in (coarse_best - FINE_TILT_DEGREES, coarse_best + FINE_TILT_DEGREES):
            slices[tilt % 180] = self._slice(tail, across, along, tilt % 180)
        best = slices[min(slices, key=lambda tilt: slices[tilt].length)]
        if best.strip is None:
            return None
        loop = self._snapped_loop(tail, best.strip.faces, best.plane_distance)
        if loop is None:
            return None
        return self.vertices[loop].tolist()

    def _slice(self, tail, across, along, tilt_degrees):
        tilt = math.radians(tilt_degrees)
        direction = (across * math.cos(tilt) + along * math.sin(tilt)).tolist()
        origin = self.position_list[tail]
        distances = {tail: 0.0}

        def plane_distance(vertex):
            distance = distances.get(vertex)
            if distance is None:
                point = self.position_list[vertex]
                distance = distances[vertex] = (
                    (point[0] - origin[0]) * direction[0]
                    + (point[1] - origin[1]) * direction[1]
                    + (point[2] - origin[2]) * direction[2]
                )
            return distance

        strip = self._strip(tail, plane_distance)
        if strip is None:
            return Slice(strip, plane_distance, math.inf)
        tails, heads = self.edges[strip.crossed_edges].T
        tail_distances = numpy.array(
            [plane_distance(vertex) for vertex in tails.tolist()]
        )
        head_distances = numpy.array(
            [plane_distance(vertex) for vertex in heads.tolist()]
        )
        share = tail_distances / (tail_distances - head_distances)
        crossings = self.positions[tails] + share[:, None] * (
            self.positions[heads] - self.positions[tails]
        )
        length = numpy.linalg.norm(
            crossings - numpy.roll(crossings, -1, axis=0), axis=1
        ).sum()
        return Slice(strip, plane_distance, float(length))

    def _crossed_edges(self, face, plane_distance):
        above = [plane_distance(vertex) >= 0 for vertex in self.face_vertices[face]]
        return [
            edge
            for tail_above, head_above, edge in zip(
                above, above[1:] + above[:1], self.face_edges[face]
            )
            if tail_above != head_above
        ]

    def _strip(self, tail, plane_distance):
        for start_face in self.vertex_faces[tail]:
            crossed = self._crossed_edges(start_face, plane_distance)
            if len(crossed) == 2:
                break
        else:
            return None
        face, leaving = start_face, crossed[1]
        strip, strip_edges = [start_face], [leaving]
        while True:
            first, second = (
                self.edge_first_faces[leaving],
                self.edge_second_faces[leaving],
            )
            if second < 0:
                return None
            face = second if first == face else first
            if face == start_face:
                return Strip(strip, strip_edges)
            crossed = self._crossed_edges(face, plane_distance)
            if len(crossed) != 2:
                return None
            leaving = crossed[1] if crossed[0] == leaving else crossed[0]
            strip.append(face)
            strip_edges.append(leaving)

    def edges_near(self, loop):
        near = set()
        for vertex in numpy.searchsorted(self.vertices, loop).tolist():
            for face in self.vertex_faces[vertex]:
                near.update(self.face_vertices[face])
        edges = []
        for vertex in near:
            for face in self.vertex_faces[vertex]:
                ring = self.face_vertices[face]
                for tail, head, edge in zip(
                    ring, ring[1:] + ring[:1], self.face_edges[face]
                ):
                    if tail in near and head in near:
                        edges.append(edge)
        return edges

    def _snapped_loop(self, tail, strip, plane_distance):
        points = self.position_list
        allowed = self.local_interior

        def step_cost(start, end):
            a, b = points[start], points[end]
            length = math.sqrt(
                (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
            )
            return length + abs(plane_distance(end))

        faces = [*strip, strip[0]]
        best = {(0, tail): (0.0, None)}
        for position, face in enumerate(faces):
            ring = self.face_vertices[face]
            if position > 0:
                for vertex in ring:
                    previous = best.get((position - 1, vertex))
                    if previous is not None:
                        best[(position, vertex)] = (previous[0], (position - 1, vertex))
            changed = True
            while changed:
                changed = False
                for index, vertex in enumerate(ring):
                    following = ring[(index + 1) % len(ring)]
                    for start, end in ((vertex, following), (following, vertex)):
                        reached = best.get((position, start))
                        if reached is None or not allowed[end]:
                            continue
                        cost = reached[0] + step_cost(start, end)
                        if cost < best.get((position, end), (math.inf,))[0]:
                            best[(position, end)] = (cost, (position, start))
                            changed = True
        node = (len(faces) - 1, tail)
        if node not in best:
            return None
        loop = []
        while node is not None:
            if not loop or loop[-1] != node[1]:
                loop.append(node[1])
            node = best[node][1]
        loop = loop[:0:-1]
        if len(loop) < 3 or len(set(loop)) != len(loop):
            return None
        return loop


def loop_side(region, loop):
    loop = numpy.asarray(loop, dtype=numpy.int64)
    joined = ~numpy.isin(
        region.pair_edge_keys, undirected_edge_keys(loop, numpy.roll(loop, -1))
    )
    parts = lowest_index_part_labels(len(region.faces), region.local_face_pairs[joined])
    sides = numpy.unique(parts)
    if len(sides) != 2:
        return None
    return region.faces[parts == sides[1]]


# flat patches score 1 and hemisphere caps 2
MINIMUM_CAP_TO_SPANNED_AREA = 1.5


def untouching_loops(loops, lengths):
    vertex_sets = [set(loop) for loop in loops]
    touching = [
        {
            other
            for other, other_vertices in enumerate(vertex_sets)
            if other != index and not vertices.isdisjoint(other_vertices)
        }
        for index, vertices in enumerate(vertex_sets)
    ]
    remaining = set(range(len(loops)))
    kept = []
    while remaining:
        chosen = min(
            remaining,
            key=lambda index: (len(touching[index] & remaining), lengths[index]),
        )
        kept.append(loops[chosen])
        remaining -= touching[chosen] | {chosen}
    return kept


def drawn_loops(edge_loops, region):
    if region.mostly_quads:
        candidates = edge_loops.quad_loops(region)
    else:
        candidates = edge_loops.sliced_loops(region)
    face_areas = edge_loops.face_areas[region.faces]
    off_interior = ~numpy.array(region.local_interior)
    touches_border = (
        numpy.bincount(
            region.corner_faces,
            weights=off_interior[region.corners],
            minlength=len(region.faces),
        )
        > 0
    )
    loops, lengths = [], []
    for loop in candidates:
        side = loop_side(region, loop)
        if side is None:
            continue
        in_side = numpy.isin(region.faces, side)
        cap_areas = [
            face_areas[faces].sum()
            for faces in (in_side, ~in_side)
            if not touches_border[faces].any()
        ]
        points = edge_loops.positions[loop]
        following = numpy.roll(points, -1, axis=0)
        length = numpy.linalg.norm(points - following, axis=1).sum()
        # the summed cross products of a closed loop are twice the area it spans
        spanned_area = numpy.linalg.norm(numpy.cross(points, following).sum(axis=0)) / 2
        if (
            not cap_areas
            or min(cap_areas) / spanned_area >= MINIMUM_CAP_TO_SPANNED_AREA
        ):
            loops.append(loop)
            lengths.append(length)
    if region.mostly_quads:
        return loops
    return untouching_loops(loops, lengths)
