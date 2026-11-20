import itertools
import math
from collections import namedtuple

import numpy

from .mesh import (
    corners_of_faces,
    face_starts,
    faces_from_arrays,
    find_root,
    next_corners_in_face,
    undirected_edge_keys,
)

Strip = namedtuple("Strip", "faces crossed_edges")
Slice = namedtuple("Slice", "strip plane_distances length")
COARSE_TILT_DEGREES = range(0, 180, 30)
FINE_TILT_DEGREES = 10
SLICE_START_CELLS_PER_REGION = 300


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


class PlaneDistances(dict):
    def __init__(self, position_list, origin_vertex, direction):
        super().__init__({origin_vertex: 0.0})
        self.position_list = position_list
        self.origin = position_list[origin_vertex]
        self.direction = direction

    def __missing__(self, vertex):
        point = self.position_list[vertex]
        origin, direction = self.origin, self.direction
        distance = self[vertex] = (
            (point[0] - origin[0]) * direction[0]
            + (point[1] - origin[1]) * direction[1]
            + (point[2] - origin[2]) * direction[2]
        )
        return distance


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
        cell_size = math.sqrt(
            self.face_areas[region.faces].sum() / SLICE_START_CELLS_PER_REGION
        )
        cells = numpy.floor(
            (region.positions - region.positions.min(axis=0)) / cell_size
        ).astype(numpy.int64)
        cell_of_vertex = dict(zip(region.vertices.tolist(), map(tuple, cells.tolist())))
        covered_cells = set()
        covered = numpy.zeros(len(region.edges), dtype=bool)
        loops = []
        for index, (tail, head) in enumerate(zip(tails.tolist(), heads.tolist())):
            skipped = (
                covered[index]
                or cell_of_vertex[tail] in covered_cells
                or not (region.interior[tail] and region.interior[head])
            )
            if skipped:
                continue
            loop = self.find_loop(region, tail, head)
            covered[region.edges_near(loop or [tail, head])] = True
            if loop is not None:
                loops.append(loop)
                covered_cells.update(cell_of_vertex[vertex] for vertex in loop)
                continue
            x, y, z = cell_of_vertex[tail]
            covered_cells.update(
                itertools.product(
                    range(x - 1, x + 2), range(y - 1, y + 2), range(z - 1, z + 2)
                )
            )
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


FaceTree = namedtuple("FaceTree", "order position end child_of_pair")


def _breadth_first_levels(root, neighbor_starts, neighbors, reached, parent_entry):
    frontier = numpy.array([root])
    reached[root] = True
    levels = [frontier]
    while True:
        counts = neighbor_starts[frontier + 1] - neighbor_starts[frontier]
        entries = numpy.repeat(
            neighbor_starts[frontier] - face_starts(counts), counts
        ) + numpy.arange(counts.sum())
        entries = entries[~reached[neighbors[entries]]]
        frontier, first = numpy.unique(neighbors[entries], return_index=True)
        if len(frontier) == 0:
            return levels
        reached[frontier] = True
        parent_entry[frontier] = entries[first]
        levels.append(frontier)


def _face_tree(face_count, face_pairs):
    sources = numpy.concatenate([face_pairs[:, 0], face_pairs[:, 1]])
    by_source = numpy.argsort(sources, kind="stable")
    entry_sources = sources[by_source]
    neighbor_starts = numpy.searchsorted(entry_sources, numpy.arange(face_count + 1))
    neighbors = numpy.concatenate([face_pairs[:, 1], face_pairs[:, 0]])[by_source]
    reached = numpy.zeros(face_count, dtype=bool)
    parent_entry = numpy.full(face_count, -1, dtype=numpy.int64)
    position = numpy.zeros(face_count, dtype=numpy.int64)
    size = numpy.ones(face_count, dtype=numpy.int64)
    placed = 0
    for root in range(face_count):
        if reached[root]:
            continue
        levels = _breadth_first_levels(
            root, neighbor_starts, neighbors, reached, parent_entry
        )
        for level in reversed(levels[1:]):
            numpy.add.at(size, entry_sources[parent_entry[level]], size[level])
        position[root] = placed
        placed += size[root]
        for level in levels[1:]:
            parents = entry_sources[parent_entry[level]]
            by_parent = numpy.argsort(parents, kind="stable")
            children, parents = level[by_parent], parents[by_parent]
            sizes_before = numpy.cumsum(size[children]) - size[children]
            first_sibling = numpy.maximum.accumulate(
                numpy.where(
                    numpy.diff(parents, prepend=-1) != 0,
                    numpy.arange(len(children)),
                    0,
                )
            )
            position[children] = (
                position[parents] + 1 + sizes_before - sizes_before[first_sibling]
            )
    order = numpy.empty(face_count, dtype=numpy.int64)
    order[position] = numpy.arange(face_count)
    child_of_pair = numpy.full(len(face_pairs), -1, dtype=numpy.int64)
    children = numpy.flatnonzero(parent_entry >= 0)
    child_of_pair[by_source[parent_entry[children]] % len(face_pairs)] = children
    return FaceTree(order, position, position + size, child_of_pair)


class FaceTreeRegion:
    def __init__(self, edge_loops, in_region, face_pairs):
        self.faces = numpy.flatnonzero(in_region)
        in_region_corners = in_region[edge_loops.face_of_corner]
        self.vertices, self.corners = numpy.unique(
            edge_loops.corners[in_region_corners], return_inverse=True
        )
        outside = numpy.bincount(
            edge_loops.corners[~in_region_corners],
            minlength=len(edge_loops.boundary),
        )
        interior = (outside == 0) & ~edge_loops.boundary
        self.interior = interior.tolist()
        self.local_interior = interior[self.vertices]
        self.face_sizes = edge_loops.totals[self.faces]
        self.corner_faces = numpy.repeat(numpy.arange(len(self.faces)), self.face_sizes)

        inside = in_region[face_pairs.first] & in_region[face_pairs.second]
        self.local_face_pairs = numpy.stack(
            [
                numpy.searchsorted(self.faces, face_pairs.first[inside]),
                numpy.searchsorted(self.faces, face_pairs.second[inside]),
            ],
            axis=1,
        )
        pair_edge_keys = undirected_edge_keys(
            face_pairs.edges[inside, 0], face_pairs.edges[inside, 1]
        )
        self.pairs_by_edge_key = numpy.argsort(pair_edge_keys)
        self.sorted_pair_edge_keys = pair_edge_keys[self.pairs_by_edge_key]
        self.tree = _face_tree(len(self.faces), self.local_face_pairs)

        pair_vertices = numpy.searchsorted(self.vertices, face_pairs.edges[inside])
        border_node = len(self.vertices)
        pair_nodes = numpy.where(
            interior[self.vertices][pair_vertices], pair_vertices, border_node
        )
        off_tree_pairs = numpy.flatnonzero(self.tree.child_of_pair < 0)
        parent = list(range(border_node + 1))
        closes_edge_cycle = []
        for tail, head in pair_nodes[off_tree_pairs].tolist():
            tail_root, head_root = find_root(parent, tail), find_root(parent, head)
            closes_edge_cycle.append(tail_root == head_root)
            parent[tail_root] = head_root
        self.pairs_closing_edge_cycles = off_tree_pairs[
            numpy.array(closes_edge_cycle, dtype=bool)
        ]


class LoopRegion(FaceTreeRegion):
    def __init__(self, edge_loops, in_region, face_pairs):
        super().__init__(edge_loops, in_region, face_pairs)
        self.positions = edge_loops.positions[self.vertices]
        self.position_list = self.positions.tolist()

        totals = self.face_sizes
        corners = self.corners
        self.mostly_quads = bool((totals == 4).mean() > 0.5)
        self.face_corner_starts = face_starts(totals)
        next_corner_vertices = corners[next_corners_in_face(totals)]
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
        self.corner_edges = corner_edges
        self.next_corner_vertices = next_corner_vertices
        self.corners_by_vertex = numpy.argsort(corners, kind="stable")
        self.vertex_corner_counts = numpy.bincount(
            corners, minlength=len(self.vertices)
        )
        self.vertex_corner_starts = face_starts(self.vertex_corner_counts)
        self.vertex_faces = faces_from_arrays(
            self.corner_faces[self.corners_by_vertex], self.vertex_corner_counts
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
        loop = self._snapped_loop(tail, best.strip.faces, best.plane_distances)
        if loop is None:
            return None
        return self.vertices[loop].tolist()

    def _slice(self, tail, across, along, tilt_degrees):
        tilt = math.radians(tilt_degrees)
        direction = (across * math.cos(tilt) + along * math.sin(tilt)).tolist()
        plane_distances = PlaneDistances(self.position_list, tail, direction)
        strip = self._strip(tail, plane_distances)
        if strip is None:
            return Slice(strip, plane_distances, math.inf)
        tails, heads = self.edges[strip.crossed_edges].T
        tail_distances = numpy.array(
            [plane_distances[vertex] for vertex in tails.tolist()]
        )
        head_distances = numpy.array(
            [plane_distances[vertex] for vertex in heads.tolist()]
        )
        share = tail_distances / (tail_distances - head_distances)
        crossings = self.positions[tails] + share[:, None] * (
            self.positions[heads] - self.positions[tails]
        )
        length = numpy.linalg.norm(
            crossings - numpy.roll(crossings, -1, axis=0), axis=1
        ).sum()
        return Slice(strip, plane_distances, float(length))

    def _crossed_edges(self, face, plane_distances):
        above = [plane_distances[vertex] >= 0 for vertex in self.face_vertices[face]]
        return [
            edge
            for tail_above, head_above, edge in zip(
                above, above[1:] + above[:1], self.face_edges[face]
            )
            if tail_above != head_above
        ]

    def _strip(self, tail, plane_distances):
        for start_face in self.vertex_faces[tail]:
            crossed = self._crossed_edges(start_face, plane_distances)
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
            crossed = self._crossed_edges(face, plane_distances)
            if len(crossed) != 2:
                return None
            leaving = crossed[1] if crossed[0] == leaving else crossed[0]
            strip.append(face)
            strip_edges.append(leaving)

    def _vertex_corners(self, vertices):
        return self.corners_by_vertex[
            corners_of_faces(
                self.vertex_corner_starts, self.vertex_corner_counts, vertices
            )
        ]

    def edges_near(self, loop):
        loop_corners = self._vertex_corners(numpy.searchsorted(self.vertices, loop))
        loop_faces = self.corner_faces[loop_corners]
        near_vertices = numpy.unique(
            self.corners[
                corners_of_faces(self.face_corner_starts, self.face_sizes, loop_faces)
            ]
        )
        near = numpy.zeros(len(self.vertices), dtype=bool)
        near[near_vertices] = True
        ring_corners = self._vertex_corners(near_vertices)
        return self.corner_edges[
            ring_corners[near[self.next_corner_vertices[ring_corners]]]
        ]

    def _snapped_loop(self, tail, strip, plane_distances):
        faces = numpy.array([*strip, strip[0]], dtype=numpy.int64)
        sizes = self.face_sizes[faces]
        node_starts = face_starts(sizes)
        node_positions = numpy.repeat(numpy.arange(len(faces)), sizes)
        node_vertices = self.corners[
            self.face_corner_starts[faces][node_positions]
            + numpy.arange(len(node_positions))
            - node_starts[node_positions]
        ]
        next_nodes = next_corners_in_face(sizes)
        next_vertices = node_vertices[next_nodes]
        offsets = self.positions[node_vertices] - self.positions[next_vertices]
        lengths = numpy.sqrt(
            offsets[:, 0] ** 2 + offsets[:, 1] ** 2 + offsets[:, 2] ** 2
        )
        distances = numpy.abs(
            [plane_distances[vertex] for vertex in node_vertices.tolist()]
        )
        allowed = self.local_interior
        forward_costs = numpy.where(
            allowed[next_vertices], lengths + distances[next_nodes], math.inf
        ).tolist()
        backward_costs = numpy.where(
            allowed[node_vertices], lengths + distances, math.inf
        ).tolist()

        node_keys = node_positions * len(self.vertices) + node_vertices
        by_key = numpy.argsort(node_keys)
        previous_keys = node_keys - len(self.vertices)
        found = numpy.minimum(
            numpy.searchsorted(node_keys[by_key], previous_keys), len(by_key) - 1
        )
        previous_nodes = numpy.where(
            node_keys[by_key][found] == previous_keys, by_key[found], -1
        ).tolist()

        node_count = len(node_vertices)
        best = [math.inf] * node_count
        came_from = [-1] * node_count
        start = int(numpy.flatnonzero(node_vertices[: sizes[0]] == tail)[0])
        best[start] = 0.0
        ring_starts = [*node_starts.tolist(), node_count]
        for position in range(len(faces)):
            ring = range(ring_starts[position], ring_starts[position + 1])
            for node in ring:
                source = previous_nodes[node]
                if source >= 0 and best[source] < math.inf:
                    best[node] = best[source]
                    came_from[node] = source
            steps = list(zip(ring, [*ring[1:], ring[0]]))
            # shortest paths with positive costs go one way round, so two laps each way cover them
            for node, ahead in steps + steps:
                cost = best[node] + forward_costs[node]
                if cost < best[ahead]:
                    best[ahead] = cost
                    came_from[ahead] = node
            for node, ahead in reversed(steps + steps):
                cost = best[ahead] + backward_costs[node]
                if cost < best[node]:
                    best[node] = cost
                    came_from[node] = ahead
        last_ring = node_vertices[ring_starts[-2] :]
        node = ring_starts[-2] + int(numpy.flatnonzero(last_ring == tail)[0])
        if best[node] == math.inf:
            return None
        vertices = node_vertices.tolist()
        loop = []
        while node >= 0:
            if not loop or loop[-1] != vertices[node]:
                loop.append(vertices[node])
            node = came_from[node]
        loop = loop[:0:-1]
        if len(loop) < 3 or len(set(loop)) != len(loop):
            return None
        return loop


def _far_side_spans(region, loop):
    loop = numpy.asarray(loop, dtype=numpy.int64)
    tree = region.tree
    loop_pairs = region.pairs_by_edge_key[
        numpy.searchsorted(
            region.sorted_pair_edge_keys,
            undirected_edge_keys(loop, numpy.roll(loop, -1)),
        )
    ]
    children = tree.child_of_pair[loop_pairs]
    children = children[children >= 0]
    side_changes = numpy.sort(
        numpy.concatenate([tree.position[children], tree.end[children]])
    )
    # loops around a handle leave the region in one piece
    checked_pairs = numpy.concatenate([loop_pairs, region.pairs_closing_edge_cycles])
    on_far_side = (
        numpy.searchsorted(
            side_changes,
            tree.position[region.local_face_pairs[checked_pairs]],
            side="right",
        )
        % 2
    )
    sides_differ = on_far_side[:, 0] != on_far_side[:, 1]
    if (sides_differ != numpy.isin(checked_pairs, loop_pairs)).any():
        return None
    return side_changes.reshape(-1, 2)


def loop_side(region, loop):
    spans = _far_side_spans(region, loop)
    if spans is None:
        return None
    far_side = numpy.concatenate(
        [region.tree.order[start:end] for start, end in spans.tolist()]
    )
    return region.faces[numpy.sort(far_side)]


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
        return [
            loop
            for loop in edge_loops.quad_loops(region)
            if _far_side_spans(region, loop) is not None
        ]
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
    area_before = numpy.concatenate(
        [[0.0], numpy.cumsum(face_areas[region.tree.order])]
    )
    border_faces_before = numpy.concatenate(
        [[0], numpy.cumsum(touches_border[region.tree.order])]
    )
    loops, lengths = [], []
    for loop in candidates:
        spans = _far_side_spans(region, loop)
        if spans is None:
            continue
        far_area = (area_before[spans[:, 1]] - area_before[spans[:, 0]]).sum()
        far_border_faces = (
            border_faces_before[spans[:, 1]] - border_faces_before[spans[:, 0]]
        ).sum()
        sides = (
            (far_area, far_border_faces),
            (area_before[-1] - far_area, border_faces_before[-1] - far_border_faces),
        )
        cap_areas = [area for area, border_faces in sides if border_faces == 0]
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
    return untouching_loops(loops, lengths)
