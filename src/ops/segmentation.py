import math
from collections import namedtuple

import numpy
from bpy_extras.view3d_utils import location_3d_to_region_2d
from mathutils import Vector

from ..no_bpy.edge_loops import EdgeLoops, LoopRegion, drawn_loops, loop_side
from ..no_bpy.mesh import faces_from_arrays, undirected_edge_keys
from ..no_bpy.segmentation import (
    edge_face_pairs,
    joined_labels,
    nested_fine_labels,
    on_pairs_with_edges,
    pair_borders,
    pair_edges,
    pairs_between_regions,
)
from ..no_bpy.symmetry import face_partners, region_partners
from ..segmentation import stored_loop_edges, write_regions
from ..utils.mesh import (
    edge_vertex_pairs,
    loop_vertices_and_face_sizes,
    vertex_position_array,
)
from .plan_drawing import (
    BORDER_COLOR,
    HOVER_COLOR,
    LINE_NUDGE,
    LINE_WIDTH,
    Shape,
    shape_batch,
)

OFF_BORDER_COLOR = (0.35, 0.35, 0.35, 0.45)
BORDER_PICK_PIXELS = 30

LoopCandidate = namedtuple("LoopCandidate", "relabeled_faces vertices edges shape")
NO_LOOPS = ()
DrawnLoops = namedtuple("DrawnLoops", "edges loop_of_edge loops shape")
NearestDrawnLoop = namedtuple("NearestDrawnLoop", "index closest")
RegionState = namedtuple("RegionState", "fine on loop_edges")


def closest_points_on_segments(starts, ends, point):
    directions = ends - starts
    along = ((point - starts) * directions).sum(axis=1)
    along = numpy.clip(along / (directions**2).sum(axis=1), 0, 1)
    return starts + along[:, None] * directions


class RegionEditor:
    def __init__(self, obj, segments, partners):
        mesh = obj.data
        self.obj = obj
        self.positions = vertex_position_array(mesh)
        corners, totals = loop_vertices_and_face_sizes(mesh)
        self.face_corners = faces_from_arrays(corners, totals)
        self.face_of_vertex = numpy.full(len(self.positions), -1, dtype=numpy.int64)
        self.face_of_vertex[corners] = numpy.repeat(numpy.arange(len(totals)), totals)
        self.vertex_partners = numpy.full(len(self.positions), -1, dtype=numpy.int64)
        self.vertex_partners[list(partners)] = list(partners.values())
        self.face_partners = face_partners(corners, totals, self.vertex_partners)
        self.face_pairs = edge_face_pairs(corners, totals)
        self.edge_loops = EdgeLoops(self.positions, corners, totals)
        self.loop_edges = set()
        if segments.hand_edited:
            self.loop_edges = set(map(tuple, stored_loop_edges(mesh).tolist()))
        edge_keys = undirected_edge_keys(*edge_vertex_pairs(mesh).T)
        self.edge_order = numpy.argsort(edge_keys)
        self.sorted_edge_keys = edge_keys[self.edge_order]

        matrix = numpy.array(obj.matrix_world)
        self.rotation_scale, self.offset = matrix[:3, :3], matrix[:3, 3]
        self.hidden_regions = set()
        self.loop_regions = {}
        self.drawn_loops = {}
        self.drawn_loop_batches = {}
        self._set_fine(nested_fine_labels(segments.labels, segments.fine_labels))
        if segments.hand_edited:
            self.on = on_pairs_with_edges(self.borders, segments.border_edges)
        else:
            self.on = pairs_between_regions(self.borders, segments.labels, self.fine)
        self.hovered = None
        self.hovered_mirror = None
        self.hovered_loop_key = None
        self.hovered_loops = NO_LOOPS
        self.hovered_part = None
        self._join_parts()

    def _world(self, points):
        return points @ self.rotation_scale.T + self.offset

    def _set_fine(self, fine):
        self.fine = fine
        self.region_partners = region_partners(fine, self.face_partners)
        self.borders = pair_borders(self.face_pairs, fine)
        self.pairs_of_region = {}
        for pair in self.borders:
            for region in pair:
                self.pairs_of_region.setdefault(region, []).append(pair)
        self.border_shapes = {
            pair: Shape(
                "LINES",
                self._world(self.positions[edges.ravel()]),
                None,
                OFF_BORDER_COLOR,
                LINE_NUDGE,
                LINE_WIDTH,
            )
            for pair, edges in self.borders.items()
        }
        # gpu batches are made inside the draw handler, where the gpu context is
        self.border_batches = {}

    def _forget_regions(self, regions):
        regions = set(regions)
        for cache in (self.loop_regions, self.drawn_loops, self.drawn_loop_batches):
            for part_regions in [key for key in cache if regions.intersection(key)]:
                del cache[part_regions]

    def _join_parts(self):
        self.joined, self.part_count = joined_labels(self.fine, self.borders, self.on)
        part_of_region = numpy.zeros(int(self.fine.max()) + 1, dtype=numpy.int64)
        part_of_region[self.fine] = self.joined
        self.regions_of_part = [()] + [
            tuple(numpy.flatnonzero(part_of_region == part).tolist())
            for part in range(1, self.part_count + 1)
        ]
        self.parts_of_part = {part: set() for part in range(1, self.part_count + 1)}
        for low, high in self.on:
            first, second = int(part_of_region[low]), int(part_of_region[high])
            if first != second:
                self.parts_of_part[first].add(second)
                self.parts_of_part[second].add(first)
        current = set(self.regions_of_part)
        for cache in (self.loop_regions, self.drawn_loops, self.drawn_loop_batches):
            for part_regions in [key for key in cache if key not in current]:
                del cache[part_regions]
        # mirrored candidates go stale when a toggle changes the mirror's part
        self.loop_candidates = {}
        self.loop_batches = {}

    def hide_regions(self, hidden_faces):
        self.hidden_regions = set(numpy.unique(self.fine[hidden_faces]).tolist())
        self.hidden_regions.discard(0)

    def edits_face(self, face):
        region = int(self.fine[face])
        return region != 0 and region not in self.hidden_regions

    def state(self):
        return RegionState(self.fine, frozenset(self.on), frozenset(self.loop_edges))

    def restore(self, state):
        for cache in (self.loop_regions, self.drawn_loops, self.drawn_loop_batches):
            cache.clear()
        self._set_fine(state.fine)
        self.on = set(state.on)
        self.loop_edges = set(state.loop_edges)
        self._changed()

    def has_hover(self):
        return self.hovered is not None or bool(self.hovered_loops)

    def border_near(self, face, point):
        nearest, nearest_point, nearest_distance = None, None, numpy.inf
        for pair in self.pairs_of_region.get(int(self.fine[face]), []):
            edges = self.borders[pair]
            closest = closest_points_on_segments(
                self.positions[edges[:, 0]], self.positions[edges[:, 1]], point
            )
            distances = numpy.linalg.norm(closest - point, axis=1)
            index = int(distances.argmin())
            if distances[index] < nearest_distance:
                nearest, nearest_point = pair, closest[index]
                nearest_distance = distances[index]
        return nearest, nearest_point

    def _pixels_apart(self, region, point, other):
        to_world = self.obj.matrix_world
        first = location_3d_to_region_2d(region, region.data, to_world @ Vector(point))
        second = location_3d_to_region_2d(region, region.data, to_world @ Vector(other))
        if first is None or second is None:
            return math.inf
        return (first - second).length

    def _edge_near(self, face, point):
        vertices = numpy.array(self.face_corners[face])
        following = numpy.roll(vertices, -1)
        closest = closest_points_on_segments(
            self.positions[vertices], self.positions[following], point
        )
        index = int(numpy.linalg.norm(closest - point, axis=1).argmin())
        low, high = sorted((int(vertices[index]), int(following[index])))
        return self.regions_of_part[self.joined[face]], low, high

    def _hover_loop_under(self, face, point):
        edge = self._edge_near(face, point)
        if edge not in self.loop_candidates:
            self.loop_candidates[edge] = self._loops_through(*edge)
        self.hovered_loop_key = edge
        self.hovered_loops = self.loop_candidates[edge]

    def hover(self, region, hit):
        hovered_before, loops_before = self.hovered, self.hovered_loops
        part_before = self.hovered_part
        self.hovered, self.hovered_part = None, None
        self.hovered_loop_key, self.hovered_loops = None, NO_LOOPS
        if hit is not None and self.edits_face(hit.face):
            face, point = hit.face, hit.point
            self.hovered_part = self.regions_of_part[self.joined[face]]
            drawn = self._drawn_loops(self.hovered_part)
            pair, closest = self.border_near(face, point)
            if drawn is not None:
                nearest = self._nearest_drawn_loop(drawn, pair, closest, point)
                if nearest is None:
                    self.hovered = pair
                elif (
                    self._pixels_apart(region, point, nearest.closest)
                    <= BORDER_PICK_PIXELS
                ):
                    key = (self.hovered_part, nearest.index)
                    if key not in self.loop_candidates:
                        self.loop_candidates[key] = self._loop_and_mirror(
                            drawn.loops[nearest.index]
                        )
                    self.hovered_loop_key = key
                    self.hovered_loops = self.loop_candidates[key]
                else:
                    self._hover_loop_under(face, point)
            elif pair is not None and (
                self._pixels_apart(region, point, closest) <= BORDER_PICK_PIXELS
            ):
                self.hovered = pair
            else:
                self._hover_loop_under(face, point)
        self.hovered_mirror = None
        if self.hovered is not None:
            self.hovered_mirror = self._mirrored_border(self.hovered)
        return (
            self.hovered != hovered_before
            or self.hovered_loops is not loops_before
            or self.hovered_part != part_before
        )

    def _drawn_loops(self, part_regions):
        if part_regions in self.drawn_loops:
            return self.drawn_loops[part_regions]
        loops = drawn_loops(self.edge_loops, self._loop_region(part_regions))
        drawn = None
        if loops:
            edges = numpy.concatenate(
                [numpy.stack([loop, numpy.roll(loop, -1)], axis=1) for loop in loops]
            )
            shape = Shape(
                "LINES",
                self._world(self.positions[edges.ravel()]),
                None,
                OFF_BORDER_COLOR,
                LINE_NUDGE,
                LINE_WIDTH / 2,
            )
            loop_of_edge = numpy.repeat(
                numpy.arange(len(loops)), [len(loop) for loop in loops]
            )
            drawn = DrawnLoops(edges, loop_of_edge, loops, shape)
        self.drawn_loops[part_regions] = drawn
        return drawn

    def _nearest_drawn_loop(self, drawn, pair, closest, point):
        on_loops = closest_points_on_segments(
            self.positions[drawn.edges[:, 0]], self.positions[drawn.edges[:, 1]], point
        )
        distances = numpy.linalg.norm(on_loops - point, axis=1)
        index = int(distances.argmin())
        if pair is not None and numpy.linalg.norm(closest - point) <= distances[index]:
            return None
        return NearestDrawnLoop(int(drawn.loop_of_edge[index]), on_loops[index])

    def _mirrored_border(self, pair):
        low, high = (self.region_partners.get(region) for region in pair)
        if low is None or high is None:
            return None
        mirrored = (min(low, high), max(low, high))
        if mirrored == pair or mirrored not in self.borders:
            return None
        return mirrored

    def _loop_region(self, part_regions):
        if part_regions not in self.loop_regions:
            self.loop_regions[part_regions] = LoopRegion(
                self.edge_loops, numpy.isin(self.fine, part_regions), self.face_pairs
            )
        return self.loop_regions[part_regions]

    def _loops_through(self, part_regions, tail, head):
        loop = self.edge_loops.find_loop(self._loop_region(part_regions), tail, head)
        if loop is None:
            return NO_LOOPS
        return self._loop_and_mirror(loop)

    def _loop_and_mirror(self, loop):
        candidate = self._loop_candidate(loop)
        if candidate is None:
            return NO_LOOPS
        mirrored = self._mirrored_loop(loop)
        if mirrored is None:
            return (candidate,)
        mirror_candidate = self._loop_candidate(mirrored)
        if mirror_candidate is None:
            return (candidate,)
        return (candidate, mirror_candidate)

    def _mirrored_loop(self, loop):
        mirrored = self.vertex_partners[loop]
        if (mirrored < 0).any() or set(mirrored.tolist()) == set(loop):
            return None
        keys = undirected_edge_keys(mirrored, numpy.roll(mirrored, -1))
        if not numpy.isin(keys, self.sorted_edge_keys).all():
            return None
        return mirrored.tolist()

    def _loop_candidate(self, loop):
        part = self.joined[self.face_of_vertex[loop[0]]]
        loop_region = self._loop_region(self.regions_of_part[part])
        if not all(loop_region.interior[vertex] for vertex in loop):
            return None
        side = loop_side(loop_region, loop)
        if side is None:
            return None
        edges = numpy.stack([loop, numpy.roll(loop, -1)], axis=1)
        shape = Shape(
            "LINES",
            self._world(self.positions[edges.ravel()]),
            None,
            HOVER_COLOR,
            LINE_NUDGE,
            LINE_WIDTH / 2,
        )
        return LoopCandidate(side, loop, numpy.sort(edges, axis=1), shape)

    def add_loops(self, candidates):
        for hovered in candidates:
            candidate = self._loop_candidate(hovered.vertices)
            if candidate is not None:
                self._split_along(candidate)
        self._changed()

    def _split_along(self, candidate):
        on_edges = pair_edges(self.borders, self.on) + candidate.edges.tolist()
        fine = self.fine.copy()
        far_side = numpy.zeros(len(fine), dtype=bool)
        far_side[candidate.relabeled_faces] = True
        split_regions = numpy.intersect1d(fine[far_side], fine[~far_side])
        new_labels = numpy.arange(len(split_regions)) + int(fine.max()) + 1
        relabeled = far_side & numpy.isin(fine, split_regions)
        fine[relabeled] = new_labels[numpy.searchsorted(split_regions, fine[relabeled])]
        self.loop_edges.update(map(tuple, candidate.edges.tolist()))
        self._forget_regions([*split_regions.tolist(), *new_labels.tolist()])
        self._set_fine(fine)
        self.on = on_pairs_with_edges(self.borders, on_edges)
        self._join_parts()

    def _is_loop_border(self, pair):
        return set(map(tuple, self.borders[pair].tolist())) <= self.loop_edges

    def _whole_loop(self, pair):
        loop_pairs = {other for other in self.borders if self._is_loop_border(other)}
        whole = {pair}
        vertices = set(self.borders[pair].ravel().tolist())
        while True:
            touching = {
                other
                for other in loop_pairs - whole
                if not vertices.isdisjoint(self.borders[other].ravel().tolist())
            }
            if not touching:
                return whole
            whole |= touching
            for other in touching:
                vertices.update(self.borders[other].ravel().tolist())

    def toggle(self, pair):
        pairs = {pair}
        mirrored = self._mirrored_border(pair)
        same_kind = mirrored is not None and (
            self._is_loop_border(mirrored) == self._is_loop_border(pair)
        )
        if same_kind:
            pairs.add(mirrored)
        if self._is_loop_border(pair):
            self._remove_loops(sorted(set().union(*map(self._whole_loop, pairs))))
        elif pair in self.on:
            self.on -= pairs
        else:
            self.on |= pairs
        self._changed()

    def _remove_loops(self, pairs):
        on_edges = pair_edges(self.borders, self.on - set(pairs))
        fine = self.fine
        for low, high in pairs:
            self.loop_edges -= set(map(tuple, self.borders[(low, high)].tolist()))
            fine = numpy.where(fine == high, low, fine)
        self._forget_regions(region for pair in pairs for region in pair)
        self._set_fine(fine)
        self.on = on_pairs_with_edges(self.borders, on_edges)

    def _changed(self):
        self.hovered = None
        self.hovered_mirror = None
        self.hovered_loop_key = None
        self.hovered_loops = NO_LOOPS
        self.hovered_part = None
        self._join_parts()
        self.write()

    def _edge_flags(self, edges):
        flags = numpy.zeros(len(self.edge_order), dtype=bool)
        if edges:
            keys = undirected_edge_keys(*numpy.array(edges).T)
            flags[self.edge_order[numpy.searchsorted(self.sorted_edge_keys, keys)]] = (
                True
            )
        return flags

    def write(self):
        write_regions(
            self.obj.data,
            self.fine,
            self._edge_flags(pair_edges(self.borders, self.on)),
            self._edge_flags(sorted(self.loop_edges)),
        )

    def draw_items(self):
        items = []
        for pair, shape in self.border_shapes.items():
            if pair[0] in self.hidden_regions:
                continue
            if pair in (self.hovered, self.hovered_mirror):
                color = HOVER_COLOR
            elif pair in self.on:
                color = BORDER_COLOR
            else:
                color = OFF_BORDER_COLOR
            if pair not in self.border_batches:
                self.border_batches[pair] = shape_batch(shape)
            items.append((shape._replace(color=color), self.border_batches[pair]))
        drawn = self.drawn_loops.get(self.hovered_part)
        if drawn is not None:
            if self.hovered_part not in self.drawn_loop_batches:
                self.drawn_loop_batches[self.hovered_part] = shape_batch(drawn.shape)
            items.append((drawn.shape, self.drawn_loop_batches[self.hovered_part]))
        for index, loop in enumerate(self.hovered_loops):
            key = (self.hovered_loop_key, index)
            if key not in self.loop_batches:
                self.loop_batches[key] = shape_batch(loop.shape)
            items.append((loop.shape, self.loop_batches[key]))
        return items
