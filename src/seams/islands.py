"""Post-unwrap repair, in uv space.

A cut-open tube unrolls into a strip as long as the tube's ring, and
the unwrap folds the longest of them, so the 3D shape does not
predict the unwrapped one. This measures the unwrap itself: ruined
islands are cut across and unwrapped again, non-disk islands are
opened, and clean islands split only when they are strips too long to
pack."""

import collections
import math

from .cuts import connect_loops, crease_relief, cut_path, edge_cost, path_cost
from .mesh import build, face_edges, find, island_groups, pair, signed_area


# strip test: length squared over uv area, about length/width. a folded
# strip is sliced into aspect-bound bins, anything else ruined is halved.
# close_rings reuses SPLIT_ASPECT, a closed ring past it would unroll into a
# strip the split would cut back up anyway
SPLIT_ASPECT = 6.0
SPLIT_LENGTH = 0.25
# sweeps of sliding faces across a fresh bin cut while that shortens it,
# capped so the cut cannot creep along a tapering strip
STRAIGHTEN_SWEEPS = 4


def uv_topology(group, faces, edges, seams):
    """Euler characteristic and uv boundary loops of one island, counted the
    way the engine reads the exported vt mesh: corners glue across interior
    non-seam edges, so a cut-open tube is the disk its unwrap is. A disk is
    1. Boundary loops come back as mesh vert sets, and loops touching at a
    glued corner count as one, a cut between them would be a point.
    """
    parent = {}

    def union(a, b):
        ra, rb = find(parent, a), find(parent, b)
        if ra != rb:
            parent[ra] = rb

    in_group = set(group)
    for f in group:
        for v in faces[f]:
            parent.setdefault((f, v), (f, v))
    for f in group:
        face = faces[f]
        n = len(face)
        for i in range(n):
            u, v = face[i], face[(i + 1) % n]
            key = pair(u, v)
            owners = edges[key]
            if len(owners) == 2 and key not in seams:
                g = owners[1] if owners[0] == f else owners[0]
                if g != f and f < g and g in in_group:
                    union((f, u), (g, u))
                    union((f, v), (g, v))

    edge_count = collections.Counter()
    for f in group:
        face = faces[f]
        n = len(face)
        for i in range(n):
            a, b = find(parent, (f, face[i])), find(parent, (f, face[(i + 1) % n]))
            edge_count[(a, b) if a < b else (b, a)] += 1

    classes = {find(parent, node) for node in parent}
    ec = len(classes) - len(edge_count) + len(group)

    comp_parent = {}
    boundary = [key for key, count in edge_count.items() if count == 1]
    for a, b in boundary:
        comp_parent.setdefault(a, a)
        comp_parent.setdefault(b, b)
        ra, rb = find(comp_parent, a), find(comp_parent, b)
        if ra != rb:
            comp_parent[ra] = rb

    loops = collections.defaultdict(set)
    for a, b in boundary:
        # a corner class only ever holds one mesh vert, its node's second slot
        loops[find(comp_parent, a)].update((a[1], b[1]))
    return ec, list(loops.values())


def crosses(a, b, c, d):
    """Mirror of the engine's Test2DSegmentSegment with eps 0, collinear
    branch included: collinear segments only count when their projections
    overlap. Segments sharing an endpoint always read as crossing, the caller
    must skip those pairs."""
    a1 = signed_area((a, b, d))
    a2 = signed_area((a, b, c))
    if a1 * a2 > 0:
        return False
    a3 = signed_area((c, d, a))
    a4 = a3 + a2 - a1
    if a3 * a4 > 0:
        return False
    if a1 == 0 and a2 == 0:
        ab = (b[0] - a[0], b[1] - a[1])
        sq = ab[0] ** 2 + ab[1] ** 2
        if sq == 0:
            return False
        coef_c = ((c[0] - a[0]) * ab[0] + (c[1] - a[1]) * ab[1]) / sq
        coef_d = ((d[0] - a[0]) * ab[0] + (d[1] - a[1]) * ab[1]) / sq
        lo, hi = sorted((coef_c, coef_d))
        return not (lo > 1.0 or hi < 0.0)
    return True


def island_ruined(group, faces, uvs, edges, seams):
    """A flipped or collapsed face, a non-disk island, or two boundary
    segments crossing: what makes the engine throw the island's layout
    away and re-cut it. Crossings between two different islands do not
    happen out of blender's packer, only inside one island.
    """
    signed = [signed_area(uvs[f]) for f in group]
    total = sum(signed)
    if total == 0 or any(s * total <= 0 for s in signed):
        return True

    ec, _ = uv_topology(group, faces, edges, seams)
    if ec != 1:
        return True

    segs = []
    for f in group:
        face = faces[f]
        n = len(face)
        for i in range(n):
            key = pair(face[i], face[(i + 1) % n])
            if key in seams or len(edges[key]) != 2:
                segs.append((uvs[f][i], uvs[f][(i + 1) % n]))
    if not segs:
        return True  # an island with no boundary cannot be flat
    cell = sum(math.dist(a, b) for a, b in segs) / len(segs)
    if cell == 0:
        return True
    grid = collections.defaultdict(list)
    for idx, (a, b) in enumerate(segs):
        xs = sorted((a[0], b[0]))
        ys = sorted((a[1], b[1]))
        for cx in range(int(xs[0] // cell), int(xs[1] // cell) + 1):
            for cy in range(int(ys[0] // cell), int(ys[1] // cell) + 1):
                grid[(cx, cy)].append(idx)
    seen = set()
    for members in grid.values():
        for i, ei in enumerate(members):
            for ej in members[i + 1 :]:
                key = (ei, ej) if ei < ej else (ej, ei)
                if key in seen:
                    continue
                seen.add(key)
                a, b = segs[ei]
                c, d = segs[ej]
                # touching means a shared uv corner, not a mesh vertex: the two
                # sides of a seam edge share verts yet can cross unwrapped
                if {a, b} & {c, d}:
                    continue
                if crosses(a, b, c, d):
                    return True
    return False


def straighten_cut(
    verts, group, faces, edges, seams, bin_of, weights=None, relief=None
):
    """Straighten an island's bin cut, returning its seam edges.

    The bin line zigzags on diagonal edge flow. Capped sweeps of sliding
    faces across the cut, strictly shortening, flatten single-face teeth.
    Then each connected run of cut edges is swapped for the shortest
    interior path between its own two endpoints when strictly shorter,
    which keeps the cut network separating the same way, so this can never
    cut a tube wall lengthwise. Runs that loop or branch stay. Both passes
    count a painted edge as longer, so the cut also moves out of restricted
    areas.
    """
    lengths = {}
    interior = collections.defaultdict(set)
    neighbors = collections.defaultdict(list)
    for f in group:
        face = faces[f]
        for i in range(len(face)):
            key = pair(face[i], face[(i + 1) % len(face)])
            owners = edges[key]
            if key in seams or len(owners) != 2:
                continue
            other = owners[1] if owners[0] == f else owners[0]
            if other == f:
                continue
            if key not in lengths:
                lengths[key] = edge_cost(verts, weights, key[0], key[1], relief)
                interior[key[0]].add(key[1])
                interior[key[1]].add(key[0])
            neighbors[f].append((other, lengths[key]))

    for _ in range(STRAIGHTEN_SWEEPS):
        moved = False
        for f in sorted(neighbors):
            here = bin_of[f]
            costs = collections.defaultdict(float)
            for other, length in neighbors[f]:
                costs[bin_of[other]] += length
            # the cut length a bin costs this face is its edges to every
            # neighbour outside that bin
            total = sum(costs.values())
            move, target = min((total - c, b) for b, c in costs.items())
            if target != here and move < total - costs.get(here, 0.0):
                bin_of[f] = target
                moved = True
        if not moved:
            break

    cuts = collections.defaultdict(set)
    for key in lengths:
        a, b = edges[key]
        if bin_of[a] != bin_of[b]:
            cuts[pair(bin_of[a], bin_of[b])].add(key)

    extra = set()
    for cut_edges in cuts.values():
        by_vert = collections.defaultdict(list)
        for key in cut_edges:
            by_vert[key[0]].append(key)
            by_vert[key[1]].append(key)
        seen = set()
        for start in sorted(cut_edges):
            if start in seen:
                continue
            run = []
            stack = [start]
            seen.add(start)
            while stack:
                key = stack.pop()
                run.append(key)
                for v in key:
                    for near in by_vert[v]:
                        if near not in seen:
                            seen.add(near)
                            stack.append(near)
            degree = collections.Counter(v for key in run for v in key)
            ends = [v for v, d in degree.items() if d % 2]
            if len(ends) != 2:
                extra.update(run)
                continue
            path = cut_path(
                verts, interior, set(ends[:1]), set(ends[1:]), weights, relief
            )
            straight = [pair(a, b) for a, b in zip(path, path[1:])]
            if all(d <= 2 for d in degree.values()):
                # a simple chain orders, so both sides price their turns and
                # a straight line beats a marginally shorter staircase
                run_keys = set(run)
                seq = [ends[0]]
                walked = set()
                while len(walked) < len(run):
                    key = next(
                        k for k in by_vert[seq[-1]] if k in run_keys and k not in walked
                    )
                    walked.add(key)
                    seq.append(key[1] if key[0] == seq[-1] else key[0])
                better = straight and path_cost(
                    verts, path, weights, relief
                ) < path_cost(verts, seq, weights, relief)
            else:
                better = straight and sum(lengths[k] for k in straight) < sum(
                    lengths[k] for k in run
                )
            extra.update(straight if better else run)
    return extra


def split_islands(verts, faces, seams, uvs, weights=None, groups=None):
    """Extra seam edges that cut ruined uv islands into smaller pieces.

    Runs on the unwrap of the seams this package chose and measures the
    unwrap itself, the 3D shape does not predict it: per island, the extent
    along the principal axis of its face centroids against its uv area. A
    folded strip bins along that axis and the edges between bins are the
    seams for a second unwrap, a ruined island that is not a strip is
    halved, and a non-disk island is opened with a path joining two of its
    boundary loops, splitting cannot fix topology. A clean island is left
    whole unless it is a strip past the length and aspect bounds, those
    pack badly and artists slice them too. groups restricts the scan, for
    a caller that knows the rest is unchanged.
    """
    edges = face_edges(faces)
    if groups is None:
        groups = island_groups(faces, seams, edges)
    relief = None

    def cut_relief():
        # the normals and relief only pay off once a cut is actually needed,
        # and most calls find nothing to cut
        nonlocal relief
        if relief is None:
            weighted, _, _ = build(verts, faces)
            relief = crease_relief(verts, faces, weighted, edges)
        return relief

    extra = set()
    # an island is what the unwrap made one: faces joined by unseamed edges
    for group in groups:
        centroids = {}
        areas = {}
        for f in group:
            pts = uvs[f]
            centroids[f] = (
                sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts),
            )
            areas[f] = abs(signed_area(pts))
        size = sum(areas.values())
        if len(group) < 2 or size <= 0:
            continue
        cx = sum(areas[f] * centroids[f][0] for f in group) / size
        cy = sum(areas[f] * centroids[f][1] for f in group) / size
        xx = xy = yy = 0.0
        for f in group:
            dx, dy = centroids[f][0] - cx, centroids[f][1] - cy
            xx += areas[f] * dx * dx
            xy += areas[f] * dx * dy
            yy += areas[f] * dy * dy
        angle = 0.5 * math.atan2(2 * xy, xx - yy)
        axis = (math.cos(angle), math.sin(angle))
        ts = {f: centroids[f][0] * axis[0] + centroids[f][1] * axis[1] for f in group}
        lo, hi = min(ts.values()), max(ts.values())
        length = hi - lo
        aspect = length * length / size
        strip = length >= SPLIT_LENGTH and aspect > SPLIT_ASPECT
        if not island_ruined(group, faces, uvs, edges, seams):
            if not strip:
                continue
        elif len(loops := uv_topology(group, faces, edges, seams)[1]) > 1:
            # non-disk: open it with a path joining two boundary loops, only
            # interior edges can become cuts
            adjacent = collections.defaultdict(set)
            for f in group:
                face = faces[f]
                n = len(face)
                for i in range(n):
                    key = pair(face[i], face[(i + 1) % n])
                    if len(edges[key]) == 2 and key not in seams:
                        adjacent[key[0]].add(key[1])
                        adjacent[key[1]].add(key[0])
            extra |= connect_loops(verts, adjacent, loops, weights, cut_relief())
            continue
        if length <= 0:
            # every centroid projects to one point (doubled faces), no bin split
            continue
        bins = math.ceil(aspect / SPLIT_ASPECT) if strip else 2
        bin_of = {f: min(bins - 1, int((ts[f] - lo) / length * bins)) for f in group}
        extra |= straighten_cut(
            verts, group, faces, edges, seams, bin_of, weights, cut_relief()
        )
    return extra
