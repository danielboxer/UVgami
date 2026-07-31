"""Partition and merge, the core of the mode.

A per-edge angle test cannot seam a beveled model: a bevel splits one
crease into several small turns, and what separates it from a corner
is width, not angle. So partition at a low angle, which over-segments
bevels and curved surfaces into narrow bands, then merge back: absorb
dissolves anything narrower than the auto width, merge_smooth takes
any boundary that turns less than a crease, merge_flat joins
neighbours whose union is still nearly flat, and close_rings rejoins
disk pairs whose union is a short annulus. Every merge refuses to
leave a region with a hole, the engine throws non-disk charts away."""

import collections
import functools
import heapq
import math

from .islands import SPLIT_ASPECT
from .mesh import find, norm, pair, turn_angle


# auto width: at the low partition angle region widths are dominated by
# narrow bands, real surfaces are the top few percent, and no clean gap
# separates them, so take a high quantile with clearance on top. the cap, a
# fraction of the diagonal, keeps sparse cases like a beveled cube honest
WIDTH_CAP = 0.05
WIDTH_QUANTILE = 0.9
WIDTH_FACTOR = 2.0
# flat merge: |sum of weighted normals| / (2 * area) is 1 on a plane. merge
# while the union stays above a spherical cap of this half-angle, so shallow
# creases merge and real corners (~0.7 for a 90 degree pair) survive
FLAT_ANGLE = 30
# smooth merge: how far a boundary must turn to count as a crease. read at
# the boundary because flatness cannot pass a cylinder, whose panels spread
# as far as a corner once enough of them merge
CREASE_ANGLE = 30


# per region boundary, the length-weighted sums the smooth merge reads: turn
# carried across dissolved bands, turn at the boundary's own edges, the
# width that carry crossed, and length to divide by
Boundaries = collections.namedtuple("Boundaries", "turn step spread length")


def partition(faces, weighted, edges, angle, forced=None):
    """Union-find over faces, cutting every edge sharper than angle, and every
    edge in forced whatever it turns."""
    parent = list(range(len(faces)))

    for key, owners in edges.items():
        if len(owners) != 2:
            continue
        if turn_angle(weighted, owners) > angle:
            continue
        if forced and key in forced:
            continue
        a, b = owners
        ra, rb = find(parent, a), find(parent, b)
        if ra != rb:
            parent[ra] = rb
    return functools.partial(find, parent)


def boundary_turns(verts, weighted, edges, label):
    """Per region boundary, its turn summed over the edges, weighted by length.

    Divided by the boundary's length this is the angle the surface turns
    crossing it. Kept as a sum so merges can add boundaries together."""
    total = collections.defaultdict(float)
    for (v0, v1), owners in edges.items():
        if len(owners) != 2:
            continue
        ra, rb = label[owners[0]], label[owners[1]]
        if ra == rb:
            continue
        length = norm([verts[v0][i] - verts[v1][i] for i in range(3)])
        total[pair(ra, rb)] += length * turn_angle(weighted, owners)
    return total


def region_topology(edges, label):
    """Per-region euler characteristic, vertex sets and shared edge counts.

    EC is 1 for a disk, 0 once a region has a hole. Both merge passes refuse
    anything that lowers it, the engine throws non-disk charts away.
    """
    face_count = collections.Counter(label.values())
    rverts = collections.defaultdict(set)
    edge_count = collections.Counter()
    shared = collections.Counter()
    for (v0, v1), owners in edges.items():
        regions = {label[o] for o in owners}
        for r in regions:
            rverts[r].update((v0, v1))
            edge_count[r] += 1
        if len(regions) == 2:
            ra, rb = sorted(regions)
            shared[(ra, rb)] += 1
    ec = {r: len(rverts[r]) - edge_count[r] + face_count[r] for r in face_count}
    return ec, rverts, shared


def joint_count(rverts, shared, a, b):
    """Shared vertices minus shared edges: 1 when the two regions meet along a
    single path, 2 when they meet twice (so the union has a hole), 0 when the
    contact is a closed loop (so the union is closed)."""
    return len(rverts[a] & rverts[b]) - shared[pair(a, b)]


def keeps_topology(ec, rverts, shared, a, b):
    """A merge is allowed when the union is no worse than the worse of the two
    and not a closed surface: meeting along one path keeps a disk, meeting
    twice would open a hole, and swallowing the region inside a hole closes
    one.
    """
    merged = ec[a] + ec[b] - joint_count(rverts, shared, a, b)
    return min(ec[a], ec[b]) <= merged <= 1


def locked_pairs(edges, label, forced):
    """Region pairs whose shared boundary holds a forced seam. No merge may
    take one, so a hand-marked edge survives as a region boundary and the
    passes route around it.
    """
    locked = set()
    if not forced:
        return locked
    for key, owners in edges.items():
        if len(owners) == 2 and key in forced:
            ra, rb = label[owners[0]], label[owners[1]]
            if ra != rb:
                locked.add(pair(ra, rb))
    return locked


def region_stats(verts, areas, edges, label):
    area = collections.defaultdict(float)
    shared = collections.defaultdict(float)  # (ra, rb) -> boundary length
    perimeter = collections.defaultdict(float)
    for i, a in enumerate(areas):
        area[label[i]] += a
    for (v0, v1), owners in edges.items():
        length = norm([verts[v0][i] - verts[v1][i] for i in range(3)])
        regions = {label[o] for o in owners}
        if len(regions) == 2:
            ra, rb = sorted(regions)
            shared[(ra, rb)] += length
            perimeter[ra] += length
            perimeter[rb] += length
        elif len(owners) == 1:
            perimeter[label[owners[0]]] += length
    return area, shared, perimeter


def detect_width(verts, faces, areas, edges, root, scale):
    """Absolute merge width from a high quantile of the region widths."""
    label = {i: root(i) for i in range(len(faces))}
    area, _, perimeter = region_stats(verts, areas, edges, label)
    ws = sorted(2 * area[r] / perimeter[r] for r in area if perimeter[r] > 0)
    if not ws:
        return 0.0
    quantile = ws[min(len(ws) - 1, int(WIDTH_QUANTILE * len(ws)))]
    return min(WIDTH_FACTOR * quantile, WIDTH_CAP * scale)


def absorb(verts, faces, weighted, areas, edges, root, min_width, forced=None):
    """Repeatedly merge the narrowest region into its longest-shared neighbour.

    Stats update incrementally per merge and the heap is lazy, an entry
    whose width no longer matches its region is stale. Dissolving a band
    moves its turn and width onto the boundaries it leaves, so an absorbed
    bevel still reads as the crease it was, while step, the turn at a
    boundary's own edges, never moves. Returns labels plus the per-boundary
    sums the smooth merge reads.
    """
    label = {i: root(i) for i in range(len(faces))}
    area, shared, perimeter = region_stats(verts, areas, edges, label)
    turns = boundary_turns(verts, weighted, edges, label)
    steps = collections.defaultdict(float, turns)
    spread = collections.defaultdict(float)
    ec, rverts, shared_edges = region_topology(edges, label)
    locked = locked_pairs(edges, label, forced)
    neighbors = collections.defaultdict(set)
    for ra, rb in shared:
        neighbors[ra].add(rb)
        neighbors[rb].add(ra)
    alive = set(area)
    parent = {r: r for r in area}

    def width(r):
        return 2 * area[r] / perimeter[r] if perimeter[r] > 0 else math.inf

    heap = [(width(r), r) for r in alive if width(r) < min_width]
    heapq.heapify(heap)
    while heap:
        w, region = heapq.heappop(heap)
        if region not in alive or width(region) != w:
            continue
        partners = [
            (shared[pair(region, n)], n) for n in neighbors[region] if n in alive
        ]
        if not partners:
            continue
        # prefer a neighbour already over the threshold: a bevel should
        # dissolve into its surface, not accrete into a wider strip that
        # then survives the threshold itself
        wide = [p for p in partners if width(p[1]) >= min_width]
        rest = [p for p in partners if width(p[1]) < min_width]
        best = next(
            (
                n
                for _, n in sorted(wide, reverse=True) + sorted(rest, reverse=True)
                if pair(region, n) not in locked
                and keeps_topology(ec, rverts, shared_edges, region, n)
            ),
            None,
        )
        # nothing can take this strip without opening a hole, leave it be
        if best is None:
            continue

        contact = shared[pair(region, best)]
        # the turn crossing the band that is about to disappear, and how wide
        # it is, so the far side can tell a crease from spread out curvature
        crossed = turns[pair(region, best)] / contact if contact > 0 else 0.0
        crossed_width = width(region) + (
            spread[pair(region, best)] / contact if contact > 0 else 0.0
        )
        area[best] += area[region]
        perimeter[best] += perimeter[region] - 2 * contact
        ec[best] += ec[region] - joint_count(rverts, shared_edges, region, best)
        rverts[best] |= rverts[region]
        for n in list(neighbors[region]):
            if n != best and n in alive:
                turns[pair(best, n)] += (
                    turns[pair(region, n)] + shared[pair(region, n)] * crossed
                )
                spread[pair(best, n)] += (
                    spread[pair(region, n)] + shared[pair(region, n)] * crossed_width
                )
                steps[pair(best, n)] += steps[pair(region, n)]
                shared[pair(best, n)] = (
                    shared.get(pair(best, n), 0.0) + shared[pair(region, n)]
                )
                shared_edges[pair(best, n)] += shared_edges[pair(region, n)]
                if pair(region, n) in locked:
                    locked.add(pair(best, n))
                neighbors[n].add(best)
                neighbors[best].add(n)
            shared.pop(pair(region, n), None)
            turns.pop(pair(region, n), None)
            steps.pop(pair(region, n), None)
            spread.pop(pair(region, n), None)
            shared_edges.pop(pair(region, n), None)
            locked.discard(pair(region, n))
            neighbors[n].discard(region)
        neighbors[best].discard(best)
        del neighbors[region]
        alive.discard(region)
        parent[region] = best
        new_width = width(best)
        if new_width < min_width:
            heapq.heappush(heap, (new_width, best))

    bounds = Boundaries(turns, steps, spread, shared)
    return {i: find(parent, r) for i, r in label.items()}, bounds


def merge_smooth(edges, label, bounds, min_width, angle=CREASE_ANGLE, forced=None):
    """Merge neighbours whose shared boundary is not a crease.

    A crease is a turn with no width: every boundary on a cylinder wall
    turns one segment angle however much has merged, while a corner turns
    the whole corner. The turn absorb carried over counts only while its
    spread stays band-narrow, past that the boundary is judged by its own
    edges alone. Least turn first, lazy heap.
    """
    turns, steps, spread, shared = bounds
    neighbors = collections.defaultdict(set)
    for ra, rb in shared:
        neighbors[ra].add(rb)
        neighbors[rb].add(ra)
    ec, rverts, shared_edges = region_topology(edges, label)
    locked = locked_pairs(edges, label, forced)
    alive = set(label.values())
    parent = {r: r for r in alive}

    def crease(a, b):
        """How sharply the surface turns crossing this boundary: its own edges
        always count, a dissolved band's carry only while it is still band
        width.
        """
        key = pair(a, b)
        length = shared[key]
        if length <= 0:
            return math.inf
        step = steps[key] / length
        if spread[key] / length > min_width:
            return step
        return max(step, turns[key] / length)

    heap = [
        (crease(a, b), a, b)
        for a in neighbors
        for b in neighbors[a]
        if a < b and crease(a, b) < angle
    ]
    heapq.heapify(heap)

    while heap:
        value, a, b = heapq.heappop(heap)
        if (
            a not in alive
            or b not in alive
            or crease(a, b) != value
            or pair(a, b) in locked
            or not keeps_topology(ec, rverts, shared_edges, a, b)
        ):
            continue
        ec[a] += ec[b] - joint_count(rverts, shared_edges, a, b)
        rverts[a] |= rverts[b]
        for n in list(neighbors[b]):
            if n != a:
                # both sides of this boundary stay put, so no turn is crossed
                turns[pair(a, n)] += turns[pair(b, n)]
                steps[pair(a, n)] += steps[pair(b, n)]
                spread[pair(a, n)] += spread[pair(b, n)]
                shared[pair(a, n)] += shared[pair(b, n)]
                shared_edges[pair(a, n)] += shared_edges[pair(b, n)]
                if pair(b, n) in locked:
                    locked.add(pair(a, n))
                neighbors[n].add(a)
                neighbors[a].add(n)
            turns.pop(pair(b, n), None)
            steps.pop(pair(b, n), None)
            spread.pop(pair(b, n), None)
            shared.pop(pair(b, n), None)
            shared_edges.pop(pair(b, n), None)
            locked.discard(pair(b, n))
            neighbors[n].discard(b)
        neighbors[a].discard(a)
        del neighbors[b]
        alive.discard(b)
        parent[b] = a
        for n in neighbors[a]:
            value = crease(a, n)
            if value < angle:
                heapq.heappush(heap, (value, *((a, n) if a < n else (n, a))))

    return {i: find(parent, r) for i, r in label.items()}


def merge_flat(weighted, areas, edges, label, angle=FLAT_ANGLE, forced=None):
    """Merge adjacent regions while their union stays nearly flat, flattest
    pair first, with the same lazy heap as absorb.
    """
    total = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
    mass = collections.defaultdict(float)
    for i, r in label.items():
        for k in range(3):
            total[r][k] += weighted[i][k]
        mass[r] += 2 * areas[i]

    neighbors = collections.defaultdict(set)
    for owners in edges.values():
        if len(owners) == 2:
            ra, rb = label[owners[0]], label[owners[1]]
            if ra != rb:
                neighbors[ra].add(rb)
                neighbors[rb].add(ra)

    def ratio(a, b):
        m = mass[a] + mass[b]
        s = [total[a][k] + total[b][k] for k in range(3)]
        return norm(s) / m if m > 0 else 1.0

    bound = (1 + math.cos(math.radians(angle))) / 2
    ec, rverts, shared_edges = region_topology(edges, label)
    locked = locked_pairs(edges, label, forced)
    alive = set(total)
    parent = {r: r for r in total}
    heap = []
    for a in neighbors:
        for b in neighbors[a]:
            if a < b and ratio(a, b) >= bound:
                heap.append((-ratio(a, b), a, b))
    heapq.heapify(heap)

    while heap:
        negr, a, b = heapq.heappop(heap)
        if (
            a not in alive
            or b not in alive
            or ratio(a, b) != -negr
            or pair(a, b) in locked
            or not keeps_topology(ec, rverts, shared_edges, a, b)
        ):
            continue
        for k in range(3):
            total[a][k] += total[b][k]
        mass[a] += mass[b]
        ec[a] += ec[b] - joint_count(rverts, shared_edges, a, b)
        rverts[a] |= rverts[b]
        for n in neighbors[b]:
            if n != a:
                shared_edges[pair(a, n)] += shared_edges[pair(b, n)]
                if pair(b, n) in locked:
                    locked.add(pair(a, n))
                neighbors[n].add(a)
                neighbors[a].add(n)
            shared_edges.pop(pair(b, n), None)
            locked.discard(pair(b, n))
            neighbors[n].discard(b)
        neighbors[a].discard(a)
        del neighbors[b]
        alive.discard(b)
        parent[b] = a
        for n in neighbors[a]:
            r = ratio(a, n)
            if r >= bound:
                heapq.heappush(heap, (-r, *((a, n) if a < n else (n, a))))

    return {i: find(parent, r) for i, r in label.items()}


def close_rings(verts, weighted, areas, edges, label, angle=CREASE_ANGLE, forced=None):
    """Merge disk pairs whose union is a short annulus, for one cut not two.

    keeps_topology leaves a coarse tube wall as two half shells with two
    seams where a fine one gets a single disk_cuts seam. Closing the ring
    is safe exactly when the cut-open wall unrolls into a compact strip,
    and the boundary must be smooth at its own edges, a lid meeting a
    channel twice turns a corner there and keeps both seams.
    """
    area, shared, perimeter = region_stats(verts, areas, edges, label)
    turns = boundary_turns(verts, weighted, edges, label)
    ec, rverts, shared_edges = region_topology(edges, label)
    locked = locked_pairs(edges, label, forced)

    candidates = []
    for (a, b), length in shared.items():
        if ec[a] != 1 or ec[b] != 1:
            continue
        if (a, b) in locked:
            continue
        if joint_count(rverts, shared_edges, a, b) != 2:
            continue
        if length <= 0 or turns[(a, b)] / length >= angle:
            continue
        strip = (perimeter[a] + perimeter[b]) / 2 - length
        size = area[a] + area[b]
        if size <= 0 or strip * strip / size > SPLIT_ASPECT:
            continue
        candidates.append((strip * strip / size, a, b))

    taken = set()
    closed = {}
    for _, a, b in sorted(candidates):
        if a in taken or b in taken:
            continue
        taken.update((a, b))
        closed[b] = a
    if not closed:
        return label
    return {i: closed.get(r, r) for i, r in label.items()}
