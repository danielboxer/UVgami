"""Post-unwrap repair, in uv space.

A cut-open tube unrolls into a strip as long as the tube's ring, and
the unwrap folds the longest of them, so the 3D shape does not
predict the unwrapped one. This measures the unwrap itself: ruined
islands are cut across and unwrapped again, non-disk islands are
opened, and clean islands are cut at feature necks and when they are
longer than the atlas their own area needs."""

import bisect
import collections
import math

from .cuts import connect_loops, crease_relief, cut_path, edge_cost, path_cost
from .mesh import build, face_edges, find, island_groups, pair, signed_area


# strip test: length squared over uv area, about length/width. a folded
# strip is sliced into bins, anything else ruined is halved.
# close_rings reuses SPLIT_ASPECT, a closed ring past it would unroll into a
# strip the split would cut back up anyway
SPLIT_ASPECT = 6.0
# length alone must not drag texture use under this, so strips are cut to the
# side of the square that would hold the scanned uv area at this use
SPLIT_TARGET = 0.5
# a clean island is scanned for cuts once it passes this fraction of the cap
SPLIT_NECK = 0.5
# width ratio across a slab boundary that reads as a neck: a wide feature
# turning into a strip, where an artist would put the seam
SPLIT_STEP = 2.5
# sweeps of sliding faces across a fresh bin cut while that shortens it,
# capped so the cut cannot creep along a tapering strip
STRAIGHTEN_SWEEPS = 4
# centroids fitting a circle this much tighter than their principal axis
# line read as an unrolled cone, and cuts go polar instead
SPLIT_ARC_BAND = 0.65
# an island thinner than this fraction of its outer distance is a disk,
# not a band, and slicing a disk into sectors helps nothing
SPLIT_ARC_ANNULUS = 0.3
# a split piece below this many faces is a boxed-in fragment, not an
# island, and rejoins a neighbor
SPLIT_MIN_PIECE = 4
# how much a split piece shrinks towards its own centre. blender decides
# islands from uv coordinates, so pieces sharing the cut line exactly stay
# one island however they are seamed, and this is what parts them
SPLIT_GAP = 0.98


def polygon_area(verts, face):
    """3d area by fan, the same fan the flatten solves."""
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


def face_adjacency(group, faces, edges, seams):
    """Face to face links inside one island, as {face: [(other, edge key)]}.
    Only interior edges link, which is the same relation island_groups used
    to build the island, so the graph is connected."""
    adjacent = collections.defaultdict(list)
    in_group = set(group)
    for f in group:
        face = faces[f]
        n = len(face)
        for i in range(n):
            key = pair(face[i], face[(i + 1) % n])
            owners = edges[key]
            if key in seams or len(owners) != 2:
                continue
            other = owners[1] if owners[0] == f else owners[0]
            if other != f and other in in_group:
                adjacent[f].append((other, key))
    return adjacent


def straighten_cut(
    verts, group, faces, edges, adjacent, bin_of, weights=None, relief=None
):
    """Straighten an island's bin cut, returning its seam edges.

    The bin line zigzags on diagonal edge flow. Capped sweeps of sliding
    faces across the cut, strictly shortening, flatten single-face teeth.
    Then each connected run of cut edges is swapped for the shortest
    interior path between its own two endpoints, when that path is strictly
    shorter and the run's two sides still come out separated. That check is
    what stops a path between two far apart boundary ends from running
    lengthwise along a tube and carving off a sliver. Runs that loop or
    branch stay. Both passes count a painted edge as longer, so the cut also
    moves out of restricted areas.
    """
    lengths = {}
    interior = collections.defaultdict(set)
    neighbors = collections.defaultdict(list)
    for f in group:
        for other, key in adjacent[f]:
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

    def separated(source, target, cut_set):
        """Whether the two faces land in different pieces once cut_set is cut."""
        seen = {source}
        stack = [source]
        while stack:
            f = stack.pop()
            for other, key in adjacent[f]:
                if key in cut_set or other in seen:
                    continue
                if other == target:
                    return False
                seen.add(other)
                stack.append(other)
        return True

    blocked = set()
    for cut_edges in cuts.values():
        blocked |= cut_edges

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
            if better:
                candidate = (blocked - set(run)) | set(straight)
                # the run's own two sides must stay apart, or the swap cut
                # the island somewhere else and left the bins joined
                a, b = edges[run[0]]
                if separated(a, b, candidate):
                    blocked = candidate
                else:
                    better = False
            extra.update(straight if better else run)
    return extra


def arc_parameter(group, centroids, areas, size, cx, cy, xx, xy, yy):
    """A polar cut parameter, for islands that unroll curved.

    A cone or tapered tube unrolls into an annulus sector, and a cut at
    one axis position is a chord there: on the mesh it climbs toward the
    thin end, runs along the rim, and comes back down. When the centroids
    fit a circle much tighter than their principal axis line, the cut
    parameter goes polar around the fitted center, whichever direction
    the sector runs longer: arc length when the strip circles the center,
    so cuts are radii, one straight line across the tube on the mesh, or
    distance from the center when the strip runs away from it, so cuts
    are arcs, a flat ring around the tube. Returns (ts, lo, length), or
    None to keep the axis.
    """
    det = xx * yy - xy * xy
    if det <= 0:
        return None
    szx = szy = 0.0
    for f in group:
        dx, dy = centroids[f][0] - cx, centroids[f][1] - cy
        sq = dx * dx + dy * dy
        szx += areas[f] * sq * dx
        szy += areas[f] * sq * dy
    a = (szx * yy - szy * xy) / (2 * det)
    b = (szy * xx - szx * xy) / (2 * det)
    radius_sq = (xx + yy) / size + a * a + b * b
    if radius_sq <= 0:
        return None
    radius = math.sqrt(radius_sq)
    band = 0.0
    dists = {}
    angles = {}
    for f in group:
        dx, dy = centroids[f][0] - cx - a, centroids[f][1] - cy - b
        dists[f] = math.hypot(dx, dy)
        band += areas[f] * (dists[f] - radius) ** 2
        angles[f] = math.atan2(dy, dx)
    across = (xx + yy) / 2 - math.hypot((xx - yy) / 2, xy)
    if across <= 0 or band > SPLIT_ARC_BAND**2 * across:
        return None
    ordered = sorted(angles.values())
    gaps = [nxt - prev for prev, nxt in zip(ordered, ordered[1:])]
    gaps.append(ordered[0] + 2 * math.pi - ordered[-1])
    widest = max(range(len(gaps)), key=lambda i: gaps[i])
    span = 2 * math.pi - gaps[widest]
    r_lo, r_hi = min(dists.values()), max(dists.values())
    if r_hi - r_lo >= span * radius:
        return dists, r_lo, r_hi - r_lo
    # a disk covers every distance down to its center, so binning it by
    # angle would slice a compact island into sectors
    if r_lo < SPLIT_ARC_ANNULUS * r_hi:
        return None
    start = ordered[(widest + 1) % len(ordered)]
    ts = {f: ((t - start) % (2 * math.pi)) * radius for f, t in angles.items()}
    return ts, 0.0, span * radius


def split_pieces(group, links, cuts):
    """The island's connected pieces once cuts are cut."""
    unvisited = set(group)
    pieces = []
    while unvisited:
        seed = unvisited.pop()
        piece = [seed]
        stack = [seed]
        while stack:
            f = stack.pop()
            for other, key in links[f]:
                if other in unvisited and key not in cuts:
                    unvisited.discard(other)
                    piece.append(other)
                    stack.append(other)
        pieces.append(piece)
    return pieces


def absorb_fragments(pieces, links, cuts):
    """Rejoin boxed-in fragments, removing their cuts from cuts.

    A replacement path and an existing seam can box in a mesh sliver,
    leaving an island of a face or two that no packer can use. Such a
    piece rejoins the full-sized neighbor it shares the most cut edges
    with, and only those edges reopen, so the other pieces stay apart.
    A tiny piece among only tiny pieces keeps its cut: that is a small
    island split on purpose, not an accident."""
    changed = True
    while changed and len(pieces) > 1:
        changed = False
        for i, piece in enumerate(pieces):
            if len(piece) >= SPLIT_MIN_PIECE:
                continue
            owner = {f: j for j, p in enumerate(pieces) for f in p}
            shared = collections.defaultdict(set)
            for f in piece:
                for other, key in links[f]:
                    j = owner[other]
                    if key in cuts and j != i and len(pieces[j]) >= SPLIT_MIN_PIECE:
                        shared[j].add(key)
            if not shared:
                continue
            best = max(shared, key=lambda j: len(shared[j]))
            cuts -= shared[best]
            pieces[best].extend(piece)
            del pieces[i]
            changed = True
            break
    return pieces


def strip_cuts(group, ts, lo, length, cap, areas):
    """Cut positions along the axis for one clean island.

    The width profile, uv area per slab of the axis, finds feature necks: a
    hard local step in width is a wide area turning into a strip, where an
    artist would cut. Only the strongest neck is returned: the caller
    re-scans each piece on its own axis, which is what finds the necks a
    bent island smears along its whole-shape axis. With no neck, a strip
    past the cap fills with even cuts, and a compact island is never
    filled, however long."""
    slabs = 24
    slab = [0.0] * slabs
    for f in group:
        slab[min(slabs - 1, int((ts[f] - lo) / length * slabs))] += areas[f]
    width = [a * slabs / length for a in slab]
    smooth = [
        (width[max(0, i - 1)] + width[i] + width[min(slabs - 1, i + 1)]) / 3
        for i in range(slabs)
    ]

    def median(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]

    best = None
    # the profile needs a few faces per slab to mean anything
    if len(group) >= 2 * slabs:
        for j in range(3, slabs - 2):
            # windows local to j, a global median cannot place the step
            left = median(smooth[max(0, j - 3) : j])
            right = median(smooth[j : min(slabs, j + 3)])
            if min(left, right) <= 0:
                continue
            ratio = max(left, right) / min(left, right)
            if ratio > SPLIT_STEP and (best is None or ratio > best[0]):
                best = (ratio, j)
    if best is not None:
        return [lo + best[1] / slabs * length]
    size = sum(slab)
    if size > 0 and length > cap and length * length / size > SPLIT_ASPECT:
        n = math.ceil(length / cap)
        return [lo + length * k / n for k in range(1, n)]
    return []


def split_islands(
    verts, faces, seams, uvs, weights=None, groups=None, edges=None, relief_cache=None
):
    """Extra seam edges that cut ruined uv islands into smaller pieces.

    Runs on the unwrap of the seams this package chose and measures the
    unwrap itself, the 3D shape does not predict it: per island, the extent
    along the principal axis of its face centroids against its uv area,
    measured as arc length instead when the island unrolled into a fan. A
    folded strip bins along that axis and the edges between bins are the
    seams for a second unwrap, a ruined island that is not a strip is
    halved, and a non-disk island is opened with a path joining two of its
    boundary loops, splitting cannot fix topology. A clean island long
    enough to matter is cut at its strongest feature neck and its pieces
    are scanned again on their own axes, and a neckless strip longer than
    the atlas its scanned area needs is sliced even, since one long strip
    caps how far everything can be scaled up. groups restricts the scan,
    for a caller that knows the rest is unchanged.
    """
    if edges is None:
        edges = face_edges(faces)
    if groups is None:
        groups = island_groups(faces, seams, edges)
    if relief_cache is None:
        relief_cache = []

    def cut_relief():
        # the normals and relief only pay off once a cut is actually needed,
        # and most calls find nothing to cut. a caller scanning many pieces
        # of one mesh shares the cache so the whole-mesh build runs once
        if not relief_cache:
            weighted, _, _ = build(verts, faces)
            relief_cache.append(crease_relief(verts, faces, weighted, edges))
        return relief_cache[0]

    extra = set()

    # every length compares at even texel density, uv lengths scaled by
    # sqrt(3d area over uv area): the engine packs each island at its own
    # scale, and a long island exported small must not slip the gate it
    # would fail once packing evens the densities out
    def measure(group):
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
        area3d = sum(polygon_area(verts, faces[f]) for f in group)
        if len(group) < 2 or size <= 0 or area3d <= 0:
            return None
        cx = sum(areas[f] * centroids[f][0] for f in group) / size
        cy = sum(areas[f] * centroids[f][1] for f in group) / size
        xx = xy = yy = 0.0
        for f in group:
            dx, dy = centroids[f][0] - cx, centroids[f][1] - cy
            xx += areas[f] * dx * dx
            xy += areas[f] * dx * dy
            yy += areas[f] * dy * dy
        arc = arc_parameter(group, centroids, areas, size, cx, cy, xx, xy, yy)
        if arc is not None:
            ts, lo, length = arc
        else:
            angle = 0.5 * math.atan2(2 * xy, xx - yy)
            axis = (math.cos(angle), math.sin(angle))
            ts = {
                f: centroids[f][0] * axis[0] + centroids[f][1] * axis[1] for f in group
            }
            lo, hi = min(ts.values()), max(ts.values())
            length = hi - lo
        return ts, lo, length, size, areas, math.sqrt(area3d / size)

    # an island too small to cut still takes up the atlas
    total = sum(polygon_area(verts, faces[f]) for g in groups for f in g)
    if total <= 0:
        return extra
    cap = math.sqrt(total / SPLIT_TARGET)

    # an island is what the unwrap made one: faces joined by unseamed edges
    queue = collections.deque(groups)
    while queue:
        group = queue.popleft()
        m = measure(group)
        if m is None:
            continue
        ts, lo, length, size, areas, density = m
        # the cap in this island's own uv units
        local_cap = cap / density
        clean = not island_ruined(group, faces, uvs, edges, seams)
        if clean:
            if length <= 0 or length <= local_cap * SPLIT_NECK:
                continue
            cuts_at = strip_cuts(group, ts, lo, length, local_cap, areas)
            if not cuts_at:
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
        else:
            if length <= 0:
                # every centroid projects to one point (doubled faces), no cut
                continue
            # a ruined strip past the cap slices even, anything else halves
            strip = length * length / size > SPLIT_ASPECT and length > local_cap
            bins = math.ceil(length / local_cap) if strip else 2
            cuts_at = [lo + length * k / bins for k in range(1, bins)]
        links = face_adjacency(group, faces, edges, seams)
        bin_of = {f: bisect.bisect(cuts_at, ts[f]) for f in group}
        new = straighten_cut(
            verts, group, faces, edges, links, bin_of, weights, cut_relief()
        )
        if not new:
            continue
        pieces = split_pieces(group, links, new)
        pieces = absorb_fragments(pieces, links, new)
        if not new:
            continue
        extra |= new
        # a bent island hides its next neck until each piece is measured on
        # its own axis, so the pieces go back through the scan
        if clean and len(pieces) > 1:
            queue.extend(pieces)
    return extra


def split_moves(verts, faces, uvs, starts, ranges=None):
    """New uvs that slice the long strips out of a preseeded engine output,
    as (loop index, u, v) triples. Plain data in and out, no bpy, so a
    caller can run it off the main thread.

    The engine leaves a developable strip whole because splitting it gains
    no distortion, so split_islands slices those (its cuts snap to creases).
    A long strip packs badly, sliced pieces fill the atlas.

    ranges are (start, stop) polygon index ranges to scan, None for the
    whole mesh, so the organic pieces of a mixed output are never scanned.
    The joined output concatenates each piece's faces unwelded, so a uv
    island lies inside one piece and its first face decides which.

    The pieces are never re-unwrapped, each keeps its engine uvs exactly and
    only shrinks a little towards its own centre, which is what parts them
    into islands and leaves a valid map behind for the pack to tighten. A
    flipped triangle the engine ships is left for Relax Island: re-unwraps
    tried here made those islands worse, not better."""
    edges = face_edges(faces)
    uv_at = [dict(zip(face, uvs[fi])) for fi, face in enumerate(faces)]
    seams = {
        key
        for key, owners in edges.items()
        if len(owners) == 2
        and any(uv_at[owners[0]][v] != uv_at[owners[1]][v] for v in key)
    }
    groups = island_groups(faces, seams, edges)
    groups = [g for g in groups if not island_ruined(g, faces, uvs, edges, seams)]
    if ranges is None:
        scanned = groups
        extra = split_islands(verts, faces, seams, uvs, None, groups, edges)
    else:
        # one call per piece: each piece's engine output has its own uv scale,
        # so its length cap has to come from its own area alone
        scanned = []
        extra = set()
        relief_cache = []
        for start, stop in ranges:
            scoped = [g for g in groups if start <= g[0] < stop]
            scanned += scoped
            extra |= split_islands(
                verts, faces, seams, uvs, None, scoped, edges, relief_cache
            )
    if not extra:
        return []
    touched = {f for e in extra for f in edges[e]}
    target_faces = {f for g in scanned if touched & set(g) for f in g}
    moves = []
    for piece in island_groups(faces, seams | extra, edges):
        if piece[0] not in target_faces:
            continue
        points = [uv for f in piece for uv in uvs[f]]
        cx = sum(u for u, _ in points) / len(points)
        cy = sum(v for _, v in points) / len(points)
        for f in piece:
            for corner, (u, v) in enumerate(uvs[f]):
                moves.append(
                    (
                        starts[f] + corner,
                        cx + (u - cx) * SPLIT_GAP,
                        cy + (v - cy) * SPLIT_GAP,
                    )
                )
    return moves
