# no bpy imports here so the algorithm stays testable outside blender
"""Feature seams by strip merging, the hard surface Seams mode.

A per-edge angle test cannot seam a beveled model: a bevel splits one crease
into several small turns, so every threshold either seams both sides of the
bevel or misses it entirely. What separates a bevel from a corner is width,
not angle: a corner is a crease with no width, a bevel is the same crease
spread over a narrow band. So partition faces per edge at a low angle (which
over-segments bevels and curved surfaces into narrow bands), then merge away
any region narrower than an auto-detected width into a neighbour, preferring
one already over the threshold so bands dissolve into their surfaces. Then
merge any boundary that turns less than a crease, which is what puts a coarse
cylinder back together: its panels each turn one segment angle, while a corner
turns the whole corner. Absorbing a band moves its turn onto the boundaries it
leaves, so a dissolved bevel still reads as the crease it was. A last pass
merges neighbours whose union is still nearly flat, so shallow features don't
each keep their own chart. On a smooth model a filleted rim reads smooth, so
a wall merges right over its end caps; split_sweeps takes those regions back
apart at the rims, reading wall and cap off the normals against a fitted
sweep axis. Every merge pass refuses anything that would leave
a region with a hole, because optcuts throws a non-disk chart away and re-cuts
the model from scratch. A region that partitions non-disk in the first
place, like a tube wall, is cut open instead. Two disks refused that way whose
union is an annulus are merged after all when the cut-open wall would unroll
into a compact strip (close_rings), so a coarse cylinder gets the same single
cut as a fine one. The surviving region boundaries plus those cuts are the
seams, after two cleanup passes: flatten_teeth relabels the single-face
zigzags the merges leave, and reroute_boundaries redraws each boundary run
between anchored ends as the cheapest nearby path, discounted along creases,
so seams straighten and settle onto sharp edges. Cut searches pay for
turning between dull edges, so a cut across featureless area comes out a
line, the way an artist cuts. A cut-open tube still unrolls into one strip
as long as the tube's ring, and the unwrap folds the longest of them, so
islands that came out long and folded are cut across and unwrapped again
(split_islands). hard_surface retries a folded island with more slim
iterations before resorting to that. Clean islands split only when they are
strips too long to pack: the cut follows the bin line, slid straight where
that shortens it.
Benchmarks against the per-edge test are in docs/agents/bench-results.md.
"""

import collections
import heapq
import math

# partition angle in degrees; low on purpose, over-segmenting is what makes
# region width meaningful, and absorb reassembles the pieces
LOW_ANGLE = 10
# auto width: at a low partition angle the region widths are dominated by
# narrow bands (chamfer strips and curvature bands), real surfaces are the top
# few percent. There is no clean gap between the two on real models, so take a
# high quantile of all widths with clearance on top. The cap (a fraction of
# the diagonal) keeps sparse cases like a beveled cube from merging real faces.
WIDTH_CAP = 0.05
WIDTH_QUANTILE = 0.9
WIDTH_FACTOR = 2.0
# flat merge: a region's flatness is |sum of weighted normals| / (2 * area),
# 1 on a plane, falling as normals spread. Merge neighbours while the union
# stays above the flatness of a spherical cap with this half-angle, so
# shallow creases merge and real corners (~0.7 for a 90 degree pair) survive
FLAT_ANGLE = 30
# smooth merge: how far a region boundary has to turn to count as a crease.
# Flatness cannot pass a cylinder, since its panels spread as far as a corner
# does once enough of them merge, so this reads the turn at the boundary
# instead, which stays one segment angle however far the wall comes round
CREASE_ANGLE = 30
# sweep split: a smooth model has no crease at a cylinder's rim, so the wall
# merges over the end cap into a "sock", a disk the unwrap flattens as a
# polar map: near-isometric, but the texture direction winds around the cap
# instead of following the axis. No distortion measure catches that, so the
# structure is read off the normals instead: against the right axis a swept
# region's faces are either wall (normal near perpendicular) or cap (normal
# near axial), with little in between, while a bent tube fills the middle
# band. Fractions of region area: a region splits at its rims when the
# 30-60 degree band holds under BAND, and has caps worth cutting when the
# over-45 share is at least CAP_MIN (a measured elbow reads band 0.31, a
# screwdriver handle 0.07, its shaft 0.01)
SWEEP_BAND = 0.1
SWEEP_CAP_MIN = 0.02
# only regions holding this share of the model's area are worth rim cuts: a
# tiny cylinder unwraps as a tiny polar blob nobody sees, and cutting every
# screw and pin explodes the chart count (circuit_board 751 to 1235 regions)
SWEEP_MIN_SHARE = 0.01
# a plate with tilted flanges reads as a degenerate wall (any in-plane axis
# fits), and rims on a plate make no sense: the wall must actually turn
# around the axis, so its normals' resultant length over its mass has to
# fall below this (1 on a plate, sin(t/2)/(t/2) for an arc of t, 0 on a
# full wall: 0.7 asks for roughly a half turn)
WALL_ROUND = 0.7
# wall/cap boundary and the band edges, as squared sines of the tilt
CAP_SPLIT = 0.5  # 45 degrees
BAND_LO = 0.25  # 30 degrees
BAND_HI = 0.75  # 60 degrees
# alternating fit rounds for the sweep axis
SWEEP_FIT_ROUNDS = 10
# strip split: length squared over uv area, about length/width for a strip.
# Only islands whose unwrap already folded are candidates, and these decide
# how they are cut: a strip (aspect over SPLIT_ASPECT and longer than
# SPLIT_LENGTH in uv units, the unwrap packs into the unit square) is sliced
# into aspect-bound bins, anything else ruined is halved.
# close_rings reuses SPLIT_ASPECT as its bound: a closed ring past it would
# unroll into a strip the split would cut back up anyway
SPLIT_ASPECT = 6.0
SPLIT_LENGTH = 0.25
# straighten: sweeps of sliding faces across a fresh bin cut while that
# shortens it. Teeth flatten in one or two, and the cap keeps the cut from
# creeping along the strip
STRAIGHTEN_SWEEPS = 4
# guided mode: what a fully painted vertex multiplies an edge's length by
# when a cut is placed over it. Bounded on purpose, an infinite cost would
# make a painted band uncrossable and drop the cut instead of moving it, so a
# path that has to cross one crosses at its narrowest
RESTRICT_COST = 9.0
# free cuts prefer sharp edges: a crease edge counts shorter, sliding from
# full length at LOW_ANGLE down to a floor at RELIEF_FULL_ANGLE, where an
# edge is a definite crease: on a beveled model the crease is spread over
# edges turning far less than 90, and steeper turns hide a seam no better.
# Concave gets the deeper discount, a groove hides a seam best, and the
# floors are mild so a cut never wanders far hunting for a crease
CONCAVE_RELIEF = 0.5
CONVEX_RELIEF = 0.3
RELIEF_FULL_ANGLE = 45
# reroute: a boundary chain may only move within this many face rings of
# where it already is, so it can snap to a crease right beside it but never
# shortcut across a region
REROUTE_RINGS = 2
# relief below this counts an edge as creased, for the reroute guards
CREASED_RELIEF = 0.9
# a closed-loop boundary has no junctions to hold it, so it is split at its
# sharpest vertices and the arcs rerouted like runs, but only when at least
# this share of its edges is creased: a dull loop has nothing to snap to
# and by length alone it would just drift
LOOP_SHARE = 0.5
# a cut crossing dull geometry should be a line, the way an artist cuts
# across a flat area: each step turning between two dull edges costs up to
# this fraction of its length extra, so among near-equal paths the straight
# one wins. Creased edges are exempt, a seam follows a crease around any
# corner it takes
TURN_COST = 1.0


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


def build(verts, faces):
    """Per-face weighted normals and areas, plus edge -> owning faces."""
    weighted, areas = [], []
    for face in faces:
        a, b, c = (verts[i] for i in face[:3])
        n = cross([b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)])
        weighted.append(n)
        areas.append(norm(n) / 2)
    return weighted, areas, face_edges(faces)


# per region boundary, length-weighted sums the smooth merge reads: the turn
# carried across dissolved bands, the turn at the boundary's own edges, the
# width that carry crossed, and the boundary length to divide any of them by
Boundaries = collections.namedtuple("Boundaries", "turn step spread length")


def turn_angle(weighted, owners):
    """Degrees the surface turns across an edge, from its two face normals."""
    na, nb = weighted[owners[0]], weighted[owners[1]]
    scale = norm(na) * norm(nb)
    if not scale:
        return 0.0
    dot = sum(x * y for x, y in zip(na, nb)) / scale
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def partition(faces, weighted, edges, angle, forced=None):
    """Union-find over faces, cutting every edge sharper than angle, and every
    edge in forced whatever it turns."""
    parent = list(range(len(faces)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for key, owners in edges.items():
        if len(owners) != 2:
            continue
        if turn_angle(weighted, owners) > angle:
            continue
        if forced and key in forced:
            continue
        a, b = owners
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return find


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
    anything that lowers it, because the engine throws a non-disk chart away
    and re-cuts the model from scratch."""
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


def pair(a, b):
    return (a, b) if a < b else (b, a)


def min_eigenvector(xx, yy, zz, xy, xz, yz):
    """Unit eigenvector for the smallest eigenvalue of a symmetric 3x3
    matrix: trigonometric eigenvalue, then the cross of the two most
    independent rows of the shifted matrix."""
    p1 = xy * xy + xz * xz + yz * yz
    if p1 == 0:
        lam = min(xx, yy, zz)
    else:
        q = (xx + yy + zz) / 3
        p2 = (xx - q) ** 2 + (yy - q) ** 2 + (zz - q) ** 2 + 2 * p1
        p = math.sqrt(p2 / 6)
        bxx, byy, bzz = (xx - q) / p, (yy - q) / p, (zz - q) / p
        bxy, bxz, byz = xy / p, xz / p, yz / p
        det = (
            bxx * (byy * bzz - byz * byz)
            - bxy * (bxy * bzz - byz * bxz)
            + bxz * (bxy * byz - byy * bxz)
        )
        r = max(-1.0, min(1.0, det / 2))
        phi = math.acos(r) / 3
        lam = q + 2 * p * math.cos(phi + 2 * math.pi / 3)
    rows = [(xx - lam, xy, xz), (xy, yy - lam, yz), (xz, yz, zz - lam)]
    best, best_norm = None, 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            c = cross(rows[i], rows[j])
            n = norm(c)
            if n > best_norm:
                best, best_norm = c, n
    if best is None:
        return (1.0, 0.0, 0.0)  # degenerate: any direction is an eigenvector
    return tuple(x / best_norm for x in best)


def sweep_axis(normals):
    """Axis against which the normals split into wall and cap, if any.

    normals is a list of (area, unit normal). Starts from the plane best
    fitting all normals (right already for a pure wall) and alternates:
    classify each normal wall or cap against the axis, then refit with cap
    normals repelled instead of attracted, which pulls the axis toward
    perpendicularity with the walls only."""

    def fit(entries):
        m = [0.0] * 6
        for w, n in entries:
            m[0] += w * n[0] * n[0]
            m[1] += w * n[1] * n[1]
            m[2] += w * n[2] * n[2]
            m[3] += w * n[0] * n[1]
            m[4] += w * n[0] * n[2]
            m[5] += w * n[1] * n[2]
        return min_eigenvector(*m)

    axis = fit(normals)
    for _ in range(SWEEP_FIT_ROUNDS):
        signed = []
        for w, n in normals:
            axial = (n[0] * axis[0] + n[1] * axis[1] + n[2] * axis[2]) ** 2
            signed.append((w if axial < CAP_SPLIT else -w, n))
        refit = fit(signed)
        done = abs(abs(sum(x * y for x, y in zip(axis, refit))) - 1) < 1e-12
        axis = refit
        if done:
            break
    return axis


def split_sweeps(weighted, areas, edges, label):
    """Split swept regions into wall and cap parts, seaming their rims.

    See SWEEP_BAND: this is what stops a smooth cylinder-like region from
    keeping its end caps and unwrapping as a polar map. A region whose
    normals sort cleanly into wall and cap against its sweep axis is
    relabeled by connected component of that classification, so each cap
    lifts off as its own region and the wall keeps unrolling; a wall left
    as an annulus is disk_cuts' job, same as any other. Any component too
    small to be a real cap is merged back into the neighbour it touches
    most, since a rim seam is not worth a speck. Regions with real
    middle-band mass, bent tubes and organic blobs, are left as they
    are."""
    members = collections.defaultdict(list)
    for i, r in label.items():
        members[r].append(i)
    model_area = sum(areas)

    relabel = {}
    for r, group in members.items():
        axial = {}
        normals = []
        for i in group:
            n = weighted[i]
            length = norm(n)
            if length:
                normals.append((i, areas[i], tuple(x / length for x in n)))
        total = sum(w for _, w, _ in normals)
        if len(normals) < 8 or total < SWEEP_MIN_SHARE * model_area:
            continue
        axis = sweep_axis([(w, n) for _, w, n in normals])
        band = cap = 0.0
        for i, w, n in normals:
            axial[i] = (n[0] * axis[0] + n[1] * axis[1] + n[2] * axis[2]) ** 2
            if BAND_LO < axial[i] < BAND_HI:
                band += w
            if axial[i] >= CAP_SPLIT:
                cap += w
        if band > SWEEP_BAND * total:
            continue
        if not SWEEP_CAP_MIN * total <= cap <= total / 2:
            continue
        wall_sum = [0.0, 0.0, 0.0]
        for i, w, n in normals:
            if axial[i] < CAP_SPLIT:
                for k in range(3):
                    wall_sum[k] += w * n[k]
        if norm(wall_sum) / (total - cap) > WALL_ROUND:
            continue
        is_cap = {i: axial.get(i, 0.0) >= CAP_SPLIT for i in group}
        in_group = set(group)

        # connected components of same-class faces over the region's edges
        parent = {i: i for i in group}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        contact = collections.Counter()
        for owners in edges.values():
            if len(owners) != 2:
                continue
            a, b = owners
            if a not in in_group or b not in in_group:
                continue
            if is_cap[a] == is_cap[b]:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
            else:
                contact[(a, b)] += 1

        comp_area = collections.defaultdict(float)
        for i in group:
            comp_area[find(i)] += areas[i]
        # a rim seam is not worth a speck: merge any component smaller than
        # a real cap into whichever neighbour it touches most
        while True:
            speck = min(comp_area, key=comp_area.get)
            if comp_area[speck] >= SWEEP_CAP_MIN * total or len(comp_area) < 2:
                break
            touch = collections.Counter()
            for (a, b), c in contact.items():
                ra, rb = find(a), find(b)
                if ra == rb:
                    continue
                if speck in (ra, rb):
                    touch[rb if ra == speck else ra] += c
            if not touch:
                break
            into = touch.most_common(1)[0][0]
            parent[speck] = into
            comp_area[into] += comp_area.pop(speck)
        if len(comp_area) < 2:
            continue
        for i in group:
            relabel[i] = find(i)

    if not relabel:
        return label
    return {i: relabel.get(i, r) for i, r in label.items()}


def joint_count(rverts, shared, a, b):
    """Shared vertices minus shared edges: 1 when the two regions meet along a
    single path, 2 when they meet twice (so the union has a hole), 0 when the
    contact is a closed loop (so the union is closed)."""
    return len(rverts[a] & rverts[b]) - shared[pair(a, b)]


def keeps_topology(ec, rverts, shared, a, b):
    """A merge is allowed when the union is no worse than the worse of the two
    and is not a closed surface. Two disks meeting along one path stay a disk;
    meeting twice opens a hole, so that is refused, while swallowing the region
    that sits inside a hole closes one and is exactly what we want."""
    merged = ec[a] + ec[b] - joint_count(rverts, shared, a, b)
    return min(ec[a], ec[b]) <= merged <= 1


def locked_pairs(edges, label, forced):
    """Region pairs whose shared boundary holds a forced seam. No merge pass
    may take one, so a hand-marked edge survives as a region boundary and the
    passes route around it: a band with one side locked dissolves into the
    other, instead of leaving a ribbon between the two seams."""
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


def detect_width(verts, faces, areas, edges, find, scale):
    """Absolute merge width from a high quantile of the region widths."""
    label = {i: find(i) for i in range(len(faces))}
    area, _, perimeter = region_stats(verts, areas, edges, label)
    ws = sorted(2 * area[r] / perimeter[r] for r in area if perimeter[r] > 0)
    if not ws:
        return 0.0
    quantile = ws[min(len(ws) - 1, int(WIDTH_QUANTILE * len(ws)))]
    return min(WIDTH_FACTOR * quantile, WIDTH_CAP * scale)


def absorb(verts, faces, weighted, areas, edges, find, min_width, forced=None):
    """Repeatedly merge the narrowest region into its longest-shared neighbour.

    Stats update incrementally per merge: only the merged pair changes, every
    other region's area and perimeter stay as they were. The heap holds lazy
    entries, an entry whose width no longer matches the region is stale.

    Returns per-boundary sums for the smooth merge along with the labels.
    Dissolving a band moves its turn onto the boundaries left behind: crossing
    from the surface that swallowed a bevel to the one on the far side still
    turns the whole crease, which is what stops the smooth merge from erasing
    it. The band's width rides along too, because a crease is a turn with no
    width and a chain of dissolved curvature bands is the same turn spread
    wide. step is the turn at the boundary's own edges, which no dissolve can
    move, so a real corner keeps its crease whatever was absorbed near it."""
    label = {i: find(i) for i in range(len(faces))}
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

    def resolve(r):
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        return r

    bounds = Boundaries(turns, steps, spread, shared)
    return {i: resolve(r) for i, r in label.items()}, bounds


def merge_smooth(edges, label, bounds, min_width, angle=CREASE_ANGLE, forced=None):
    """Merge neighbours whose shared boundary is not a crease.

    A crease is a turn with no width, so what tells a corner from a curved
    surface is the turn concentrated at the boundary, not how far the union
    spreads: every boundary on a cylinder wall turns one segment angle however
    much of the wall has already merged, while a corner turns the whole corner.
    Flatness cannot make that call, a half cylinder is as spread as a right
    angle. absorb hands over the turn of the boundaries it dissolved, so a
    bevel that has been absorbed into its surface still reads as a crease,
    while a chain of dissolved curvature bands spreads its turn wider than a
    band ever is and reads as the curved surface it came from. That carry is
    only trusted while it stays narrow: past that the boundary is judged by
    its own edges alone, so a corner that happens to have a band ending on it
    is not smoothed away.

    Least turn first, with the same lazy heap as the other passes."""
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
        """How sharply the surface turns crossing this boundary.

        The turn at its own edges always counts. What a dissolved band carried
        counts on top of that while it is still band width, past which the
        carry is a surface curving and only the edges themselves are read."""
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

    def resolve(r):
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        return r

    return {i: resolve(r) for i, r in label.items()}


def merge_flat(weighted, areas, edges, label, angle=FLAT_ANGLE, forced=None):
    """Merge adjacent regions while their union stays nearly flat.

    Flattest pair first, with the same lazy heap as absorb: an entry whose
    recomputed ratio differs is stale, the merge that changed it already
    pushed a fresh one."""
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

    def resolve(r):
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        return r

    return {i: resolve(r) for i, r in label.items()}


def close_rings(verts, weighted, areas, edges, label, angle=CREASE_ANGLE, forced=None):
    """Merge disk pairs whose union is a short annulus, for one cut not two.

    keeps_topology refuses any merge that opens a hole, so a coarse tube wall
    ends as two half shells with two seams where a fine one gets a single
    disk_cuts seam. Closing the ring is safe exactly when the cut-open wall
    unrolls into a compact strip: its length is half the union perimeter (the
    two rims), and past the bound the unwrap folds it and split_islands cuts
    it back up, so nothing would be won. The boundary must also be smooth at
    its own edges, a lid meeting a channel twice turns a corner there and
    keeps both seams."""
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


def flatten_teeth(weighted, faces, edges, label, angle=CREASE_ANGLE, forced=None):
    """Zigzag teeth relabeled away so region boundaries follow clean chains.

    The merges settle face by face, so a boundary staircases along a feature
    instead of sitting on one edge chain. A tooth is a face with every edge
    but one on the boundary to the same neighbour: relabeling it swaps those
    edges for the one it kept, the boundary strictly shortens, and the sweep
    terminates. Removing an ear held by a single edge cannot break either
    region's disk topology. A corner face on a real crease looks like a tooth
    too, so a flip never trades crease edges for a dull one: the kept edge
    must be sharp, or every lost edge dull."""
    turns = {}

    def sharp(key):
        owners = edges[key]
        if len(owners) != 2:
            return True  # a mesh rim hides a seam as well as a crease
        if key not in turns:
            turns[key] = turn_angle(weighted, owners)
        return turns[key] >= angle

    label = dict(label)
    queue = collections.deque(label)
    queued = set(queue)
    while queue:
        f = queue.popleft()
        queued.discard(f)
        keys = face_keys(faces[f])
        lost = collections.defaultdict(list)
        for key in keys:
            owners = edges[key]
            if len(owners) == 2:
                g = owners[owners[0] == f]
                if label[g] != label[f]:
                    lost[label[g]].append(key)
        if not lost:
            continue
        target, gone = max(lost.items(), key=lambda kv: len(kv[1]))
        if len(gone) != len(keys) - 1:
            continue
        kept = next(k for k in keys if k not in gone)
        if forced and not forced.isdisjoint(gone):
            continue
        if not sharp(kept) and any(sharp(k) for k in gone):
            continue
        label[f] = target
        for key in keys:
            for o in edges[key]:
                if o != f and o not in queued:
                    queued.add(o)
                    queue.append(o)
    return label


def reroute_boundaries(verts, faces, areas, edges, label, relief, forced=None):
    """Region boundaries redrawn as the cheapest paths under crease relief.

    flatten_teeth removes single-face zigzags, this moves whole runs: each
    stretch of a two-region boundary between anchored vertices (a rim, or
    where another boundary meets it) is re-routed as the cheapest path
    between the same two ends, so a staircase straightens and a seam one
    edge off a crease drops onto it. The path stays within REROUTE_RINGS of
    the old run and off every other seam, and the relabel is refused unless
    it splits the two regions' union into exactly two pieces with topology
    no worse, so a reroute can move a seam but never a junction, and never
    opens a hole. A closed loop that is mostly creased is split at its
    sharpest vertices into three arcs and each rerouted the same way; a
    mostly dull loop stays put."""
    label = dict(label)

    vert_faces = collections.defaultdict(list)
    for fi, face in enumerate(faces):
        for v in face:
            vert_faces[v].append(fi)

    # boundary edges per region pair, and the vertices no path may pass
    # through: everything on a rim or a seam, its own run's vertices excepted
    pair_keys = collections.defaultdict(list)
    anchored = set()
    vert_pairs = collections.defaultdict(set)
    for key, owners in edges.items():
        if len(owners) != 2:
            anchored.update(key)
            continue
        ra, rb = label[owners[0]], label[owners[1]]
        if ra == rb:
            continue
        pair_keys[pair(ra, rb)].append(key)
        vert_pairs[key[0]].add(pair(ra, rb))
        vert_pairs[key[1]].add(pair(ra, rb))
    blocked = anchored | set(vert_pairs)

    def loop_arcs(loop, cycle):
        n = len(loop)
        creased = sum(1 for k in loop if relief.get(k, 1.0) < CREASED_RELIEF)
        if creased < LOOP_SHARE * n:
            return
        # a vertex is as sharp as the duller of its two loop edges, so both
        # anchors of an arc sit where the seam is already right
        sharp = [
            1.0 - max(relief.get(loop[i - 1], 1.0), relief.get(loop[i], 1.0))
            for i in range(n)
        ]
        anchors = []
        gap = max(1, n // 4)
        for i in sorted(range(n), key=lambda i: sharp[i], reverse=True):
            if all(min((i - a) % n, (a - i) % n) >= gap for a in anchors):
                anchors.append(i)
            if len(anchors) == 3:
                break
        if len(anchors) < 2:
            return
        anchors.sort()
        for a, b in zip(anchors, anchors[1:] + anchors[:1]):
            arc = loop[a:b] if a < b else loop[a:] + loop[:b]
            yield arc, cycle[a], cycle[b]

    def chains(keys):
        deg = collections.Counter()
        incident = collections.defaultdict(list)
        for key in keys:
            for v in key:
                deg[v] += 1
                incident[v].append(key)
        junctions = {
            v for v in deg if deg[v] != 2 or v in anchored or len(vert_pairs[v]) > 1
        }
        seen = set()
        for j in junctions:
            for start in incident[j]:
                if start in seen:
                    continue
                seen.add(start)
                run = [start]
                v = start[start[0] == j]
                while v not in junctions:
                    key = next(k for k in incident[v] if k != run[-1])
                    seen.add(key)
                    run.append(key)
                    v = key[key[0] == v]
                if v != j:
                    yield run, j, v
        left = set(keys) - seen
        while left:
            start = left.pop()
            loop = [start]
            cycle = [start[0], start[1]]
            v = start[1]
            while True:
                key = next((k for k in incident[v] if k in left), None)
                if key is None:
                    break
                left.discard(key)
                loop.append(key)
                v = key[key[0] == v]
                cycle.append(v)
            # anything under three arcs of a few edges each is too small to move
            if cycle[-1] != cycle[0] or len(loop) < 9:
                continue
            cycle.pop()
            yield from loop_arcs(loop, cycle)

    def ec(group):
        ks = {key for f in group for key in face_keys(faces[f])}
        vs = {v for key in ks for v in key}
        return len(vs) - len(ks) + len(group)

    # the search relief squared: moving an existing boundary should chase a
    # crease harder than a free cut does, or a mild discount loses to a
    # dull shortcut straight across the surface
    pull = {k: r * r for k, r in relief.items()}

    def run_seq(run, j0):
        seq = [j0]
        for key in run:
            seq.append(key[1] if key[0] == seq[-1] else key[0])
        return seq

    def creased_share(keys):
        total = creased = 0.0
        for a, b in keys:
            length = norm([verts[a][i] - verts[b][i] for i in range(3)])
            total += length
            if relief.get(pair(a, b), 1.0) < CREASED_RELIEF:
                creased += length
        return creased / total if total else 0.0

    def reroute(p, run, j0, j1):
        ra, rb = p
        for key in run:
            owners = edges[key]
            if {label[owners[0]], label[owners[1]]} != {ra, rb}:
                return  # an earlier reroute moved this stretch, leave it
        if forced and not forced.isdisjoint(run):
            return
        run_set = set(run)
        run_verts = {v for key in run for v in key}

        corridor = set()
        ring = run_verts
        for _ in range(REROUTE_RINGS):
            grown = {
                f
                for v in ring
                for f in vert_faces[v]
                if label[f] in p and f not in corridor
            }
            corridor |= grown
            ring = {v for f in grown for v in faces[f]}
        allowed = {v for f in corridor for v in faces[f]} - blocked | run_verts
        adjacent = collections.defaultdict(set)
        for f in corridor:
            for key in face_keys(faces[f]):
                a, b = key
                if a not in allowed or b not in allowed:
                    continue
                owners = edges[key]
                if len(owners) == 2 and {label[o] for o in owners} <= set(p):
                    adjacent[a].add(b)
                    adjacent[b].add(a)

        path = cut_path(verts, adjacent, {j0}, {j1}, relief=pull)
        if not path:
            return
        new = {pair(a, b) for a, b in zip(path, path[1:])}
        if new == run_set or path_cost(verts, path, relief=pull) >= path_cost(
            verts, run_seq(run, j0), relief=pull
        ):
            return
        # a reroute puts seams on creases or straightens dull ones, it never
        # trades crease for shortcut
        if creased_share(new) < creased_share(run_set):
            return

        # split the union along the new path: other seams still divide, the
        # old run no longer does
        union = {f for f in label if label[f] in p}
        parent = {f: f for f in union}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for f in union:
            for key in face_keys(faces[f]):
                if key in new:
                    continue
                owners = edges[key]
                if len(owners) != 2:
                    continue
                g = owners[owners[0] == f]
                if g not in parent:
                    continue
                if label[f] != label[g] and key not in run_set:
                    continue
                fa, fb = find(f), find(g)
                if fa != fb:
                    parent[fa] = fb
        comps = collections.defaultdict(list)
        for f in union:
            comps[find(f)].append(f)
        if len(comps) != 2:
            return
        one, two = comps.values()

        def lean(group):
            return sum(areas[f] if label[f] == ra else -areas[f] for f in group)

        if lean(one) == lean(two):
            return
        if lean(two) > lean(one):
            one, two = two, one
        floor = min(
            ec([f for f in union if label[f] == ra]),
            ec([f for f in union if label[f] == rb]),
        )
        if max(ec(one), ec(two)) > 1 or min(ec(one), ec(two)) < floor:
            return
        for f in one:
            label[f] = ra
        for f in two:
            label[f] = rb
        blocked.update(path)

    for p, keys in pair_keys.items():
        for run, j0, j1 in chains(keys):
            reroute(p, run, j0, j1)
    return label


def boundary_edges(edges, label):
    """Edges between two regions, as sorted vertex index pairs."""
    return {
        pair for pair, owners in edges.items() if len({label[o] for o in owners}) > 1
    }


def boundary_components(edges, label):
    """Per-region boundary vertices, grouped into connected components.

    Two loops meeting at a vertex count as one component: the cut between them
    would be a point, not a path, and the region is already joined there."""
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (v0, v1), owners in edges.items():
        regions = {label[o] for o in owners}
        if len(owners) == 2 and len(regions) == 1:
            continue
        for r in regions:
            a, b = (r, v0), (r, v1)
            parent.setdefault(a, a)
            parent.setdefault(b, b)
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    grouped = collections.defaultdict(lambda: collections.defaultdict(set))
    for region, vert in parent:
        grouped[region][find((region, vert))].add(vert)
    return {r: list(comps.values()) for r, comps in grouped.items()}


def crease_relief(verts, faces, weighted, edges):
    """Per edge, a length factor under 1 where the surface creases, so free
    cuts prefer sharp edges over wandering across flat triangles. The sign
    is read off the neighbour's centroid against the face plane: risen means
    concave, a groove that hides a seam. Flat edges are left out."""
    centroids = [
        [sum(verts[v][i] for v in face) / len(face) for i in range(3)] for face in faces
    ]
    relief = {}
    for key, owners in edges.items():
        if len(owners) != 2:
            continue
        angle = turn_angle(weighted, owners)
        if angle <= LOW_ANGLE:
            continue
        na = weighted[owners[0]]
        base = verts[key[0]]
        lift = sum(na[i] * (centroids[owners[1]][i] - base[i]) for i in range(3))
        depth = min((angle - LOW_ANGLE) / (RELIEF_FULL_ANGLE - LOW_ANGLE), 1.0)
        relief[key] = 1 - (CONCAVE_RELIEF if lift > 0 else CONVEX_RELIEF) * depth
    return relief


def edge_cost(verts, weights, a, b, relief=None):
    """An edge's length, longer where a painted restriction repels cuts and
    shorter along a crease, so cuts land on clean lines."""
    length = norm([verts[a][i] - verts[b][i] for i in range(3)])
    if relief:
        length *= relief.get(pair(a, b), 1.0)
    if not weights:
        return length
    paint = (weights.get(a, 0.0) + weights.get(b, 0.0)) / 2
    return length * (1 + RESTRICT_COST * paint)


def turn_cost(verts, u, v, w, relief):
    """Extra cost fraction for the step v->w after arriving from u.

    Zero along a crease or on a straight continuation, up to TURN_COST on a
    reversal, so a path over dull geometry pays for every direction change."""
    if (
        relief.get(pair(u, v), 1.0) < CREASED_RELIEF
        or relief.get(pair(v, w), 1.0) < CREASED_RELIEF
    ):
        return 0.0
    a = [verts[v][i] - verts[u][i] for i in range(3)]
    b = [verts[w][i] - verts[v][i] for i in range(3)]
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    cos = sum(a[i] * b[i] for i in range(3)) / (na * nb)
    return TURN_COST * (1.0 - cos) / 2.0


def path_cost(verts, seq, weights=None, relief=None):
    """Cost of an ordered vertex path, turn penalties included, so a
    comparison values straightness the same way cut_path searches for it."""
    total = 0.0
    for i in range(1, len(seq)):
        step = edge_cost(verts, weights, seq[i - 1], seq[i], relief)
        if relief is not None and i >= 2:
            step *= 1.0 + turn_cost(verts, seq[i - 2], seq[i - 1], seq[i], relief)
        total += step
    return total


def cut_path(verts, adjacent, sources, targets, weights=None, relief=None):
    """Cheapest path from any source vertex to any target, over adjacent.

    Painted restrictions count as extra length, so the path bends around
    them where it can and takes the shortest way through where it cannot.
    With relief the state carries the incoming direction and turning between
    dull edges costs extra, so a cut across featureless area comes out a
    line instead of the staircase that happens to be shortest."""
    if relief is None:
        dist = {v: 0.0 for v in sources}
        prev = {}
        heap = [(0.0, v) for v in sources]
        heapq.heapify(heap)
        while heap:
            d, v = heapq.heappop(heap)
            if d != dist[v]:
                continue
            if v in targets:
                path = [v]
                while path[-1] in prev:
                    path.append(prev[path[-1]])
                return path
            for w in adjacent[v]:
                step = d + edge_cost(verts, weights, v, w, relief)
                if step < dist.get(w, math.inf):
                    dist[w] = step
                    prev[w] = v
                    heapq.heappush(heap, (step, w))
        return []

    # state is (vertex, arrived-from), -1 for a start with no direction yet
    dist = {(v, -1): 0.0 for v in sources}
    prev = {}
    heap = [(0.0, v, -1) for v in sources]
    heapq.heapify(heap)
    while heap:
        d, v, u = heapq.heappop(heap)
        if d != dist[v, u]:
            continue
        if v in targets:
            path = [v]
            node = (v, u)
            while node in prev:
                node = prev[node]
                path.append(node[0])
            return path
        for w in adjacent[v]:
            step = edge_cost(verts, weights, v, w, relief)
            if u >= 0:
                step *= 1.0 + turn_cost(verts, u, v, w, relief)
            step += d
            if step < dist.get((w, v), math.inf):
                dist[w, v] = step
                prev[w, v] = (v, u)
                heapq.heappush(heap, (step, w, v))
    return []


def snap_paths(verts, adjacent, mapped, cuts):
    """Redraw another mesh's cut network on this one, edge by edge.

    Each cut edge becomes the shortest path between the vertices its two ends
    map to. Every vertex over there maps to one vertex here, so segments that
    met still meet and a boundary loop stays the closed loop it was, which is
    what makes a cut network survive the trip. A segment whose ends land on
    one vertex shortens to nothing, and one with no path between them is
    dropped, both of which leave the rest of the network intact."""
    paths = set()
    for a, b in cuts:
        va, vb = mapped[a], mapped[b]
        if va == vb:
            continue
        path = cut_path(verts, adjacent, {va}, {vb})
        for x, y in zip(path, path[1:]):
            paths.add(pair(x, y))
    return paths


def connect_loops(verts, adjacent, comps, weights=None, relief=None):
    """Cut paths joining every boundary component to the first, one path per
    extra loop, each the shortest available at the time."""
    cuts = set()
    sources = set(comps[0])
    targets = {v: i for i, comp in enumerate(comps[1:], 1) for v in comp}
    while targets:
        path = cut_path(verts, adjacent, sources, targets, weights, relief)
        if not path:
            break  # disconnected, leave the rest to the engine
        for a, b in zip(path, path[1:]):
            cuts.add(pair(a, b))
        reached = comps[targets[path[0]]]
        sources.update(path)
        sources.update(reached)
        for v in reached:
            del targets[v]
    return cuts


def disk_cuts(verts, edges, label, weights=None, relief=None):
    """Seam paths that open every multi-loop region into a disk.

    A tube wall is an annulus straight out of the partition and no merge can
    fix that, so cut it here: a path joining two of its boundary loops opens
    the region without splitting it, one cut per extra loop. The engine throws
    a non-disk chart away and re-cuts the whole model, which is what this
    avoids. Genus is left to the engine, a handle needs a loop cut."""
    needs = {r: c for r, c in boundary_components(edges, label).items() if len(c) > 1}
    if not needs:
        return set()

    adjacent = {r: collections.defaultdict(set) for r in needs}
    for (v0, v1), owners in edges.items():
        for r in {label[o] for o in owners} & needs.keys():
            adjacent[r][v0].add(v1)
            adjacent[r][v1].add(v0)

    cuts = set()
    for region, comps in needs.items():
        cuts |= connect_loops(verts, adjacent[region], comps, weights, relief)
    return cuts


def uv_topology(group, faces, edges, seams):
    """Euler characteristic and uv boundary loops of one island, counted the
    way the engine reads the exported vt mesh: corners glue across interior
    non-seam edges and glued edge duplicates collapse, so a cut-open tube is
    the disk its unwrap is and a tip-welded slit stays interior. A disk is 1.
    Boundary loops come back as mesh vert sets, and loops touching at a
    glued corner count as one, a cut between them would be a point."""
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
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
            a, b = find((f, face[i])), find((f, face[(i + 1) % n]))
            edge_count[(a, b) if a < b else (b, a)] += 1

    classes = {find(node) for node in parent}
    ec = len(classes) - len(edge_count) + len(group)

    comp_parent = {}

    def comp_find(x):
        while comp_parent[x] != x:
            comp_parent[x] = comp_parent[comp_parent[x]]
            x = comp_parent[x]
        return x

    boundary = [key for key, count in edge_count.items() if count == 1]
    for a, b in boundary:
        comp_parent.setdefault(a, a)
        comp_parent.setdefault(b, b)
        ra, rb = comp_find(a), comp_find(b)
        if ra != rb:
            comp_parent[ra] = rb

    loops = collections.defaultdict(set)
    for a, b in boundary:
        # a corner class only ever holds one mesh vert, its node's second slot
        loops[comp_find(a)].update((a[1], b[1]))
    return ec, list(loops.values())


def signed_area(pts):
    total = 0.0
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        total += a[0] * b[1] - b[0] * a[1]
    return total / 2


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
    segments crossing: what makes the engine throw the island's layout away
    and re-cut it. Crossings between two different islands do not happen out
    of blender's packer, only inside one island."""
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
                # touching is a shared uv corner, not a shared mesh vertex:
                # the two sides of a seam edge share both verts yet can still
                # cross once unwrapped, and the engine tests vt space
                if {a, b} & {c, d}:
                    continue
                if crosses(a, b, c, d):
                    return True
    return False


def straighten_cut(
    verts, group, faces, edges, seams, bin_of, weights=None, relief=None
):
    """Straighten an island's bin cut, returning its seam edges.

    The bin line follows whatever edges cross it, so on diagonal edge flow
    it zigzags in deep teeth. Two passes fix that. Sweeps of sliding faces
    across the cut while that shortens it flatten single-face teeth and let
    the cut settle on short edges; only strictly shortening moves happen,
    and the sweeps are capped so the cut cannot creep along a tapering
    strip. Then every connected run of cut edges is swapped for the shortest
    interior path between its own two endpoints when that is strictly
    shorter, which removes the teeth sliding cannot: a two-face tooth only
    shortens once both faces move. Keeping the run's own endpoints keeps the
    cut network separating the same way, so unlike free endpoint picking
    this can never cut a tube wall lengthwise. Runs that loop or branch are
    kept as they are. Both passes measure a painted edge as longer, so the
    cut slides and reroutes out of restricted areas as well as off teeth."""
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

    Runs on the unwrap of the seams this module chose: a cut-open tube wall
    unrolls into a strip as long as the tube's ring, and the unwrap folds the
    longest of those. The 3D shape of such a region does not predict its
    unwrapped shape, so measure the unwrap itself: per island, the extent
    along the area-weighted principal axis of the face centroids, against the
    island's uv area. Faces bin along that axis and the edges between bins
    are the seams for a second unwrap, so every piece is a compact slice by
    construction, and straighten_cut then shortens each cut so the seam does
    not zigzag on diagonal edge flow. A ruined island that is not a strip
    still gets halved: smaller pieces unwrap cleanly where the whole did
    not, and it is lost to the engine as it stands. A non-disk island is
    opened instead of split, a path joining two of its boundary loops, since
    no amount of splitting fixes topology. A clean island is left whole
    unless it is a strip past the same length and aspect bounds: a strip
    spanning the atlas packs badly, and artists slice those too.

    groups restricts the scan to those islands, for a caller that knows the
    rest is unchanged since its last look."""
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
            # non-disk: a path joining two boundary loops opens the island
            # without splitting it, like disk_cuts does for regions. only
            # interior edges can become cuts, a boundary edge already is one
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
        bins = math.ceil(aspect / SPLIT_ASPECT) if strip else 2
        bin_of = {f: min(bins - 1, int((ts[f] - lo) / length * bins)) for f in group}
        extra |= straighten_cut(
            verts, group, faces, edges, seams, bin_of, weights, cut_relief()
        )
    return extra


def feature_labels(verts, faces, angle=CREASE_ANGLE, rims=True, forced=None):
    """Region labels from the merge passes: partition at auto width, the three
    merges, sweep rims. What survives is the feature structure the seams will
    trace, before the boundary cleanup passes move any edge."""
    weighted, areas, edges = build(verts, faces)
    find = partition(faces, weighted, edges, LOW_ANGLE, forced)
    min_width = detect_width(verts, faces, areas, edges, find, diagonal(verts))
    label, bounds = absorb(
        verts, faces, weighted, areas, edges, find, min_width, forced
    )
    label = merge_smooth(edges, label, bounds, min_width, angle, forced)
    label = merge_flat(weighted, areas, edges, label, angle, forced)
    label = close_rings(verts, weighted, areas, edges, label, angle, forced)
    if rims:
        label = split_sweeps(weighted, areas, edges, label)
    return weighted, areas, edges, label


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


# auto mode guards, tuned on the bench model sets (table in
# docs/agents/bench-results.md). A part is hard surface only when every one
# holds: one region covering the part means no structure (a smooth blob),
ORGANIC_SHARE = 0.9
# regions averaging under this many faces mean the partition found noise,
# not panels (chainmail reads 1 face per region),
FRAGMENT_FACES = 8
# turn between LOW_ANGLE and this is spread curvature. A bevel is spread
# turn beside a feature boundary and a sculpt is spread turn everywhere, so
# the share is read away from the region boundaries, and a part with too
# much of its interior edge length spread is sculpted,
SPREAD_ANGLE = 25
SPREAD_SHARE = 0.21
# and the region boundaries must mostly be deliberate: on an edge that is
# itself creased, or a rim split_sweeps placed. A duck's wing outline is
# neither, a screwdriver's dull boundaries are all rims
BOUNDARY_ANGLE = 20
BOUNDARY_CREASED = 0.6


def is_hard_surface(verts, faces):
    """Whether a loose part's features are worth preseeding.

    Reads the merged region structure at the CREASE_ANGLE floor, not the
    feature angle knob: the question is whether structure exists at all, and
    the knob only tunes seam density once it does. split_sweeps runs so a
    smooth cylinder whose rims no angle test sees still reads hard, and its
    rims count as deliberate boundaries. Misreading organic costs a slow
    from-scratch unwrap, misreading hard costs seams on sculpt ridges, so
    ties fall organic."""
    weighted, areas, edges, presweep = feature_labels(verts, faces, rims=False)
    label = split_sweeps(weighted, areas, edges, presweep)
    total = sum(areas)
    if total <= 0:
        return False
    region = collections.defaultdict(float)
    for i, r in label.items():
        region[r] += areas[i]
    if max(region.values()) / total >= ORGANIC_SHARE:
        return False
    if len(faces) / len(region) < FRAGMENT_FACES:
        return False

    near = set()
    for key, owners in edges.items():
        if len(owners) == 2 and label[owners[0]] != label[owners[1]]:
            near.update(owners)
    # two rings, so a dissolved bevel band beside a seam stays out of the
    # interior read
    for _ in range(2):
        grown = set(near)
        for owners in edges.values():
            if len(owners) == 2 and not near.isdisjoint(owners):
                grown.update(owners)
        near = grown

    spread = interior = boundary = boundary_creased = 0.0
    for (a, b), owners in edges.items():
        if len(owners) != 2:
            continue
        length = norm([verts[a][i] - verts[b][i] for i in range(3)])
        turn = turn_angle(weighted, owners)
        if label[owners[0]] != label[owners[1]]:
            boundary += length
            if turn >= BOUNDARY_ANGLE or presweep[owners[0]] == presweep[owners[1]]:
                boundary_creased += length
        elif owners[0] not in near and owners[1] not in near:
            interior += length
            if LOW_ANGLE < turn < SPREAD_ANGLE:
                spread += length
    if not boundary:
        return False
    return (
        (not interior or spread / interior < SPREAD_SHARE)
        and boundary_creased / boundary >= BOUNDARY_CREASED
    )


def seam_edges(verts, faces, angle=CREASE_ANGLE, rims=True, weights=None, forced=None):
    """The full pipeline at auto width: partition, the three merges, seams.

    angle is what counts as a feature: boundaries turning less than it merge
    away, so lower keeps more shallow-feature seams, artist style, at the
    cost of shattering coarse curved walls whose panels turn more than it.
    rims off skips split_sweeps, so a smooth cylinder keeps its end caps.
    weights are painted restrictions, which the cuts avoid; region boundaries
    are where the shape says they are and paint does not move them.
    forced are hand-marked edges to seam whatever the shape says. They cut
    from the partition on, so the merges see them and route around them, and
    they are seams in the end even where they only slit a region."""
    weighted, areas, edges, label = feature_labels(verts, faces, angle, rims, forced)
    label = flatten_teeth(weighted, faces, edges, label, angle, forced)
    relief = crease_relief(verts, faces, weighted, edges)
    label = reroute_boundaries(verts, faces, areas, edges, label, relief, forced)
    seams = boundary_edges(edges, label) | disk_cuts(
        verts, edges, label, weights, relief
    )
    if forced:
        seams |= forced
    if (
        not seams
        and angle > CREASE_ANGLE
        and all(len(owners) != 1 for owners in edges.values())
    ):
        # a closed mesh that merged seamless cannot flatten at all: every
        # feature sat under the angle (a hex head smears to just under 60),
        # so retry at the floor instead of handing the engine nothing
        return seam_edges(verts, faces, CREASE_ANGLE, rims, weights, forced)
    return seams
