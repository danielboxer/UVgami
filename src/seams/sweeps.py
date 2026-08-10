"""Rim cuts for swept shapes.

On a smooth model a filleted rim reads smooth, so a cylinder wall
merges over its end caps and the unwrap flattens the region as a
polar map: near isometric, but the texture direction winds around the
cap instead of following the axis. No distortion measure catches
that, so the structure is read off the normals: against the right
axis a swept region's faces are either wall or cap with little in
between, while a bent tube fills the middle band."""

import collections
import math

from .islands import SPLIT_ASPECT
from .mesh import build, cross, face_keys, find, norm
from .regions import CREASE_ANGLE, partition


# a region splits at its rims when the 30-60 degree middle band holds under
# BAND of its area and caps hold at least CAP_MIN: a bent tube fills the
# band, a swept one does not (a measured elbow reads 0.31, a screwdriver
# handle 0.07)
SWEEP_BAND = 0.1
SWEEP_CAP_MIN = 0.02
# only regions holding this share of the model get rim cuts: cutting every
# screw and pin explodes the chart count
SWEEP_MIN_SHARE = 0.01
# and enough faces that the normal fit means something
SWEEP_MIN_FACES = 8
# the wall must actually turn around the axis: its normals' resultant length
# over its mass, 1 on a plate, 0 on a full wall, 0.7 asks for a half turn,
# so a plate with tilted flanges keeps whole
WALL_ROUND = 0.7
# two touching walls sweeping the same axis need no rim between them: a
# grooved ring's bands unroll together
SHARED_AXIS_COS = 0.95
# how finely a wall with a handle through it is trimmed back along its axis
# while hunting the cut that leaves a flattenable surface
GENUS_TRIM_LEVELS = 24
# wall/cap boundary and the band edges, as squared sines of the tilt
CAP_SPLIT = 0.5  # 45 degrees
BAND_LO = 0.25  # 30 degrees
BAND_HI = 0.75  # 60 degrees
# alternating fit rounds for the sweep axis
SWEEP_FIT_ROUNDS = 10
# faces a run seed probes before the patch around it is shed: a trumpet
# flare never passes anywhere, so probing must stop and move on instead of
# tasting the whole cluster from every bad seed. counted in faces, not
# growth rings, because a spiral-strip triangulation grows one face a ring
RUN_SEED_FACES = 512
# position-based straightness: the run's centroid variance off its main
# direction over the variance along it. 0.2 admits a straight tube about
# four diameters long, a sphere zone reads near 1 and can never pass
RUN_SLENDER = 0.2
# growth rings behind a run cut searched for a concave boundary to snap
# to: the groove between hose ribs, where an artist hides the cut
VALLEY_SNAP_LAYERS = 6


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


def eigenvalues3(xx, yy, zz, xy, xz, yz):
    """Eigenvalues of a symmetric 3x3 matrix, largest first."""
    p1 = xy * xy + xz * xz + yz * yz
    if p1 == 0:
        return tuple(sorted((xx, yy, zz), reverse=True))
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
    high = q + 2 * p * math.cos(phi)
    low = q + 2 * p * math.cos(phi + 2 * math.pi / 3)
    return high, 3 * q - high - low, low


def sweep_axis(normals):
    """Axis against which the normals, a list of (area, unit normal), split
    into wall and cap. Alternates: classify each normal against the axis,
    then refit with cap normals repelled, pulling the axis toward
    perpendicularity with the walls only.
    """

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


def class_components(group, is_cap, edges, areas):
    """Connected components of same-class faces over the group's edges.
    Returns the parent map, the cross-class contact counts and per-component
    areas, membership read with find."""
    in_group = set(group)
    parent = {i: i for i in group}
    contact = collections.Counter()
    for owners in edges.values():
        if len(owners) != 2:
            continue
        a, b = owners
        if a not in in_group or b not in in_group:
            continue
        if is_cap[a] == is_cap[b]:
            ra, rb = find(parent, a), find(parent, b)
            if ra != rb:
                parent[ra] = rb
        else:
            contact[(a, b)] += 1
    comp_area = collections.defaultdict(float)
    for i in group:
        comp_area[find(parent, i)] += areas[i]
    return parent, contact, comp_area


def merge_specks(parent, contact, comp_area, floor):
    """Merge every component smaller than floor into whichever neighbour it
    touches most: a rim seam is not worth a speck."""
    while True:
        speck = min(comp_area, key=comp_area.get)
        if comp_area[speck] >= floor or len(comp_area) < 2:
            break
        touch = collections.Counter()
        for (a, b), count in contact.items():
            ra, rb = find(parent, a), find(parent, b)
            if ra != rb and speck in (ra, rb):
                touch[rb if ra == speck else ra] += count
        if not touch:
            break
        into = touch.most_common(1)[0][0]
        parent[speck] = into
        comp_area[into] += comp_area.pop(speck)


def normal_fit(entries):
    """The normal-based run judge: a run passes the same wall test a whole
    cluster takes, empty middle band and a wrapping wall, and gets the
    fitted sweep axis. entries is {face: (area, unit normal)}."""

    def fit(run):
        picked = [entries[i] for i in run if i in entries]
        if len(picked) < SWEEP_MIN_FACES:
            return None
        m = [0.0] * 6
        for w, n in picked:
            m[0] += w * n[0] * n[0]
            m[1] += w * n[1] * n[1]
            m[2] += w * n[2] * n[2]
            m[3] += w * n[0] * n[1]
            m[4] += w * n[0] * n[2]
            m[5] += w * n[1] * n[2]
        axis = min_eigenvector(*m)
        total = band = wall_mass = 0.0
        wall_sum = [0.0, 0.0, 0.0]
        for w, n in picked:
            axial = (n[0] * axis[0] + n[1] * axis[1] + n[2] * axis[2]) ** 2
            total += w
            if BAND_LO < axial < BAND_HI:
                band += w
            if axial < CAP_SPLIT:
                wall_mass += w
                for k in range(3):
                    wall_sum[k] += w * n[k]
        # an end run carries the tube tip, so caps split out instead of
        # counting against the band, but a run may not be mostly cap
        if band > SWEEP_BAND * total or total - wall_mass > total / 2:
            return None
        if not wall_mass or norm(wall_sum) / wall_mass > WALL_ROUND:
            return None
        return axis

    return fit


def slender_fit(verts, faces, areas, entries):
    """A run judge from face positions: a straight tube's centroids hug a
    line whatever its profile, so a corrugated hose reads straight where
    the normal test sees a bend at every rib. A run passes while its
    centroid mass off the main direction stays under RUN_SLENDER of the
    mass along it, and its normals must cancel out, or a thin crescent of
    a sphere would read as a tube."""
    centers = {}

    def center(i):
        if i not in centers:
            face = faces[i]
            centers[i] = [sum(verts[v][k] for v in face) / len(face) for k in range(3)]
        return centers[i]

    def fit(run):
        if len(run) < SWEEP_MIN_FACES:
            return None
        total = 0.0
        mean = [0.0, 0.0, 0.0]
        resultant = [0.0, 0.0, 0.0]
        for i in run:
            w = areas[i]
            total += w
            c = center(i)
            for k in range(3):
                mean[k] += w * c[k]
            if i in entries:
                n = entries[i][1]
                for k in range(3):
                    resultant[k] += w * n[k]
        if not total:
            return None
        if norm(resultant) / total > WALL_ROUND:
            return None
        mean = [x / total for x in mean]
        m = [0.0] * 6
        for i in run:
            w = areas[i]
            c = center(i)
            d0, d1, d2 = (c[k] - mean[k] for k in range(3))
            m[0] += w * d0 * d0
            m[1] += w * d1 * d1
            m[2] += w * d2 * d2
            m[3] += w * d0 * d1
            m[4] += w * d0 * d2
            m[5] += w * d1 * d2
        high, middle, low = eigenvalues3(*m)
        if high <= 0 or (middle + low) / high > RUN_SLENDER:
            return None
        return min_eigenvector(-m[0], -m[1], -m[2], -m[3], -m[4], -m[5])

    return fit


def valley_snap(verts, faces, edges, entries):
    """A run-cut snapper: score the growth boundaries just behind the cut
    by signed concave turn and move the cut to the deepest valley. On a
    corrugated hose that is the groove between ribs, where an artist
    hides the cut, instead of across a rib in the open."""

    def centroid(i):
        face = faces[i]
        return [sum(verts[v][k] for v in face) / len(face) for k in range(3)]

    def score(layer, next_layer):
        ahead = set(next_layer)
        total = 0.0
        for f in layer:
            if f not in entries:
                continue
            normal = entries[f][1]
            base = centroid(f)
            for key in face_keys(faces[f]):
                owners = edges[key]
                if len(owners) != 2:
                    continue
                a, b = owners
                other = b if a == f else a
                if other not in ahead or other not in entries:
                    continue
                v0, v1 = key
                length = norm([verts[v1][k] - verts[v0][k] for k in range(3)])
                dot = sum(x * y for x, y in zip(normal, entries[other][1]))
                turn = math.acos(max(-1.0, min(1.0, dot)))
                across = centroid(other)
                toward = sum(normal[k] * (across[k] - base[k]) for k in range(3))
                total += length * turn * (1.0 if toward > 0 else -1.0)
        return total

    def snap(flat, ends, cut):
        best, best_score = cut, 0.0
        for c in range(max(2, cut - VALLEY_SNAP_LAYERS), cut + 1):
            valley = score(flat[ends[c - 2] : ends[c - 1]], flat[ends[c - 1] : ends[c]])
            if valley > best_score:
                best, best_score = c, valley
        return best

    return snap


def split_sweeps(verts, faces, weighted, areas, edges, label):
    """Split swept regions into wall and cap parts, seaming their rims.

    See SWEEP_BAND: this stops a cylinder-like region from keeping its end
    caps and unwrapping as a polar map. A region whose normals sort cleanly
    into wall and cap is relabeled by connected component of that split, a
    component too small for a real cap merges back. A region with real
    middle-band mass is a bent tube: it relabels into straight runs, one
    region each, so a coiled cable unrolls as short straight strips instead
    of one curled snake. Organic blobs pass neither test and are left
    alone. This is where a coarse bent tube gets its runs: its cross-section
    turns past CREASE_ANGLE, so sweep_rims never sees it whole, but the
    merges rebuild it into one region by the time this pass runs.
    """
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
        if len(normals) < SWEEP_MIN_FACES or total < SWEEP_MIN_SHARE * model_area:
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
            # either judge accepts a run: normals catch a fat smooth tube,
            # positions catch a corrugated hose that reads bent at every
            # rib to the normal test. no cap split, rib shoulders tilt
            # like caps but are profile
            entries = {i: (w, n) for i, w, n in normals}
            by_normals = normal_fit(entries)
            by_positions = slender_fit(verts, faces, areas, entries)

            def either(run):
                return by_normals(run) or by_positions(run)

            snap = valley_snap(verts, faces, edges, entries)
            for run, _ in straight_runs(group, entries, edges, either, snap):
                for i in run:
                    relabel[i] = run[0]
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
        parent, contact, comp_area = class_components(group, is_cap, edges, areas)
        merge_specks(parent, contact, comp_area, SWEEP_CAP_MIN * total)
        if len(comp_area) < 2:
            continue
        for i in group:
            relabel[i] = find(parent, i)

    if not relabel:
        return label
    return {i: relabel.get(i, r) for i, r in label.items()}


def component_faces(subset, edges):
    """Connected components of these faces across shared edges."""
    in_set = set(subset)
    parent = {i: i for i in subset}
    for owners in edges.values():
        if len(owners) == 2 and owners[0] in in_set and owners[1] in in_set:
            ra, rb = find(parent, owners[0]), find(parent, owners[1])
            if ra != rb:
                parent[ra] = rb
    groups = collections.defaultdict(list)
    for i in subset:
        groups[find(parent, i)].append(i)
    return list(groups.values())


def surface_genus(subset, faces, edges):
    """Genus of these faces as one open surface. 0 flattens, extra boundary
    loops just take connecting cuts, but above 0 a handle runs through the
    surface and no cut disk_cuts can place opens it."""
    in_set = set(subset)
    used = set()
    keys = set()
    for i in subset:
        used.update(faces[i])
        keys.update(face_keys(faces[i]))
    euler = len(used) - len(keys) + len(subset)
    parent = {}
    for key in keys:
        if sum(o in in_set for o in edges[key]) != 1:
            continue
        v0, v1 = key
        parent.setdefault(v0, v0)
        parent.setdefault(v1, v1)
        ra, rb = find(parent, v0), find(parent, v1)
        if ra != rb:
            parent[ra] = rb
    loops = len({find(parent, v) for v in parent})
    return (2 - loops - euler) // 2


def trim_genus(wall, verts, faces, edges, areas, axis):
    """The wall faces minus whatever it takes to make every piece flattenable.

    A handle through a wall (the hanging loop on a wrench handle) makes a
    surface no flatten can open, and shipping it ruins the whole strip: the
    engine rejects the island and improvises over all of it. So peel the
    wall back along its axis until no handle remains, and leave the peeled
    end to the engine like any unclaimed area. A component that cannot be
    saved by giving up half its area is dropped whole."""
    kept = set()
    for comp in component_faces(wall, edges):
        if surface_genus(comp, faces, edges) <= 0:
            kept.update(comp)
            continue
        position = {}
        for i in comp:
            face = faces[i]
            centroid = [sum(verts[v][k] for v in face) / len(face) for k in range(3)]
            position[i] = sum(centroid[k] * axis[k] for k in range(3))
        ordered = sorted(comp, key=position.get)
        best = None
        for direction in (ordered, ordered[::-1]):
            for level in range(1, GENUS_TRIM_LEVELS):
                cut = len(comp) * level // GENUS_TRIM_LEVELS
                rest = direction[cut:]
                if not rest:
                    break
                pieces = component_faces(rest, edges)
                if all(surface_genus(p, faces, edges) <= 0 for p in pieces):
                    loss = sum(areas[i] for i in direction[:cut])
                    if best is None or loss < best[0]:
                        best = (loss, rest)
                    break
        if best is not None and best[0] <= sum(areas[i] for i in comp) / 2:
            kept.update(best[1])
    return kept


def wall_hoops(wall, axis, verts, faces, edges):
    """Whether the wall is a hoop: it would unroll into a strip past the
    slicer bound, and rims would shred it into thin loops. Also the guard
    that keeps a sphere from reading as a stack of thin latitude runs."""
    boundary = 0.0
    for i in wall:
        for key in face_keys(faces[i]):
            owners = edges[key]
            if len(owners) == 2 and (owners[0] in wall) != (owners[1] in wall):
                v0, v1 = key
                boundary += norm([verts[v1][k] - verts[v0][k] for k in range(3)])
    along = [
        sum(verts[v][k] * axis[k] for k in range(3)) for i in wall for v in faces[i]
    ]
    extent = max(along) - min(along) if along else 0.0
    return boundary > 0 and (extent <= 0 or boundary / 2 > SPLIT_ASPECT * extent)


def claim_wall(group, axial, axis, total, verts, faces, edges, areas):
    """The group's wall faces after the cap split, the genus trim and the
    hoop check, empty when nothing worth claiming is left."""
    is_cap = {i: axial.get(i, 0.0) >= CAP_SPLIT for i in group}
    parent, contact, comp_area = class_components(group, is_cap, edges, areas)
    cls = {find(parent, i): is_cap[i] for i in group}
    merge_specks(parent, contact, comp_area, SWEEP_CAP_MIN * total)
    wall = {i for i in group if not cls[find(parent, i)]}
    wall = trim_genus(wall, verts, faces, edges, areas, axis)
    if not wall:
        return set()
    if wall_hoops(wall, axis, verts, faces, edges):
        return set()
    return wall


def straight_runs(group, entries, edges, fit_of=None, snap=None):
    """Contiguous pieces of a bent swept cluster, each straight enough to
    pass the wall test against its own axis, with that axis.

    A coiled cable fails the whole-cluster test even though every short
    piece of it is a clean tube. So pieces grow outward from an extremity
    and close where the accumulated turn breaks the test, cutting the
    tube into runs. Probe sizes double then bisect to the break, so a
    cluster with nothing straight in it costs a few failed fits, not one
    per growth step. fit_of overrides the normal-based test with another
    judge of a piece: it gets the faces, returns the axis or None. snap
    gets (flat, ends, cut) and may move the cut a few growth rings back
    to a better boundary, a valley say, returning the new cut."""
    in_group = set(group)
    adjacency = collections.defaultdict(list)
    for owners in edges.values():
        if len(owners) == 2:
            a, b = owners
            if a in in_group and b in in_group:
                adjacency[a].append(b)
                adjacency[b].append(a)

    def spread(seed, allowed):
        """Reachable faces grouped by steps taken from the seed."""
        layers = [[seed]]
        seen = {seed}
        while True:
            grown = []
            for face in layers[-1]:
                for other in adjacency[face]:
                    if other in allowed and other not in seen:
                        seen.add(other)
                        grown.append(other)
            if not grown:
                return layers
            layers.append(grown)

    if fit_of is None:
        fit_of = normal_fit(entries)

    remaining = set(group)
    runs = []
    seed = None
    while remaining:
        if seed is None or seed not in remaining:
            start = next(iter(remaining))
            seed = spread(start, remaining)[-1][0]
        layers = spread(seed, remaining)
        flat, ends = [], []
        for layer in layers:
            flat.extend(layer)
            ends.append(len(flat))
        count = len(layers)

        span = count
        for depth, end in enumerate(ends, 1):
            if end >= RUN_SEED_FACES:
                span = depth
                break
        low, axis = 0, None
        probe = 1
        while probe < span:
            fit = fit_of(flat[: ends[probe - 1]])
            if fit:
                low, axis = probe, fit
                break
            probe *= 2
        if axis is None:
            fit = fit_of(flat[: ends[span - 1]])
            if fit:
                low, axis = span, fit
            else:
                # nothing straight near this seed: shed just the probed
                # patch and walk on, engine territory
                remaining.difference_update(flat[: ends[span - 1]])
                seed = layers[span][0] if span < count else None
                continue
        high = None
        while high is None and low < count:
            probe = min(low * 2, count)
            fit = fit_of(flat[: ends[probe - 1]])
            if fit:
                low, axis = probe, fit
            else:
                high = probe
        while high is not None and high - low > 1:
            mid = (low + high) // 2
            fit = fit_of(flat[: ends[mid - 1]])
            if fit:
                low, axis = mid, fit
            else:
                high = mid
        if snap and 1 < low < count:
            low = snap(flat, ends, low)
        run = flat[: ends[low - 1]]
        runs.append((run, axis))
        remaining.difference_update(run)
        seed = layers[low][0] if low < count else None
    return runs


def sweep_rims(verts, faces):
    """Rim seams for swept shapes, read before any merge pass.

    absorb never checks how far a boundary turns, so a coarse cylinder's
    narrow wall columns dissolve into its caps and no later pass can undo
    that. So find the wall first: faces cluster across smooth edges only,
    and a cluster whose normals curl around an axis with an empty middle
    band is a swept wall. Faces tilted into the axis split off as caps, so
    a closed tube end becomes its own island and the rim follows the tilt
    contour instead of the ragged cluster border. A cluster too bent for
    one axis splits into straight runs, each rimmed apart from the next.
    Returns the rim edges as vertex pairs, to be forced so no merge
    crosses them, plus the wall faces themselves."""
    weighted, areas, edges = build(verts, faces)
    root = partition(faces, weighted, edges, CREASE_ANGLE)
    groups = collections.defaultdict(list)
    for i in range(len(faces)):
        groups[root(i)].append(i)
    model_area = sum(areas)

    walls = set()
    wall_axis = {}
    wall_run = {}
    wall_cluster = {}
    run_count = 0

    def claim(run, axis, cluster, axial, total):
        nonlocal run_count
        wall = claim_wall(run, axial, axis, total, verts, faces, edges, areas)
        if not wall:
            return
        run_count += 1
        walls.update(wall)
        for i in wall:
            wall_axis[i] = axis
            wall_run[i] = run_count
            wall_cluster[i] = cluster

    for group in groups.values():
        normals = []
        for i in group:
            n = weighted[i]
            length = norm(n)
            if length:
                normals.append((i, areas[i], tuple(x / length for x in n)))
        total = sum(w for _, w, _ in normals)
        if len(normals) < SWEEP_MIN_FACES or total < SWEEP_MIN_SHARE * model_area:
            continue
        m = [0.0] * 6
        for _, w, n in normals:
            m[0] += w * n[0] * n[0]
            m[1] += w * n[1] * n[1]
            m[2] += w * n[2] * n[2]
            m[3] += w * n[0] * n[1]
            m[4] += w * n[0] * n[2]
            m[5] += w * n[1] * n[2]
        axis = min_eigenvector(*m)
        axial = {}
        band = off_wall = 0.0
        resultant = [0.0, 0.0, 0.0]
        for i, w, n in normals:
            axial[i] = (n[0] * axis[0] + n[1] * axis[1] + n[2] * axis[2]) ** 2
            if axial[i] > BAND_LO:
                off_wall += w
            if BAND_LO < axial[i] < BAND_HI:
                band += w
            for k in range(3):
                resultant[k] += w * n[k]
        # tilted mass means a bent tube, a cone too steep, or a sock with
        # a large cap smoothly attached. real middle-band mass is the bent
        # tube: its straight runs still get claimed. the rest, and a flat
        # or gently bowed panel, stay with the merge passes
        if off_wall > SWEEP_BAND * total or norm(resultant) / total > WALL_ROUND:
            if band > SWEEP_BAND * total:
                entries = {i: (w, n) for i, w, n in normals}
                for run, run_axis in straight_runs(group, entries, edges):
                    run_axial = {}
                    run_total = 0.0
                    for i in run:
                        if i in entries:
                            w, n = entries[i]
                            run_total += w
                            run_axial[i] = (
                                n[0] * run_axis[0]
                                + n[1] * run_axis[1]
                                + n[2] * run_axis[2]
                            ) ** 2
                    claim(run, run_axis, group[0], run_axial, run_total)
            continue
        claim(group, axis, group[0], axial, total)

    if not walls:
        return set(), walls
    rims = set()
    for key, owners in edges.items():
        if len(owners) != 2:
            continue
        a, b = owners
        in_a, in_b = a in walls, b in walls
        if in_a != in_b:
            rims.add(key)
        elif in_a and wall_run[a] != wall_run[b]:
            # runs of one bent tube always part: that cut is what made
            # them straight
            if wall_cluster[a] == wall_cluster[b]:
                rims.add(key)
            else:
                dot = abs(sum(x * y for x, y in zip(wall_axis[a], wall_axis[b])))
                if dot < SHARED_AXIS_COS:
                    rims.add(key)
    return rims, walls
