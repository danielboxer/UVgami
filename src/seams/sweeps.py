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

from .mesh import cross, find, norm


# a region splits at its rims when the 30-60 degree middle band holds under
# BAND of its area and caps hold at least CAP_MIN: a bent tube fills the
# band, a swept one does not (a measured elbow reads 0.31, a screwdriver
# handle 0.07)
SWEEP_BAND = 0.1
SWEEP_CAP_MIN = 0.02
# only regions holding this share of the model get rim cuts: cutting every
# screw and pin explodes the chart count
SWEEP_MIN_SHARE = 0.01
# the wall must actually turn around the axis: its normals' resultant length
# over its mass, 1 on a plate, 0 on a full wall, 0.7 asks for a half turn,
# so a plate with tilted flanges keeps whole
WALL_ROUND = 0.7
# wall/cap boundary and the band edges, as squared sines of the tilt
CAP_SPLIT = 0.5  # 45 degrees
BAND_LO = 0.25  # 30 degrees
BAND_HI = 0.75  # 60 degrees
# alternating fit rounds for the sweep axis
SWEEP_FIT_ROUNDS = 10


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


def split_sweeps(weighted, areas, edges, label):
    """Split swept regions into wall and cap parts, seaming their rims.

    See SWEEP_BAND: this stops a cylinder-like region from keeping its end
    caps and unwrapping as a polar map. A region whose normals sort cleanly
    into wall and cap is relabeled by connected component of that split, a
    component too small for a real cap merges back, and regions with real
    middle-band mass, bent tubes and organic blobs, are left alone.
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
        # a rim seam is not worth a speck: merge any component smaller than
        # a real cap into whichever neighbour it touches most
        while True:
            speck = min(comp_area, key=comp_area.get)
            if comp_area[speck] >= SWEEP_CAP_MIN * total or len(comp_area) < 2:
                break
            touch = collections.Counter()
            for (a, b), c in contact.items():
                ra, rb = find(parent, a), find(parent, b)
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
            relabel[i] = find(parent, i)

    if not relabel:
        return label
    return {i: relabel.get(i, r) for i, r in label.items()}
