"""Boundary targets that straighten near-rectangular islands.

The solve itself stays in the addon: the caller pins these targets and runs
a pinned unwrap over the island interior. Everything walks uv points, not
mesh vertices: an island bordering its own cut carries two uvs on each cut
vertex and the slit is part of the boundary."""

import math

from .mesh import signed_area

# uv area over fitted rectangle area an island needs to rectify. a circle
# fills 0.785 of its square, so blobs stay under the gate
RECTANGLE_SHARE = 0.8
# boundary length squared over circle length squared at equal area. strips
# measure 3 and up, blobs 1 to 2.5, so this admits the wavy and curled
# strips the share gate misses
STRIP_ELONGATION = 3.0


def island_area(group, uvs):
    return abs(sum(signed_area(uvs[fi]) for fi in group))


def _boundary_loop(group, uvs):
    """The island's single boundary loop as uv points in walk order. None
    when the boundary splits into several loops (holes) or pinches through
    a point."""
    counts = {}
    for fi in group:
        face = uvs[fi]
        for i in range(len(face)):
            a, b = face[i], face[(i + 1) % len(face)]
            key = (a, b) if a < b else (b, a)
            counts[key] = counts.get(key, 0) + 1
    boundary = [edge for edge, count in counts.items() if count == 1]
    if len(boundary) < 4:
        return None

    neighbors = {}
    for a, b in boundary:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)
    if any(len(around) != 2 for around in neighbors.values()):
        return None

    start = boundary[0][0]
    loop = [start]
    previous, current = None, start
    while True:
        a, b = neighbors[current]
        following = b if a == previous else a
        if following == start:
            break
        loop.append(following)
        previous, current = current, following
    if len(loop) != len(neighbors):
        return None
    return loop


def _arc_lengths(points, start, stop):
    """Cumulative distance along points from index start to index stop,
    cyclic, one entry per point of the segment including both ends."""
    lengths = [0.0]
    i = start
    while i != stop:
        after = (i + 1) % len(points)
        lengths.append(lengths[-1] + math.dist(points[i], points[after]))
        i = after
    return lengths


def _rectangle_targets(loop, area):
    points = loop
    if signed_area(points) < 0:
        points = points[::-1]

    perimeter = sum(
        math.dist(points[i], points[(i + 1) % len(points)]) for i in range(len(points))
    )
    elongation = perimeter**2 / (4 * math.pi * area)

    mean_x = sum(p[0] for p in points) / len(points)
    mean_y = sum(p[1] for p in points) / len(points)
    centered = [(x - mean_x, y - mean_y) for x, y in points]
    xx = sum(x * x for x, y in centered)
    xy = sum(x * y for x, y in centered)
    yy = sum(y * y for x, y in centered)
    angle = 0.5 * math.atan2(2 * xy, xx - yy)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rotated = [(x * cos_a + y * sin_a, y * cos_a - x * sin_a) for x, y in centered]

    xs = [p[0] for p in rotated]
    ys = [p[1] for p in rotated]
    box_width = max(xs) - min(xs)
    box_height = max(ys) - min(ys)
    if box_width <= 0 or box_height <= 0:
        return None
    share = area / (box_width * box_height)
    if share < RECTANGLE_SHARE and elongation < STRIP_ELONGATION:
        return None

    box_corners = [
        (min(xs), min(ys)),
        (max(xs), min(ys)),
        (max(xs), max(ys)),
        (min(xs), max(ys)),
    ]
    picks = []
    for corner_x, corner_y in box_corners:
        picks.append(
            min(
                range(len(rotated)),
                key=lambda i: (
                    ((rotated[i][0] - corner_x) / box_width) ** 2
                    + ((rotated[i][1] - corner_y) / box_height) ** 2
                ),
            )
        )
    if len(set(picks)) != 4:
        return None
    offsets = [(k - picks[0]) % len(points) for k in picks]
    if not offsets[1] < offsets[2] < offsets[3]:
        return None

    sides = [_arc_lengths(rotated, picks[s], picks[(s + 1) % 4]) for s in range(4)]
    if any(side[-1] == 0 for side in sides):
        return None
    # side lengths come from the boundary itself, not the box: a strip that
    # still curls unrolls to its real length, which the box undershoots
    width = (sides[0][-1] + sides[2][-1]) / 2
    height = (sides[1][-1] + sides[3][-1]) / 2
    rectangle = [
        (-width / 2, -height / 2),
        (width / 2, -height / 2),
        (width / 2, height / 2),
        (-width / 2, height / 2),
    ]

    targets = {}
    for s in range(4):
        ax, ay = rectangle[s]
        bx, by = rectangle[(s + 1) % 4]
        lengths = sides[s]
        for step, distance in enumerate(lengths[:-1]):
            t = distance / lengths[-1]
            x = ax + (bx - ax) * t
            y = ay + (by - ay) * t
            targets[points[(picks[s] + step) % len(points)]] = (
                x * cos_a - y * sin_a + mean_x,
                x * sin_a + y * cos_a + mean_y,
            )
    return targets


def rectify_targets(uvs, groups):
    """Per qualifying island, the faces and each boundary uv mapped onto the
    island's fitted rectangle: corners at the boundary points nearest the
    bounding box corners, everything between spread by arc length."""
    plans = []
    for group in groups:
        loop = _boundary_loop(group, uvs)
        if loop is None:
            continue
        area = island_area(group, uvs)
        if area <= 0:
            continue
        targets = _rectangle_targets(loop, area)
        if targets is not None:
            plans.append((group, targets))
    return plans
