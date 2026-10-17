import heapq
import math

from .mesh import LOW_ANGLE, norm, pair, turn_angle

# bounded, an infinite cost drops a cut instead of moving it
RESTRICT_COST = 9.0
# a crease edge counts shorter, concave gets the deeper discount
CONCAVE_RELIEF = 0.5
CONVEX_RELIEF = 0.3
# relief slides from full length at LOW_ANGLE to its floor here
RELIEF_FULL_ANGLE = 45
# relief below this counts an edge as creased
CREASED_RELIEF = 0.9
# a step turning between two dull edges costs this fraction of its length extra
TURN_COST = 1.0
# an exposed vertex repels a cut a tenth as hard as a painted one, so it only breaks ties
VISIBILITY_SHARE = 0.1


# the sign comes from the neighbour's centroid against the face plane
def crease_relief(verts, faces, weighted, edges):
    centroids = []
    for face in faces:
        x = y = z = 0.0
        for v in face:
            px, py, pz = verts[v]
            x += px
            y += py
            z += pz
        centroids.append([x / len(face), y / len(face), z / len(face)])
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


# painted weights win where both apply, exposure is 1 on open surface and 0 in a crevice
def restriction_weights(painted, exposure):
    weights = {v: VISIBILITY_SHARE * e for v, e in (exposure or {}).items() if e > 0}
    for v, w in (painted or {}).items():
        weights[v] = max(weights.get(v, 0.0), w)
    return weights or None


# longer where a painted restriction repels cuts, shorter along a crease
def edge_cost(verts, weights, a, b, relief=None):
    length = norm([verts[a][i] - verts[b][i] for i in range(3)])
    if relief:
        length *= relief.get(pair(a, b), 1.0)
    if not weights:
        return length
    paint = (weights.get(a, 0.0) + weights.get(b, 0.0)) / 2
    return length * (1 + RESTRICT_COST * paint)


# zero along a crease or a straight continuation, up to TURN_COST on a reversal
def turn_cost(verts, u, v, w, relief):
    if (
        relief.get(pair(u, v), 1.0) < CREASED_RELIEF
        and relief.get(pair(v, w), 1.0) < CREASED_RELIEF
    ):
        return 0.0
    a = [verts[v][i] - verts[u][i] for i in range(3)]
    b = [verts[w][i] - verts[v][i] for i in range(3)]
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    cos = sum(a[i] * b[i] for i in range(3)) / (na * nb)
    return TURN_COST * (1.0 - cos) / 2.0


# turn penalties included, so a comparison matches what cut_path searches for
def path_cost(verts, seq, weights=None, relief=None):
    total = 0.0
    for i in range(1, len(seq)):
        step = edge_cost(verts, weights, seq[i - 1], seq[i], relief)
        if relief is not None and i >= 2:
            step *= 1.0 + turn_cost(verts, seq[i - 2], seq[i - 1], seq[i], relief)
        total += step
    return total


# with relief the state carries the incoming direction
def cut_path(verts, adjacent, sources, targets, weights=None, relief=None):
    if relief is None:
        # without relief an edge costs at least its length
        goal = verts[next(iter(targets))] if len(targets) == 1 else None

        def remaining(v):
            if goal is None:
                return 0.0
            return norm([verts[v][i] - goal[i] for i in range(3)])

        dist = dict.fromkeys(sources, 0.0)
        prev = {}
        heap = [(remaining(v), 0.0, v) for v in sources]
        heapq.heapify(heap)
        while heap:
            _, d, v = heapq.heappop(heap)
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
                    heapq.heappush(heap, (step + remaining(w), step, w))
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


def part_labels(adjacent):
    label = {}
    for start in adjacent:
        if start in label:
            continue
        label[start] = start
        stack = [start]
        while stack:
            v = stack.pop()
            for w in adjacent[v]:
                if w not in label:
                    label[w] = start
                    stack.append(w)
    return label


# every vertex maps to one vertex here, so segments that met still meet
def snap_paths(verts, adjacent, mapped, cuts):
    part = part_labels(adjacent)
    paths = set()
    for a, b in cuts:
        va, vb = mapped[a], mapped[b]
        if va == vb or part.get(va) != part.get(vb):
            continue
        path = cut_path(verts, adjacent, {va}, {vb})
        for x, y in zip(path, path[1:]):
            paths.add(pair(x, y))
    return paths


# one path per extra loop, each the shortest available at the time
def connect_loops(verts, adjacent, comps, weights=None, relief=None):
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
