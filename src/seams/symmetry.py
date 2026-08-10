"""Symmetric output without bisecting the mesh: close the preseed's seam set
under the mesh's own mirror maps, then stack the mirrored uv islands after
the unwrap."""

import collections

from .mesh import face_edges, pair, uv_island_groups


def mirror_seams(seams, mirrors, edges):
    """Close seams under these vertex mirror maps, so every seam's mirror
    image is a seam too. The maps may be partial, a seam outside their
    coverage stays as it is. Only pairs present in edges are added, which
    keeps a subset detection from marking edges outside its own faces."""
    result = set(seams)
    queue = list(result)
    while queue:
        a, b = queue.pop()
        for m in mirrors:
            if a not in m or b not in m:
                continue
            key = pair(m[a], m[b])
            if key in edges and key not in result:
                result.add(key)
                queue.append(key)
    return result


def stack_mirrored(faces, uvs, mirrors):
    """Corner assignments that stack each set of mirrored islands: every
    island in a mirror-connected group takes the uvs of the group's first
    island, corner for corner through the composed mirror map, so the copies
    overlap exactly and the pack keeps them together. An island a map sends
    onto itself straddles the plane and stays put. Returns (target face,
    target corner, source face, source corner) tuples: the caller copies raw
    uv values so the stack survives 9-decimal matching."""
    groups = uv_island_groups(faces, uvs, face_edges(faces))
    group_of = {}
    for gi, group in enumerate(groups):
        for fi in group:
            group_of[fi] = gi
    by_verts = collections.defaultdict(list)
    for fi, face in enumerate(faces):
        by_verts[tuple(sorted(face))].append(fi)

    def image(gi, m):
        """The one island m sends this island onto whole, or None."""
        targets = set()
        for fi in groups[gi]:
            candidates = by_verts.get(tuple(sorted(m.get(v, -1) for v in faces[fi])))
            if not candidates:
                return None
            targets.update(group_of[c] for c in candidates)
        if len(targets) == 1:
            target = targets.pop()
            if len(groups[target]) == len(groups[gi]):
                return target
        return None

    links = collections.defaultdict(list)
    for gi in range(len(groups)):
        for m in mirrors:
            hi = image(gi, m)
            if hi is not None and hi != gi:
                links[gi].append((hi, m))

    assignments = []
    seen = set()
    for gi in range(len(groups)):
        if gi in seen:
            continue
        component = {gi}
        queue = [gi]
        while queue:
            for hi, _ in links[queue.pop()]:
                if hi not in component:
                    component.add(hi)
                    queue.append(hi)
        seen |= component
        if len(component) == 1:
            continue
        rep = min(component, key=lambda g: min(groups[g]))
        # composed[g] maps each rep-island vertex to its counterpart in g
        composed = {rep: {v: v for fi in groups[rep] for v in faces[fi]}}
        queue = [rep]
        while queue:
            current = queue.pop()
            for hi, m in links[current]:
                if hi not in composed:
                    composed[hi] = {v: m[w] for v, w in composed[current].items()}
                    queue.append(hi)
        for twin, vmap in composed.items():
            if twin == rep:
                continue
            for fi in groups[rep]:
                key = tuple(sorted(vmap[v] for v in faces[fi]))
                target = next(c for c in by_verts[key] if group_of[c] == twin)
                for corner, v in enumerate(faces[fi]):
                    assignments.append(
                        (target, faces[target].index(vmap[v]), fi, corner)
                    )
    return assignments
