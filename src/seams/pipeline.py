"""The entry points. seam_edges runs the whole thing: partition and
merges (feature_labels), boundary cleanup, then the surviving region
boundaries plus disk cuts are the seams. is_hard_surface reads the
same region structure to decide whether a loose part has features
worth preseeding. Benchmarks against the per-edge test:
docs/agents/bench-results.md."""

import collections

from .boundaries import boundary_edges, flatten_teeth, reroute_boundaries
from .cuts import crease_relief, disk_cuts
from .mesh import LOW_ANGLE, build, diagonal, norm, turn_angle
from .regions import (
    CREASE_ANGLE,
    absorb,
    close_rings,
    detect_width,
    merge_flat,
    merge_smooth,
    partition,
)
from .sweeps import split_sweeps


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


# auto mode guards, tuned on the bench sets (docs/agents/bench-results.md).
# a part is hard surface only when every one holds:
# one region covering the part means no structure, a smooth blob,
ORGANIC_SHARE = 0.9
# regions averaging under this many faces mean the partition found noise
# (chainmail reads 1 face per region),
FRAGMENT_FACES = 8
# turn between LOW_ANGLE and this is spread curvature. a bevel is spread
# turn beside a boundary and a sculpt is spread turn everywhere, so the
# share is read away from the region boundaries,
SPREAD_ANGLE = 25
SPREAD_SHARE = 0.21
# and the boundaries must mostly be deliberate: creased, or a rim
# split_sweeps placed
BOUNDARY_ANGLE = 20
BOUNDARY_CREASED = 0.6


def is_hard_surface(verts, faces):
    """Whether a loose part's features are worth preseeding.

    Reads the merged region structure at the CREASE_ANGLE floor, not the
    feature angle knob: the question is whether structure exists at all.
    split_sweeps runs so a smooth cylinder still reads hard, its rims count
    as deliberate boundaries. Misreading organic costs a slow from-scratch
    unwrap, misreading hard costs seams on sculpt ridges, so ties fall
    organic.
    """
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
        not interior or spread / interior < SPREAD_SHARE
    ) and boundary_creased / boundary >= BOUNDARY_CREASED


def seam_edges(verts, faces, angle=CREASE_ANGLE, rims=True, weights=None, forced=None):
    """The full pipeline at auto width: partition, merges, cleanup, seams.

    angle is what counts as a feature: boundaries turning less than it
    merge away, so lower keeps more shallow-feature seams at the cost of
    shattering coarse curved walls. rims off skips split_sweeps, so a
    smooth cylinder keeps its end caps. weights are painted restrictions
    the cuts avoid, region boundaries sit where the shape says and paint
    does not move them. forced edges cut from the partition on, so the
    merges route around them, and they are seams in the end.
    """
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
        # a closed mesh that merged seamless cannot flatten, every feature
        # sat under the angle, so retry at the floor
        return seam_edges(verts, faces, CREASE_ANGLE, rims, weights, forced)
    return seams
