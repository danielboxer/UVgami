import tempfile
from collections import namedtuple
from pathlib import Path

import numpy

from .engine_run import EngineError, EngineRun, launch, poll_run, stdout_path, write_obj
from .mesh import faces_from_arrays
from .preseed import SURFACE_MODE_CODES, hard_faces, submesh

# from .proxy_transfer import connected_labels

INPUT_NAME = "segment.obj"
LABELS_NAME = "labels.txt"
DONE_PREFIX = "done: "
# PIECE_PREFIX = "piece: "

# labels number regions from one, zero outside them, border_edges are (low, high) pairs
SegmentRegions = namedtuple(
    "SegmentRegions",
    [
        "surface_modes",
        "border_edges",
        "region_count",
        "labels",
    ],
)

# (face, face, (low, high) vertex pair) per edge two faces share
EdgeFacePairs = namedtuple(
    "EdgeFacePairs",
    [
        "first",
        "second",
        # "lengths",
        "edges",
    ],
)

# SegmentOutput = namedtuple("SegmentOutput", ["labels", "region_count", "piece"])

# # the clicked piece carries the new label, piece is the merge tree ancestor the run ended on
# SplitRegion = namedtuple("SplitRegion", ["labels", "piece"])


def _read_labels(path, face_count):
    stdout = stdout_path(path.parent).read_text().splitlines()
    done = [line for line in stdout if line.startswith(DONE_PREFIX)]
    if len(done) != 1:
        raise EngineError("segmentation engine printed no region count")
    # region_count = int(done[0][len(DONE_PREFIX) :])
    # piece_lines = [line for line in stdout if line.startswith(PIECE_PREFIX)]
    # piece = int(piece_lines[0][len(PIECE_PREFIX) :]) if piece_lines else None
    labels = numpy.loadtxt(path, dtype=numpy.int64, ndmin=1)
    if len(labels) != face_count:
        raise EngineError(
            f"segmentation engine labelled {len(labels)} faces, expected {face_count}"
        )
    return labels
    # return SegmentOutput(labels, region_count, piece)


# # seed is a face whose piece becomes a region, step moves an earlier run's piece, loop_corner names an edge of the seed
def start_segmentation(
    segment_command,
    workdir,
    verts,
    faces,
    # seed=None,
    # piece=None,
    # step=None,
    # loop_corner=None,
):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(dir=workdir))
    input_path = folder / INPUT_NAME
    labels_path = folder / LABELS_NAME
    write_obj(input_path, verts, faces)
    args = [*(str(part) for part in segment_command), str(input_path), str(labels_path)]
    # if seed is not None:
    #     args += ["--seed", str(seed)]
    # if step is not None:
    #     args += ["--piece", str(piece), "--step", step]
    # if loop_corner is not None:
    #     args += ["--loop-corner", str(loop_corner)]
    process = launch(folder, args, None)
    return EngineRun(process, folder, labels_path, len(faces), _read_labels)


# Auto faces get the classifier's pick
def resolve_surface_modes(verts, faces, surface_modes, cancelled):
    hard = numpy.fromiter(
        hard_faces(verts, faces, surface_modes, cancelled), dtype=numpy.int64
    )
    resolved = surface_modes.copy()
    resolved[resolved == SURFACE_MODE_CODES["AUTO"]] = SURFACE_MODE_CODES["ORGANIC"]
    resolved[hard] = SURFACE_MODE_CODES["HARD"]
    return resolved


def classify_job(positions, corners, totals, surface_modes, cancelled=None):
    faces = faces_from_arrays(corners, totals)
    return resolve_surface_modes(positions.tolist(), faces, surface_modes, cancelled)


def edge_face_pairs(positions, corners, totals):
    face_count = len(totals)
    starts = numpy.cumsum(totals) - totals
    face_of_corner = numpy.repeat(numpy.arange(face_count), totals)
    offsets = numpy.arange(len(corners)) - starts[face_of_corner]
    next_corner = starts[face_of_corner] + (offsets + 1) % totals[face_of_corner]
    low = numpy.minimum(corners, corners[next_corner])
    high = numpy.maximum(corners, corners[next_corner])
    keys = low * len(positions) + high
    order = numpy.argsort(keys, kind="stable")
    shared = keys[order][:-1] == keys[order][1:]
    first = face_of_corner[order[:-1][shared]]
    second = face_of_corner[order[1:][shared]]
    distinct = first != second
    first, second = first[distinct], second[distinct]
    edge_corners = order[:-1][shared][distinct]
    edges = numpy.stack([low[edge_corners], high[edge_corners]], axis=1)
    # lengths = numpy.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    return EdgeFacePairs(first, second, edges)


def border_edges(face_pairs, labels):
    differ = labels[face_pairs.first] != labels[face_pairs.second]
    # an edge with three or more faces is listed once per face pair
    edges = numpy.unique(face_pairs.edges[differ], axis=0)
    return [tuple(edge) for edge in edges.tolist()]


# # the clicked region joins the neighbor it shares the longest border with
# def merge_region(labels, face_pairs, region):
#     inside = labels == region
#     across = inside[face_pairs.first] != inside[face_pairs.second]
#     neighbors = numpy.where(
#         inside[face_pairs.first], labels[face_pairs.second], labels[face_pairs.first]
#     )[across]
#     lengths = face_pairs.lengths[across]
#     in_region = neighbors > 0
#     if not in_region.any():
#         raise EngineError("the region touches no other region")
#     shared = numpy.bincount(neighbors[in_region], weights=lengths[in_region])
#     merged = labels.copy()
#     merged[inside] = int(numpy.argmax(shared))
#     return merged


def segment_job(
    segment_command,
    workdir,
    positions,
    corners,
    totals,
    surface_modes,
    kept_faces,
    # stored_labels=None,
    cancelled=None,
):
    verts = positions.tolist()
    faces = faces_from_arrays(corners, totals)
    resolved = resolve_surface_modes(verts, faces, surface_modes, cancelled)
    segmented = (resolved == SURFACE_MODE_CODES["ORGANIC"]) & ~kept_faces
    labels = numpy.zeros(len(faces), dtype=numpy.int64)
    if not segmented.any():
        return SegmentRegions(resolved, [], 0, labels)

    # if stored_labels is not None and stored_labels.any():
    #     labels[segmented] = stored_labels[segmented]
    # else:
    segmented_faces = numpy.flatnonzero(segmented)
    sub_verts, sub_faces, _ = submesh(verts, faces, segmented_faces.tolist(), ())
    run = start_segmentation(segment_command, workdir, sub_verts, sub_faces)
    labels[segmented_faces] = poll_run(run, cancelled) + 1
    region_count = len(numpy.unique(labels[labels > 0]))
    face_pairs = edge_face_pairs(positions, corners, totals)
    return SegmentRegions(
        resolved, border_edges(face_pairs, labels), region_count, labels
    )


# # splits the region at a node of the engine's merge tree
# def split_region_job(
#     segment_command,
#     workdir,
#     positions,
#     corners,
#     totals,
#     labels,
#     seed,
#     piece,
#     step,
#     cancelled=None,
# ):
#     faces = faces_from_arrays(corners, totals)
#     region_faces = numpy.flatnonzero(labels == labels[seed])
#     sub_verts, sub_faces, _ = submesh(
#         positions.tolist(), faces, region_faces.tolist(), ()
#     )
#     sub_seed = int(numpy.searchsorted(region_faces, seed))
#     run = start_segmentation(
#         segment_command,
#         workdir,
#         sub_verts,
#         sub_faces,
#         seed=sub_seed,
#         piece=piece,
#         step=step,
#     )
#     output = poll_run(run, cancelled)
#     if output.region_count < 2:
#         raise EngineError("no piece around the click passes the border check")
#     split = labels.copy()
#     split[region_faces[output.labels == output.labels[sub_seed]]] = labels.max() + 1
#     return SplitRegion(split, output.piece)


# # splits the clicked region along the closed edge loop through one edge of the seed face
# def split_loop_job(
#     segment_command,
#     workdir,
#     positions,
#     corners,
#     totals,
#     labels,
#     seed,
#     corner,
#     cancelled=None,
# ):
#     faces = faces_from_arrays(corners, totals)
#     face_pairs = edge_face_pairs(positions, corners, totals)
#     # the loop runs over the whole loose part, so it can close through other regions
#     part = connected_labels(
#         len(faces), numpy.stack([face_pairs.first, face_pairs.second], axis=1)
#     )
#     part_faces = numpy.flatnonzero(part == part[seed])
#     sub_verts, sub_faces, _ = submesh(
#         positions.tolist(), faces, part_faces.tolist(), ()
#     )
#     sub_seed = int(numpy.searchsorted(part_faces, seed))
#     run = start_segmentation(
#         segment_command,
#         workdir,
#         sub_verts,
#         sub_faces,
#         seed=sub_seed,
#         loop_corner=corner,
#     )
#     output = poll_run(run, cancelled)
#     in_region = labels[part_faces] == labels[seed]
#     seed_side = output.labels == output.labels[sub_seed]
#     if not (in_region & ~seed_side).any():
#         raise EngineError("the edge loop does not cut the clicked region")
#     split = labels.copy()
#     split[part_faces[in_region & seed_side]] = labels.max() + 1
#     return split
