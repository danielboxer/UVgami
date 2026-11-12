import functools
from collections import namedtuple

import numpy

from .engine_run import (
    INTEGER,
    new_run_folder,
    packed_reader,
    poll_run,
    start_run,
    write_obj,
)
from .mesh import (
    faces_from_arrays,
    find_root,
    next_corners_in_face,
    undirected_edge_keys,
)
from .preseed import SegmentRegions, resolved_surface_modes, write_modes

# from collections import namedtuple
# from .engine_run import EngineError, stdout_path
# from .mesh import edge_keys, following_corners
# from .preseed import submesh
# from .proxy import connected_labels

INPUT_STEM = "segment"
OUTPUT_NAME = "regions.bin"
# INPUT_NAME = "segment.obj"
# LABELS_NAME = "labels.txt"
# DONE_PREFIX = "done: "
# PIECE_PREFIX = "piece: "

# # EdgeFacePairs = namedtuple(
#     "EdgeFacePairs",
#     [
#         "first",
#         "second",
#         # "lengths",
#         "edges",
#     ],
# )

# SegmentOutput = namedtuple("SegmentOutput", ["labels", "region_count", "piece"])

# # the clicked piece carries the new label, piece is the merge tree ancestor the run ended on
# SplitRegion = namedtuple("SplitRegion", ["labels", "piece"])


def _read_regions(path, face_count, *, fine_cut):
    take = packed_reader(path.read_bytes())
    region_count, border_count = take(INTEGER, 2).tolist()
    surface_modes = take(INTEGER, face_count)
    labels = take(INTEGER, face_count)
    edges = take(INTEGER, 2 * border_count).reshape(-1, 2).tolist()
    fine_labels = take(INTEGER, face_count) if fine_cut else None
    return SegmentRegions(
        surface_modes,
        [tuple(edge) for edge in edges],
        region_count,
        labels,
        fine_labels,
        hand_edited=False,
    )


# def _read_labels(path, face_count):
#     stdout = stdout_path(path.parent).read_text().splitlines()
#     done = [line for line in stdout if line.startswith(DONE_PREFIX)]
#     if len(done) != 1:
#         raise EngineError("segmentation engine printed no region count")
#     # region_count = int(done[0][len(DONE_PREFIX) :])
#     # piece_lines = [line for line in stdout if line.startswith(PIECE_PREFIX)]
#     # piece = int(piece_lines[0][len(PIECE_PREFIX) :]) if piece_lines else None
#     labels = numpy.loadtxt(path, dtype=numpy.int64, ndmin=1)
#     if len(labels) != face_count:
#         raise EngineError(
#             f"segmentation engine labelled {len(labels)} faces, expected {face_count}"
#         )
#     return labels
#     # return SegmentOutput(labels, region_count, piece)
#
#
# # # seed is a face whose piece becomes a region, step moves an earlier run's piece, loop_corner names an edge of the seed
# def start_segmentation(
#     segment_command,
#     workdir,
#     verts,
#     faces,
#     # seed=None,
#     # piece=None,
#     # step=None,
#     # loop_corner=None,
# ):
#     workdir = Path(workdir)
#     workdir.mkdir(parents=True, exist_ok=True)
#     folder = Path(tempfile.mkdtemp(dir=workdir))
#     input_path = folder / INPUT_NAME
#     labels_path = folder / LABELS_NAME
#     write_obj(input_path, verts, faces)
#     args = [*(str(part) for part in segment_command), str(input_path), str(labels_path)]
#     # if seed is not None:
#     #     args += ["--seed", str(seed)]
#     # if step is not None:
#     #     args += ["--piece", str(piece), "--step", step]
#     # if loop_corner is not None:
#     #     args += ["--loop-corner", str(loop_corner)]
#     process = launch(folder, args, None)
#     return EngineRun(process, folder, labels_path, len(faces), _read_labels)


# (face, face, (low, high) vertex pair) per edge two faces share
EdgeFacePairs = namedtuple("EdgeFacePairs", "first second edges")


def edge_face_pairs(corners, totals):
    face_of_corner = numpy.repeat(numpy.arange(len(totals)), totals)
    next_corner = next_corners_in_face(totals)
    low = numpy.minimum(corners, corners[next_corner])
    high = numpy.maximum(corners, corners[next_corner])
    keys = undirected_edge_keys(corners, corners[next_corner])
    order = numpy.argsort(keys, kind="stable")
    shared = keys[order][:-1] == keys[order][1:]
    first = face_of_corner[order[:-1][shared]]
    second = face_of_corner[order[1:][shared]]
    distinct = first != second
    first, second = first[distinct], second[distinct]
    edge_corners = order[:-1][shared][distinct]
    edges = numpy.stack([low[edge_corners], high[edge_corners]], axis=1)
    return EdgeFacePairs(first, second, edges)


def pair_borders(face_pairs, fine_labels):
    first, second = fine_labels[face_pairs.first], fine_labels[face_pairs.second]
    between = (first != second) & (first > 0) & (second > 0)
    low = numpy.minimum(first, second)[between]
    high = numpy.maximum(first, second)[between]
    edges = face_pairs.edges[between]
    borders = {}
    for pair in numpy.unique(numpy.stack([low, high], axis=1), axis=0).tolist():
        on_pair = (low == pair[0]) & (high == pair[1])
        # an edge with three or more faces is listed once per face pair
        borders[tuple(pair)] = numpy.unique(edges[on_pair], axis=0)
    return borders


def pair_edges(borders, pairs):
    edges = [borders[pair] for pair in sorted(pairs)]
    if not edges:
        return []
    return [
        tuple(edge) for edge in numpy.unique(numpy.concatenate(edges), axis=0).tolist()
    ]


def pairs_between_regions(borders, labels, fine_labels):
    region_of_fine = {}
    for fine, region in zip(fine_labels.tolist(), labels.tolist()):
        region_of_fine.setdefault(fine, region)
    return {
        pair for pair in borders if region_of_fine[pair[0]] != region_of_fine[pair[1]]
    }


def joined_labels(fine_labels, borders, on_pairs):
    parent = list(range(int(fine_labels.max()) + 1))
    for low, high in borders:
        if (low, high) not in on_pairs:
            parent[find_root(parent, high)] = find_root(parent, low)
    roots = numpy.array([find_root(parent, fine) for fine in range(len(parent))])
    segmented = fine_labels > 0
    joined_roots, dense = numpy.unique(
        roots[fine_labels[segmented]], return_inverse=True
    )
    labels = numpy.zeros(len(fine_labels), dtype=numpy.int64)
    labels[segmented] = dense + 1
    return labels, len(joined_roots)


def on_pairs_with_edges(borders, edges):
    flagged = set(map(tuple, numpy.asarray(edges).reshape(-1, 2).tolist()))
    return {
        pair
        for pair, pair_edges in borders.items()
        if any(tuple(edge) in flagged for edge in pair_edges.tolist())
    }


def nested_fine_labels(labels, fine_labels):
    span = int(fine_labels.max()) + 1
    pairs = numpy.where(fine_labels > 0, labels * span + fine_labels, 0)
    _, dense = numpy.unique(pairs[pairs > 0], return_inverse=True)
    nested = numpy.zeros(len(fine_labels), dtype=numpy.int64)
    nested[pairs > 0] = dense + 1
    return nested


def segment_job(
    command,
    workdir,
    positions,
    corners,
    totals,
    surface_modes,
    kept_faces,
    # stored_labels=None,
    *,
    fine_cut,
    cancelled=None,
    progress=None,
):
    faces = faces_from_arrays(corners, totals)
    folder = new_run_folder(workdir)
    input_path = folder / f"{INPUT_STEM}.obj"
    write_obj(input_path, positions.tolist(), faces)
    write_modes(folder / f"{INPUT_STEM}_modes", surface_modes)
    if kept_faces.any():
        (folder / f"{INPUT_STEM}_kept").write_text(
            "".join(f"{int(kept)}\n" for kept in kept_faces.tolist())
        )
    # if stored_labels is not None and stored_labels.any():
    #     labels[segmented] = stored_labels[segmented]
    output_path = folder / OUTPUT_NAME
    fine_flags = ["--fine"] if fine_cut else []
    run = start_run(
        folder,
        [*command.args, "segment", input_path, output_path, *fine_flags],
        command.env,
        output_path,
        len(faces),
        functools.partial(_read_regions, fine_cut=fine_cut),
    )
    return poll_run(run, cancelled, progress)


def stored_segment_job(
    command,
    workdir,
    positions,
    corners,
    totals,
    surface_modes,
    fine_labels,
    border_edges,
    cancelled=None,
):
    faces = faces_from_arrays(corners, totals)
    resolved = resolved_surface_modes(
        command, workdir, positions.tolist(), faces, surface_modes, cancelled
    )
    borders = pair_borders(edge_face_pairs(corners, totals), fine_labels)
    on_pairs = on_pairs_with_edges(borders, border_edges)
    labels, region_count = joined_labels(fine_labels, borders, on_pairs)
    return SegmentRegions(
        resolved,
        pair_edges(borders, on_pairs),
        region_count,
        labels,
        fine_labels,
        hand_edited=True,
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
#     sub_verts, sub_faces = submesh(positions.tolist(), faces, region_faces.tolist())
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
#     sub_verts, sub_faces = submesh(positions.tolist(), faces, part_faces.tolist())
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
