# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import math
from dataclasses import dataclass

import numpy as np

# planner is bpy-free: it turns plain mesh data into a complete uv transfer plan
# or a structured failure. loops are numbered consecutively in polygon order,
# matching blender's poly.loop_start layout.


@dataclass
class TransferPlan:
    loop_uvs: dict  # input loop index -> (u, v)
    seam_edges: set  # sorted (input v0, input v1) tuples
    exact_topology: bool
    ok: bool = True


@dataclass
class TransferFailure:
    reason: str  # machine string: vertex_match, ambiguous_geometry, ...
    detail: str
    ok: bool = False


def _default_tol(positions):
    if len(positions) == 0:
        return 1e-6
    diag = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    return max(diag * 1e-5, 1e-9)


def _grid_key(p, inv):
    return (
        int(math.floor(p[0] * inv)),
        int(math.floor(p[1] * inv)),
        int(math.floor(p[2] * inv)),
    )


def _build_grid(positions, inv):
    grid = {}
    for i in range(len(positions)):
        grid.setdefault(_grid_key(positions[i], inv), []).append(i)
    return grid


def _query_grid(grid, inv, positions, p, tol2):
    base = _grid_key(p, inv)
    found = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                bucket = grid.get((base[0] + dx, base[1] + dy, base[2] + dz))
                if not bucket:
                    continue
                for i in bucket:
                    d = positions[i] - p
                    if float(d @ d) <= tol2:
                        found.append(i)
    return found


def plan_transfer(
    input_positions,
    input_polygons,
    output_positions,
    output_polygons,
    output_uvs,
    output_seams,
    tol=None,
):
    in_pos = np.asarray(input_positions, dtype=float)
    out_pos = np.asarray(output_positions, dtype=float)

    if tol is None:
        tol = _default_tol(in_pos)
    tol2 = tol * tol
    inv = 1.0 / tol

    in_grid = _build_grid(in_pos, inv)

    faces_by_vertex = [[] for _ in range(len(in_pos))]
    input_face_vsets = []
    input_vertex_local = []
    input_loop_start = []
    loop_cursor = 0
    for fi, poly in enumerate(input_polygons):
        input_loop_start.append(loop_cursor)
        loop_cursor += len(poly)
        input_face_vsets.append(set(poly))
        local = {}
        for corner, v in enumerate(poly):
            faces_by_vertex[v].append(fi)
            local[v] = corner
        input_vertex_local.append(local)
    total_input_loops = loop_cursor

    candidate_cache = {}

    def candidates(out_v):
        cached = candidate_cache.get(out_v)
        if cached is None:
            cached = _query_grid(in_grid, inv, in_pos, out_pos[out_v], tol2)
            candidate_cache[out_v] = cached
        return cached

    loop_uvs = {}
    any_subset = False

    for fo, out_verts in enumerate(output_polygons):
        corner_candidates = []
        for out_v in out_verts:
            cands = candidates(out_v)
            if not cands:
                return TransferFailure(
                    "vertex_match",
                    f"output vertex {out_v} has no matching input vertex",
                )
            corner_candidates.append(cands)

        # input faces incident to a candidate of every output corner
        cand_faces = set()
        for v in corner_candidates[0]:
            cand_faces.update(faces_by_vertex[v])
        for corner in corner_candidates[1:]:
            allowed = set()
            for v in corner:
                allowed.update(faces_by_vertex[v])
            cand_faces &= allowed
            if not cand_faces:
                break
        if not cand_faces:
            return TransferFailure(
                "face_match", f"output face {fo} maps to no input face"
            )

        # each surviving face must give one input vertex per output corner
        valid = []
        for f in cand_faces:
            fvs = input_face_vsets[f]
            assign = []
            for corner in corner_candidates:
                matches = [v for v in corner if v in fvs]
                if len(matches) > 1:
                    return TransferFailure(
                        "ambiguous_geometry",
                        f"output face {fo} corner matches multiple vertices"
                        " of one input face",
                    )
                assign.append(matches[0])
            valid.append((f, assign))

        if len(valid) > 1:
            return TransferFailure(
                "ambiguous_geometry", f"output face {fo} matches multiple input faces"
            )
        f, assign = valid[0]

        if len(out_verts) < len(input_polygons[f]):
            any_subset = True

        local = input_vertex_local[f]
        base = input_loop_start[f]
        face_uvs = output_uvs[fo]
        for corner, in_v in enumerate(assign):
            in_loop = base + local[in_v]
            uv = (float(face_uvs[corner][0]), float(face_uvs[corner][1]))
            prev = loop_uvs.get(in_loop)
            if prev is None:
                loop_uvs[in_loop] = uv
            elif abs(prev[0] - uv[0]) > 1e-6 or abs(prev[1] - uv[1]) > 1e-6:
                return TransferFailure(
                    "uv_conflict",
                    f"input loop {in_loop} assigned conflicting uvs",
                )

    if len(loop_uvs) != total_input_loops:
        missing = total_input_loops - len(loop_uvs)
        return TransferFailure(
            "incomplete_coverage", f"{missing} input loops received no uv"
        )

    input_edges = set()
    for poly in input_polygons:
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            input_edges.add((a, b) if a < b else (b, a))

    seam_edges = set()
    for oa, ob in output_seams:
        for a in candidates(oa):
            for b in candidates(ob):
                key = (a, b) if a < b else (b, a)
                if key in input_edges:
                    seam_edges.add(key)

    exact_topology = not any_subset and len(output_polygons) == len(input_polygons)
    return TransferPlan(loop_uvs, seam_edges, exact_topology)
