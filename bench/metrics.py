"""UV-quality metrics for an output OBJ. bpy-free, numpy only.

Every metric is scale-invariant so the two engines compare on equal footing
regardless of model or UV scale.
"""

import numpy as np

# a triangle counts as degenerate when its area is this fraction of the mesh
# total or less; relative so the test is scale-invariant
DEGENERATE_FRACTION = 1e-12


def _resolve(index, count):
    """OBJ index to 0-based, handling negative (relative) references."""
    return index - 1 if index > 0 else count + index


def parse_obj(path):
    """Return (positions Nx3, uvs Mx2, face_pos Kx3, face_uv Kx3), polygons
    fan-triangulated. Faces without a vt on every corner are skipped."""
    positions = []
    uvs = []
    face_pos = []
    face_uv = []
    with open(path) as file:
        for line in file:
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "v":
                positions.append([float(v) for v in parts[1:4]])
            elif tag == "vt":
                uvs.append([float(parts[1]), float(parts[2])])
            elif tag == "f":
                corners = []
                for token in parts[1:]:
                    fields = token.split("/")
                    if len(fields) < 2 or fields[1] == "":
                        corners = None
                        break
                    corners.append(
                        (
                            _resolve(int(fields[0]), len(positions)),
                            _resolve(int(fields[1]), len(uvs)),
                        )
                    )
                if not corners:
                    continue
                for i in range(1, len(corners) - 1):
                    tri = (corners[0], corners[i], corners[i + 1])
                    face_pos.append([c[0] for c in tri])
                    face_uv.append([c[1] for c in tri])
    return (
        np.array(positions, dtype=np.float64),
        np.array(uvs, dtype=np.float64),
        np.array(face_pos, dtype=np.int64).reshape(-1, 3),
        np.array(face_uv, dtype=np.int64).reshape(-1, 3),
    )


def _chart_count(face_uv):
    # connected components of faces linked by shared vt-index edges (UV islands)
    parent = list(range(len(face_uv)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edge_owner = {}
    for face, tri in enumerate(face_uv):
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            key = (a, b) if a < b else (b, a)
            other = edge_owner.get(key)
            if other is None:
                edge_owner[key] = face
            else:
                parent[find(face)] = find(other)
    return len({find(f) for f in range(len(face_uv))})


def _weld(positions):
    """Map vertices sharing a 3D position to one canonical id. Engines that split
    a mesh into UV islands duplicate boundary vertices, so shared 3D edges must be
    matched by coordinate, not by index. Returns (canonical id per vertex, reps)."""
    span = positions.max(axis=0) - positions.min(axis=0)
    tol = float(np.linalg.norm(span)) * 1e-6 or 1e-6
    keys = np.round(positions / tol).astype(np.int64)
    canon = {}
    ids = np.empty(len(positions), dtype=np.int64)
    reps = []
    for i, key in enumerate(map(tuple, keys)):
        cid = canon.get(key)
        if cid is None:
            cid = canon[key] = len(reps)
            reps.append(positions[i])
        ids[i] = cid
    return ids, np.array(reps)


def _seam_length(reps, welded_pos, face_uv, total_area_3d):
    # a 3D edge shared by two faces is a seam when the two faces disagree on the
    # vt indices at its endpoints; total seam length normalized by sqrt(area)
    edges = {}
    for tri_p, tri_t in zip(welded_pos, face_uv):
        for k in range(3):
            pa, pb = int(tri_p[k]), int(tri_p[(k + 1) % 3])
            ta, tb = int(tri_t[k]), int(tri_t[(k + 1) % 3])
            if pa < pb:
                key, vt = (pa, pb), (ta, tb)
            else:
                key, vt = (pb, pa), (tb, ta)
            edges.setdefault(key, []).append(vt)
    seam = 0.0
    for (pa, pb), vts in edges.items():
        if len(vts) >= 2 and len(set(vts)) > 1:
            seam += float(np.linalg.norm(reps[pa] - reps[pb]))
    return seam / np.sqrt(total_area_3d) if total_area_3d > 0 else 0.0


def compute_metrics(path):
    positions, uvs, face_pos, face_uv = parse_obj(path)
    if len(face_pos) == 0 or len(uvs) == 0:
        return {
            "chart_count": 0,
            "seam_length": 0.0,
            "area_distortion": 0.0,
            "angle_distortion": 0.0,
            "uv_utilization": 0.0,
            "degenerate_tris": 0,
        }

    p = positions[face_pos]  # (K,3,3)
    t = uvs[face_uv]  # (K,3,2)
    area_3d = 0.5 * np.linalg.norm(
        np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1
    )
    area_uv = 0.5 * np.abs(
        (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
        - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1])
    )
    total_3d = float(area_3d.sum())
    total_uv = float(area_uv.sum())

    eps_3d = DEGENERATE_FRACTION * total_3d
    eps_uv = DEGENERATE_FRACTION * total_uv
    valid = (area_3d > eps_3d) & (area_uv > eps_uv)
    degenerate = int((~valid).sum())

    chart_count = _chart_count(face_uv)
    welded_ids, reps = _weld(positions)
    seam_length = _seam_length(reps, welded_ids[face_pos], face_uv, total_3d)
    uv_bbox = uvs[np.unique(face_uv)]
    span = uv_bbox.max(axis=0) - uv_bbox.min(axis=0)
    bbox_area = float(span[0] * span[1])
    uv_utilization = total_uv / bbox_area if bbox_area > 0 else 0.0

    pv = p[valid]
    a3 = area_3d[valid]
    r = (area_uv[valid] / total_uv) / (a3 / total_3d)
    area_distortion = float(np.average(np.maximum(r, 1.0 / r), weights=a3))

    # Sander et al. L2 stretch: singular values of the texture->surface Jacobian,
    # on UVs globally scaled so total UV area equals total 3D area (isometry -> 1)
    scale = np.sqrt(total_3d / total_uv)
    tv = uvs[face_uv][valid] * scale
    s0, s1, s2 = tv[:, 0, 0], tv[:, 1, 0], tv[:, 2, 0]
    d0, d1, d2 = tv[:, 0, 1], tv[:, 1, 1], tv[:, 2, 1]
    double_area = (s1 - s0) * (d2 - d0) - (s2 - s0) * (d1 - d0)
    inv = 1.0 / double_area[:, None]
    ss = inv * (
        pv[:, 0] * (d1 - d2)[:, None]
        + pv[:, 1] * (d2 - d0)[:, None]
        + pv[:, 2] * (d0 - d1)[:, None]
    )
    st = inv * (
        pv[:, 0] * (s2 - s1)[:, None]
        + pv[:, 1] * (s0 - s2)[:, None]
        + pv[:, 2] * (s1 - s0)[:, None]
    )
    l2_sq = (np.einsum("ij,ij->i", ss, ss) + np.einsum("ij,ij->i", st, st)) / 2.0
    angle_distortion = float(np.sqrt(np.average(l2_sq, weights=a3)))

    return {
        "chart_count": chart_count,
        "seam_length": seam_length,
        "area_distortion": area_distortion,
        "angle_distortion": angle_distortion,
        "uv_utilization": uv_utilization,
        "degenerate_tris": degenerate,
    }
