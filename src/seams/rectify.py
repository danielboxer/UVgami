import math

from .mesh import signed_area

# a boundary triangle pinned collinear reads as flipped at float noise
FLIP_NOISE = 1e-6


# scale-free symmetric Dirichlet, 4.0 at isometry
def flatten_distortion(verts, faces, uvs, group):
    signed_total = sum(signed_area(uvs[fi]) for fi in group)
    orientation = 1.0 if signed_total >= 0 else -1.0
    floor = FLIP_NOISE * abs(signed_total)
    grow = shrink = total = 0.0
    for fi in group:
        face = faces[fi]
        face_uv = uvs[fi]
        p0 = verts[face[0]]
        u0 = face_uv[0]
        for i in range(1, len(face) - 1):
            p1 = verts[face[i]]
            p2 = verts[face[i + 1]]
            e1x, e1y, e1z = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
            e2x, e2y, e2z = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
            length = math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
            sx = e1y * e2z - e1z * e2y
            sy = e1z * e2x - e1x * e2z
            sz = e1x * e2y - e1y * e2x
            area = math.sqrt(sx * sx + sy * sy + sz * sz) / 2
            u1 = face_uv[i]
            u2 = face_uv[i + 1]
            uv_area = (
                (
                    (u0[0] * u1[1] - u1[0] * u0[1])
                    + (u1[0] * u2[1] - u2[0] * u1[1])
                    + (u2[0] * u0[1] - u0[0] * u2[1])
                )
                / 2
                * orientation
            )
            if length <= 0 or area <= 0 or abs(uv_area) <= floor:
                continue
            if uv_area < 0:
                return math.inf
            x2 = (e1x * e2x + e1y * e2y + e1z * e2z) / length
            y2 = 2 * area / length
            a = (u1[0] - u0[0]) * orientation / length
            b = ((u2[0] - u0[0]) * orientation - x2 * a) / y2
            c = (u1[1] - u0[1]) / length
            d = (u2[1] - u0[1] - x2 * c) / y2
            det = a * d - b * c
            if det <= 0:
                return math.inf
            frob2 = a * a + b * b + c * c + d * d
            grow += area * frob2
            shrink += area * frob2 / (det * det)
            total += area
    if total <= 0:
        return math.inf
    return 2 * math.sqrt(grow * shrink) / total
