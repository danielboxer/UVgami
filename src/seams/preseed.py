"""Preseed uvs without Blender.

The flatten solve and the pack run in the optcuts binary's flatten mode,
everything here is plain data: verts, polygon faces, per-face corner uvs.
hard_surface.py adapts a Blender mesh onto these calls, so this module and
the repair logic it drives stay unit-testable and can run off the main
thread."""

import math
import subprocess
from pathlib import Path

from .islands import island_ruined, split_islands
from .mesh import face_edges, island_groups, signed_area
from .pipeline import seam_edges
from .regions import CREASE_ANGLE

# slim iterations for re-unwrapping a folded island alone. 50 flattens the
# long folded strips, but applied globally it breaks other models, so only
# islands that already came out ruined get it (the engine's default is 10)
REPAIR_ITERATIONS = 50
# repair rounds: a split piece can come out of its own unwrap still ruined,
# so repair and split repeat until clean, this many times at most. Each round
# halves the stubborn pieces, so this is plenty (pipe_wrench needs 4), and a
# clean model exits on round one
REPAIR_ROUNDS = 6


class FlattenError(RuntimeError):
    pass


class FlattenEngine:
    """Client for the engine's flatten mode. workdir holds the obj roundtrip
    files and is reused call to call."""

    def __init__(self, engine_path, workdir):
        self.engine_path = str(engine_path)
        self.workdir = Path(workdir)

    def flatten(self, verts, faces, seams, iterations=10):
        """Per-face corner uvs for faces cut along seams, packed."""
        return self._run(verts, faces, seams=seams, iterations=iterations)

    def pack(self, verts, faces, uvs):
        """The same uvs with island scale averaged and repacked."""
        return self._run(verts, faces, uvs=uvs)

    def _run(self, verts, faces, seams=None, uvs=None, iterations=None):
        self.workdir.mkdir(parents=True, exist_ok=True)
        obj_path = self.workdir / "flatten.obj"
        seam_path = self.workdir / "flatten_seams"
        out_dir = self.workdir / "flatten_out"

        with obj_path.open("w") as f:
            for x, y, z in verts:
                f.write(f"v {x} {y} {z}\n")
            if uvs is None:
                for face in faces:
                    f.write("f " + " ".join(str(v + 1) for v in face) + "\n")
            else:
                # corners at one vertex with one uv value are one uv-vertex,
                # which is how the engine reads islands back out of the file
                index = {}
                face_lines = []
                for face, corner_uvs in zip(faces, uvs):
                    refs = []
                    for v, uv in zip(face, corner_uvs):
                        t = index.setdefault((v, uv), len(index))
                        refs.append(f"{v + 1}/{t + 1}")
                    face_lines.append("f " + " ".join(refs) + "\n")
                for _, uv in index:
                    f.write(f"vt {uv[0]} {uv[1]}\n")
                f.writelines(face_lines)

        if seams:
            seam_path.write_text("".join(f"{a} {b}\n" for a, b in sorted(seams)))
        elif seam_path.exists():
            seam_path.unlink()

        args = [self.engine_path, "-i", str(obj_path), "-o", str(out_dir)]
        if uvs is None:
            args.append("-flatten")
            if iterations is not None:
                args += ["-flatten_iters", str(iterations)]
        else:
            args.append("-pack_only")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            result = subprocess.run(
                args, capture_output=True, text=True, creationflags=creationflags
            )
        except OSError as error:
            raise FlattenError(f"flatten engine failed to start: {error}") from error
        if result.returncode != 0:
            raise FlattenError(
                f"flatten engine exited {result.returncode}: {result.stderr.strip()}"
            )
        return _read_uvs(out_dir / "flatten.obj", len(faces))


def _read_uvs(path, face_count):
    uvs = []
    face_uvs = []
    with Path(path).open() as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "vt":
                uvs.append((float(parts[1]), float(parts[2])))
            elif parts[0] == "f":
                face_uvs.append(
                    [uvs[int(token.split("/")[1]) - 1] for token in parts[1:]]
                )
    if len(face_uvs) != face_count:
        raise FlattenError(
            f"flatten output has {len(face_uvs)} faces, expected {face_count}"
        )
    return face_uvs


def polygon_area(verts, face):
    """3d area by fan, the same fan the flatten solves."""
    x0, y0, z0 = verts[face[0]]
    total = 0.0
    for i in range(1, len(face) - 1):
        ax, ay, az = verts[face[i]]
        bx, by, bz = verts[face[i + 1]]
        ux, uy, uz = ax - x0, ay - y0, az - z0
        vx, vy, vz = bx - x0, by - y0, bz - z0
        cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        total += math.sqrt(cx * cx + cy * cy + cz * cz) / 2.0
    return total


def submesh(verts, faces, subset, seams):
    """Compact copy of just these faces, with the seams reindexed."""
    vmap = {}
    sub_verts = []
    sub_faces = []
    for f in subset:
        face = []
        for v in faces[f]:
            idx = vmap.get(v)
            if idx is None:
                idx = vmap[v] = len(sub_verts)
                sub_verts.append(verts[v])
            face.append(idx)
        sub_faces.append(tuple(face))
    sub_seams = set()
    for a, b in seams:
        ma, mb = vmap.get(a), vmap.get(b)
        if ma is not None and mb is not None:
            sub_seams.add((ma, mb) if ma < mb else (mb, ma))
    return sub_verts, sub_faces, sub_seams


def uv_density(verts, faces, uvs, subset):
    """Uv area per unit surface area, the map's texel density."""
    total_uv = sum(abs(signed_area(uvs[f])) for f in subset)
    total_3d = sum(polygon_area(verts, faces[f]) for f in subset)
    return total_uv / total_3d if total_3d else 1.0


def island_center(uvs, group):
    points = [uv for f in group for uv in uvs[f]]
    xs = [u for u, _ in points]
    ys = [v for _, v in points]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def restore_island(verts, faces, uvs, group, density, center):
    """Put a freshly unwrapped island back at the map's scale and position.

    A subset flatten packs just the subset into the unit square, so the
    island comes back oversized somewhere else. Scaling it to the map's texel
    density instead of repacking the atlas keeps every measure that reads uv
    lengths against the atlas meaningful, and costs nothing. Islands may
    overlap until the single pack at the end, which is harmless because
    every test in the loop reads one island at a time."""
    area_uv = sum(abs(signed_area(uvs[f])) for f in group)
    area_3d = sum(polygon_area(verts, faces[f]) for f in group)
    if area_uv <= 0 or area_3d <= 0:
        return
    scale = math.sqrt(density * area_3d / area_uv)
    points = [uv for f in group for uv in uvs[f]]
    xs = [u for u, _ in points]
    ys = [v for _, v in points]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    for f in group:
        uvs[f] = [
            (center[0] + (u - cx) * scale, center[1] + (v - cy) * scale)
            for u, v in uvs[f]
        ]


def preseed_uvs(
    engine,
    verts,
    faces,
    angle=CREASE_ANGLE,
    marked="NONE",
    weights=None,
    only=None,
    marked_seams=frozenset(),
):
    """Seam the strip-merged feature boundaries, flatten, repair, pack.

    The data twin of hard_surface.build_seam_uvs: engine is a FlattenEngine
    (or anything with its flatten/pack signature), marked/weights/only mean
    what they do there, marked_seams are the mesh's own marked edges as
    vertex pairs. Returns (seams, uvs) where uvs is per-face corner lists,
    None for faces outside only."""
    if marked == "ONLY":
        seams = set(marked_seams)
    else:
        forced = set(marked_seams) if marked == "ADD" else None
        detect = faces if only is None else [faces[i] for i in sorted(only)]
        seams = seam_edges(verts, detect, angle, weights=weights, forced=forced)

    subset = list(range(len(faces))) if only is None else sorted(only)
    all_uvs = [None] * len(faces)

    def flatten_into(target_faces, iterations=10):
        sub_verts, sub_faces, sub_seams = submesh(verts, faces, target_faces, seams)
        for f, face_uv in zip(
            target_faces, engine.flatten(sub_verts, sub_faces, sub_seams, iterations)
        ):
            all_uvs[f] = face_uv

    flatten_into(subset)
    edges = face_edges(faces)
    density = uv_density(verts, faces, all_uvs, subset)
    repaired = False

    def redo_islands(targets, iterations=10):
        """Flatten these islands again, each landing back at its own scale."""
        centers = [island_center(all_uvs, group) for group in targets]
        flatten_into(sorted(f for group in targets for f in group), iterations)
        for group, center in zip(targets, centers):
            restore_island(verts, faces, all_uvs, group, density, center)

    # islands untouched by a round can't have changed, so later rounds only
    # rescan what was redone or cut; None means the first round checks all
    dirty = None
    for _ in range(REPAIR_ROUNDS):
        groups = island_groups(faces, seams, edges)
        if only is not None:
            # loose parts are disjoint, so an island is entirely in or out
            groups = [g for g in groups if g[0] in only]
        if dirty is not None:
            groups = [g for g in groups if dirty & set(g)]
        ruined = [g for g in groups if island_ruined(g, faces, all_uvs, edges, seams)]
        if ruined:
            repaired = True
            redo_islands(ruined, REPAIR_ITERATIONS)

        # cuts ruined islands, and clean strips too long to pack
        extra = split_islands(verts, faces, seams, all_uvs, weights, groups)
        if not extra:
            break
        repaired = True
        seams |= extra
        touched = {f for e in extra for f in edges[e]}
        # the split pieces of one old island go back where it was, together
        redo_islands([g for g in groups if touched & set(g)])
        dirty = touched | {f for g in ruined for f in g}

    if repaired:
        sub_verts, sub_faces, _ = submesh(verts, faces, subset, seams)
        for f, face_uv in zip(
            subset, engine.pack(sub_verts, sub_faces, [all_uvs[f] for f in subset])
        ):
            all_uvs[f] = face_uv
    return seams, all_uvs
