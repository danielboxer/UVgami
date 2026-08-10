"""Preseed uvs without Blender.

The flatten solve runs in the optcuts binary's flatten mode, everything here
is plain data: verts, polygon faces, per-face corner uvs. hard_surface.py
adapts a Blender mesh onto these calls, so this module stays unit-testable
and can run off the main thread."""

import shutil
import subprocess
import tempfile
from pathlib import Path

from .mesh import face_edges
from .pipeline import seam_edges
from .regions import CREASE_ANGLE
from .symmetry import mirror_seams


class FlattenError(RuntimeError):
    pass


def check_manifold(faces):
    """Guard for flattens whose result ships as the final map. The engine
    doesn't validate in flatten mode: a non-manifold mesh comes back as
    degenerate uvs with exit 0. The unwrap path skips this on purpose, a
    ruined island there is recut by the engine."""
    if any(len(owners) > 2 for owners in face_edges(faces).values()):
        raise FlattenError("Non Manifold Edges")


class FlattenEngine:
    """Client for the engine's flatten mode. Each call gets its own subdir of
    workdir: the preview operator, a builder thread and transfer_cuts can all
    flatten at once, and shared filenames would swap uvs between them."""

    def __init__(self, engine_path, workdir):
        self.engine_path = str(engine_path)
        self.workdir = Path(workdir)

    def flatten(self, verts, faces, seams):
        """Per-face corner uvs for faces cut along seams, packed."""
        return self._run(verts, faces, seams)

    def _run(self, verts, faces, seams):
        self.workdir.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(dir=self.workdir))
        try:
            return self._run_in(workdir, verts, faces, seams)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _run_in(self, workdir, verts, faces, seams):
        obj_path = workdir / "flatten.obj"
        seam_path = workdir / "flatten_seams"
        out_dir = workdir / "flatten_out"

        with obj_path.open("w") as f:
            for x, y, z in verts:
                f.write(f"v {x} {y} {z}\n")
            for face in faces:
                f.write("f " + " ".join(str(v + 1) for v in face) + "\n")

        if seams:
            seam_path.write_text("".join(f"{a} {b}\n" for a, b in sorted(seams)))

        args = [self.engine_path, "-i", str(obj_path), "-o", str(out_dir), "-flatten"]
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


def preseed_uvs(
    engine,
    verts,
    faces,
    angle=CREASE_ANGLE,
    marked="NONE",
    weights=None,
    only=None,
    marked_seams=frozenset(),
    mirrors=None,
):
    """Seam the strip-merged feature boundaries and flatten.

    The data twin of hard_surface.build_seam_uvs: engine is a FlattenEngine
    (or anything with its flatten signature), marked/weights/only mean
    what they do there, marked_seams are the mesh's own marked edges as
    vertex pairs, mirrors are per-axis vertex maps the seam set is closed
    under, so a symmetric mesh flattens into mirrored islands with no cut at
    the plane. Returns (seams, uvs) where uvs is per-face corner lists,
    None for faces outside only. Returns None when the subset is closed and
    the seam set came out empty, which cannot flatten: the caller falls back
    to a scratch unwrap.

    A ruined island ships as-is on purpose: the engine rejects it and its
    own cut search replaces it, which benches better than repairing here."""
    subset = list(range(len(faces))) if only is None else sorted(only)
    in_subset = set(subset)
    edges = face_edges(faces)
    if marked == "ONLY":
        seams = set(marked_seams)
    else:
        forced = set(marked_seams) if marked == "ADD" else None
        detect = faces if only is None else [faces[i] for i in subset]
        seams = seam_edges(verts, detect, angle, weights=weights, forced=forced)
    if mirrors:
        allowed = edges if only is None else face_edges([faces[i] for i in subset])
        seams = mirror_seams(seams, mirrors, allowed)
    if not seams and all(
        len(owners) != 1 for owners in edges.values() if owners[0] in in_subset
    ):
        return None

    all_uvs = [None] * len(faces)
    sub_verts, sub_faces, sub_seams = submesh(verts, faces, subset, seams)
    for f, face_uv in zip(subset, engine.flatten(sub_verts, sub_faces, sub_seams)):
        all_uvs[f] = face_uv
    return seams, all_uvs
