import os
import time

from .cancel import Cancelled, check_cancelled
from .mesh import vertex_components
from .pipeline import WholeMesh, seam_edges, seams_at_angle, whole_mesh_inputs
from .regions import CREASE_ANGLE
from .worker import WorkerError, WorkerProcess

POLL_INTERVAL = 0.05


# largest first onto the lightest, so the workers carry about the same face count
def balanced_chunks(parts, count):
    loads = [0] * count
    chunks = [[] for _ in range(count)]
    for part in sorted(parts, key=len, reverse=True):
        lightest = loads.index(min(loads))
        loads[lightest] += len(part)
        chunks[lightest].append(part)
    return [chunk for chunk in chunks if chunk]


# what one worker runs
def chunk_seams(verts, chunk, job):
    whole = WholeMesh(job.pop("min_width"), job.pop("model_area"), None)
    mesh_ids = job.pop("face_ids")
    results = []
    for ids, faces in chunk:
        if mesh_ids is not None:
            ids = [mesh_ids[i] for i in ids]
        results.append(
            seams_at_angle(
                verts, faces, cancelled=None, whole=whole._replace(face_ids=ids), **job
            )
        )
    return results


# the parts keep the mesh's vertex ids, reindexing would move the tie breaks
def seam_edges_parallel(
    verts,
    faces,
    angle=CREASE_ANGLE,
    rims=True,
    weights=None,
    forced=None,
    cancelled=None,
    whole=None,
):
    parts = vertex_components(faces)
    count = min(os.cpu_count() or 1, len(parts))
    if count < 2:
        return seam_edges(verts, faces, angle, rims, weights, forced, cancelled, whole)
    if whole is None:
        min_width, model_area = whole_mesh_inputs(verts, faces, rims, forced)
    else:
        min_width, model_area = whole.min_width, whole.model_area
    check_cancelled(cancelled)
    job = {
        "angle": angle,
        "rims": rims,
        "weights": weights,
        "forced": forced,
        "min_width": min_width,
        "model_area": model_area,
        "face_ids": None if whole is None else whole.face_ids,
    }
    chunks = [
        [(part, [faces[i] for i in part]) for part in chunk]
        for chunk in balanced_chunks(parts, count)
    ]
    workers = [WorkerProcess(chunk_seams, verts, chunk, job) for chunk in chunks]
    seams = set()
    closed = True
    try:
        for worker in workers:
            while not worker.done():
                if cancelled is not None and cancelled():
                    raise Cancelled
                time.sleep(POLL_INTERVAL)
            try:
                results = worker.result()
            except WorkerError as error:
                raise RuntimeError(f"seam worker failed: {error}") from error
            for part_seams, part_closed in results:
                seams |= part_seams
                closed = closed and part_closed
    finally:
        for worker in workers:
            worker.close()
    if not seams and angle > CREASE_ANGLE and closed:
        return seam_edges_parallel(
            verts, faces, CREASE_ANGLE, rims, weights, forced, cancelled, whole
        )
    return seams
