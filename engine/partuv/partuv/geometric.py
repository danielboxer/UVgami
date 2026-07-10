"""Checkpoint-free segmentation: geometric features instead of PartField.

Produces the same face merge tree the pipeline expects, using ward
agglomerative clustering on face normals and centroids over the face
adjacency graph. No torch, no model download; needs only the base deps.
"""

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering

from .preprocess_utils.manifold import fix_mesh_trimesh
from .preprocess_utils.merge_V_obj import load_mesh_and_merge


def face_adjacency_matrix(mesh: trimesh.Trimesh) -> csr_matrix:
    """Symmetric csr adjacency over faces, with dummy edges joining
    disconnected components so the clustering yields one tree."""
    num_faces = len(mesh.faces)
    pairs = mesh.face_adjacency
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data = np.ones(len(rows), dtype=np.int8)
    adjacency = csr_matrix((data, (rows, cols)), shape=(num_faces, num_faces))

    count, labels = connected_components(adjacency, directed=False)
    if count > 1:
        first_faces = [int(np.argmax(labels == c)) for c in range(count)]
        extra = np.array(
            [(first_faces[i], first_faces[i + 1]) for i in range(count - 1)]
        )
        rows = np.concatenate([rows, extra[:, 0], extra[:, 1]])
        cols = np.concatenate([cols, extra[:, 1], extra[:, 0]])
        data = np.ones(len(rows), dtype=np.int8)
        adjacency = csr_matrix((data, (rows, cols)), shape=(num_faces, num_faces))
    return adjacency


def build_face_tree(mesh: trimesh.Trimesh, centroid_weight: float = 0.3) -> dict:
    """Merge tree in the PartField format: {internal_id: {left, right}} with
    faces as leaves 0..F-1 and internal nodes numbered from F."""
    centroids = mesh.triangles_center
    extent = max(float(np.ptp(centroids, axis=0).max()), 1e-12)
    centroids = (centroids - centroids.min(axis=0)) / extent
    features = np.hstack([mesh.face_normals, centroid_weight * centroids])

    clustering = AgglomerativeClustering(
        connectivity=face_adjacency_matrix(mesh), n_clusters=1
    ).fit(features)

    num_faces = len(mesh.faces)
    return {
        num_faces + i: {"left": int(left), "right": int(right)}
        for i, (left, right) in enumerate(clustering.children_)
    }


def preprocess_geometric(
    mesh_path, centroid_weight: float = 0.3, merge_vertices_epsilon: float = 1e-7
):
    """Torch-free counterpart of preprocess(): returns (mesh, tree_dict)."""
    mesh = load_mesh_and_merge(mesh_path, epsilon=merge_vertices_epsilon)
    mesh = fix_mesh_trimesh(mesh)
    return mesh, build_face_tree(mesh, centroid_weight)
