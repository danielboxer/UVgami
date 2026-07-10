import numpy as np
import trimesh
from scipy.spatial import cKDTree

def get_submesh_face_indices_by_proximity(
    mesh_orig: trimesh.Trimesh,
    mesh_sub: trimesh.Trimesh,
    tol: float = 1e-6
) -> np.ndarray:
    """
    Find, for each face of mesh_sub, the corresponding face index in mesh_orig by vertex proximity.

    Parameters
    ----------
    mesh_orig : trimesh.Trimesh
        The full original mesh.
    mesh_sub : trimesh.Trimesh
        A submesh of mesh_orig (possibly extracted via trimesh.submesh or loaded separately).
    tol : float
        Maximum allowed distance for snapping submesh vertices to original vertices.

    Returns
    -------
    face_indices : (M,) int array
        Indices into mesh_orig.faces such that mesh_orig.faces[face_indices[i]]
        corresponds to mesh_sub.faces[i].
    """
    # Build a KD‐tree on the original mesh’s vertices
    kdt = cKDTree(mesh_orig.vertices)

    # For each submesh vertex, find nearest original‐mesh vertex
    dists, orig_vidx = kdt.query(mesh_sub.vertices)
    if np.any(dists > tol):
        bad = np.where(dists > tol)[0]
        raise ValueError(
            f"{len(bad)} submesh vertices farther than {tol} from any original vertex."
        )

    # Build a lookup: sorted‐tuple of vertex‐indices → original face index
    face_map = {
        tuple(sorted(face)): idx
        for idx, face in enumerate(mesh_orig.faces)
    }

    # For each submesh face, map its 3 vertices → original vertex‐indices, then find face
    result = []
    for face in mesh_sub.faces:
        global_face = tuple(sorted(orig_vidx[face]))
        if global_face not in face_map:
            raise ValueError(f"Submesh face {face} (global {global_face}) not found.")
        result.append(face_map[global_face])

    return np.array(result, dtype=int)


if __name__ == "__main__":
    # Example: load your meshes from files
    mesh_orig = trimesh.load("/ariesdv0/zhaoning/workspace/IUV/lscm/libigl-example-project/meshes/stock_merged.obj", process=False)
    mesh_sub  = trimesh.load("/ariesdv0/zhaoning/workspace/IUV/lscm/libigl-example-project/meshes/stock_side.obj", process=False)

    sub_face_idxs = get_submesh_face_indices_by_proximity(mesh_orig, mesh_sub, tol=1e-6)
    print("Submesh faces correspond to original mesh face indices:")
    # Print the first few indices for verification
    print(sub_face_idxs[:10])
    print(mesh_sub.vertices.shape)
    
    # Save the face indices to a text file for loading in C++
    output_file = "submesh_face_indices.txt"
    np.savetxt(output_file, sub_face_idxs, fmt='%d')
    print(f"Face indices saved to {output_file}")
