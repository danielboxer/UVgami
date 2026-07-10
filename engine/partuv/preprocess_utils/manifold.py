#handles non-manifold meshes

import argparse
from collections import defaultdict
import numpy as np
import trimesh
from collections import defaultdict
import numpy as np
import trimesh
from tqdm import tqdm


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        # path compression
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def split_nonmanifold_edges_as_seams(mesh: trimesh.Trimesh):
    """
    Cut all non-2-manifold edges as seams by duplicating vertices per
    connected corner-group (like UV seams), not per-face.
    """
    mesh = mesh.copy()
    V = mesh.vertices.copy()
    F = mesh.faces.copy()  # IMPORTANT: copy since we will remap

    # edge -> incident faces
    edge_to_faces = defaultdict(list)
    for fi, tri in enumerate(F):
        a, b, c = map(int, tri)
        edge_to_faces[tuple(sorted((a, b)))].append(fi)
        edge_to_faces[tuple(sorted((b, c)))].append(fi)
        edge_to_faces[tuple(sorted((c, a)))].append(fi)

    nonmanifold_edges_count = sum(1 for _, faces in edge_to_faces.items() if len(faces) > 2)

    # For each face, map vertex index -> local corner id (0/1/2)
    # triangles => each vertex appears once, so dict lookup is fine.
    face_v2li = []
    for tri in F:
        tri = list(map(int, tri))
        face_v2li.append({tri[0]: 0, tri[1]: 1, tri[2]: 2})

    # Union-Find over all corners (3 per face)
    uf = UnionFind(3 * len(F))

    # Glue corners ONLY across manifold edges (exactly 2 incident faces).
    # Non-manifold edges (>=3 faces) are seams: no glue across them.
    print("Fixing non-manifold edges as seams...")
    for (u, v), faces in edge_to_faces.items():
        if len(faces) != 2:
            continue  # boundary or non-manifold => seam / no glue
        f0, f1 = faces

        # corner ids for u
        c0u = 3 * f0 + face_v2li[f0][u]
        c1u = 3 * f1 + face_v2li[f1][u]
        uf.union(c0u, c1u)

        # corner ids for v
        c0v = 3 * f0 + face_v2li[f0][v]
        c1v = 3 * f1 + face_v2li[f1][v]
        uf.union(c0v, c1v)

    # Assign a new vertex per connected corner-group
    root_to_newv = {}
    new_vertices = []
    new_faces = np.empty_like(F)

    for fi, tri in enumerate(F):
        for li, vid in enumerate(map(int, tri)):
            cid = 3 * fi + li
            root = uf.find(cid)
            new_vid = root_to_newv.get(root)
            if new_vid is None:
                new_vid = len(new_vertices)
                root_to_newv[root] = new_vid
                new_vertices.append(V[vid])
            new_faces[fi, li] = new_vid

    fixed = trimesh.Trimesh(
        vertices=np.asarray(new_vertices),
        faces=new_faces,
        process=False
    )
    return fixed, nonmanifold_edges_count


def split_nonmanifold_edges(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Return a copy of *mesh* in which every edge is incident
    to at most two faces (2-manifold).
    """
    mesh = mesh.copy()
    V_orig = mesh.vertices.copy()
    F = mesh.faces

    # Build a map: sorted edge -> list[face indices] that contain it
    edge_to_faces = defaultdict(list)
    for fi, tri in enumerate(F):
        for a, b in ((tri[0], tri[1]),
                     (tri[1], tri[2]),
                     (tri[2], tri[0])):
            edge_to_faces[tuple(sorted((a, b)))].append(fi)

    vertices = V_orig.tolist()          # mutable growing list
    non_manifold_edges_count = 0
    for edge, faces in edge_to_faces.items():
        if len(faces) <= 2:
            continue                    # already manifold
        
        non_manifold_edges_count += 1
        # print(f"edge is not manifold: {edge}, faces: {faces}")
        v1, v2 = edge
        # Keep the first two faces untouched; fix the rest
        for fi in faces[2:]:
            # duplicate the two vertices
            v1_new = len(vertices)
            vertices.append(V_orig[v1])
            v2_new = len(vertices)
            vertices.append(V_orig[v2])

            face = F[fi].copy()
            # Replace occurrences of the old indices with the new ones
            face = np.where(face == v1, v1_new, face)
            face = np.where(face == v2, v2_new, face)
            F[fi] = face

    return trimesh.Trimesh(vertices=np.asarray(vertices),
                           faces=F,
                           process=False), non_manifold_edges_count   # keep indices as-is

def fix_mesh(mesh_path: str):
    mesh = trimesh.load(mesh_path, process=False)
    fixed, non_manifold_edges_count = split_nonmanifold_edges_as_seams(mesh)
    if non_manifold_edges_count > 0:
        fixed.export(mesh_path)



def fix_mesh_trimesh(mesh: trimesh.Trimesh):
    fixed, non_manifold_edges_count = split_nonmanifold_edges_as_seams(mesh)
    if non_manifold_edges_count > 0:
        return fixed
    return mesh
    
def main():
    mesh_path = "/home/wzn/workspace/partuv/00000017_e8e79d1385fd4c78be414f6c_trimesh_000_pum_ori.obj"

    # mesh = trimesh.load(mesh_path, process=False)
    mesh = load_mesh_and_merge(mesh_path, epsilon=None)
    mesh.export(mesh_path.rsplit(".", 1)[0] + "_loaded.obj")
    fixed, non_manifold_edges_count = split_nonmanifold_edges_as_seams(mesh)

    print(f"Non-manifold edges count: {non_manifold_edges_count}")

    out_path = (mesh_path.rsplit(".", 1)[0] + "_fixed2.obj")
    
    fixed.export(out_path)
    print(f"Non-manifold edges split.  Saved to: {out_path}")


if __name__ == "__main__":
    main()
