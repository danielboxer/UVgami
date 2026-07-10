from sklearn.cluster import AgglomerativeClustering
from scipy.sparse.csgraph import connected_components
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
import os
import joblib
import itertools
import time
from collections import defaultdict
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

import numpy as np
import trimesh

import trimesh
from scipy.spatial import KDTree
import networkx as nx
import numpy as np
from tqdm import tqdm
#   import line_profiler
import torch

def pca_gpu(X, num_components=2, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    # Move data to GPU if available
    X = X.to(device).to(torch.float32)

    # Step 1: Mean-center the data
    X_mean = torch.mean(X, dim=0)
    X_centered = X - X_mean
    # Step 2: Compute covariance matrix
    cov_matrix = torch.matmul(X_centered.T, X_centered) / (X.shape[0] - 1)
    # Step 3: Eigen decomposition
    eigvals, eigvecs = torch.linalg.eigh(cov_matrix)
    # Step 4: Sort by descending eigenvalues
    sorted_indices = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]
    # Step 5: Project data onto top-k eigenvectors
    components = eigvecs[:, :num_components]
    X_reduced = torch.matmul(X_centered, components)
    return X_reduced, eigvals[:num_components], components



def find_original_vertex_indices(original_mesh, component_mesh):
    """
    Finds the original vertex indices of a component mesh within the original mesh.

    Parameters:
    - original_mesh (trimesh.Trimesh): The original mesh.
    - component_mesh (trimesh.Trimesh): The component mesh.

    Returns:
    - np.ndarray: Array of original vertex indices corresponding to the component's vertices.
                  If a vertex is not found, its index will be -1.
    """
    # Ensure vertices are numpy arrays
    original_vertices = original_mesh.vertices
    component_vertices = component_mesh.vertices

    # Step 1: Create a structured view for both meshes
    dtype = np.dtype((np.void, original_vertices.dtype.itemsize * original_vertices.shape[1]))
    original_view = original_vertices.view(dtype).reshape(-1)
    component_view = component_vertices.view(dtype).reshape(-1)

    # Step 2: Sort the original vertices
    sorted_indices = np.argsort(original_view)
    sorted_original_view = original_view[sorted_indices]

    # Step 3: Use searchsorted to find indices
    search_indices = np.searchsorted(sorted_original_view, component_view)

    # Step 4: Initialize the result array with -1 (not found)
    original_indices = np.full(len(component_vertices), -1, dtype=int)

    # Step 5: Create a mask for valid indices
    mask = (search_indices < len(sorted_original_view)) & \
           (sorted_original_view[search_indices] == component_view)

    # Step 6: Assign the original indices where matches are found
    original_indices[mask] = sorted_indices[search_indices[mask]]

    return original_indices


def construct_adjacency_matrix(face_list, num_vertices):
    row = []
    col = []

    for face in face_list:
        v0, v1, v2 = face
        row.extend([v0, v1, v1, v2, v2, v0])
        col.extend([v1, v0, v2, v1, v0, v2])
    
    data = np.ones(len(row), dtype=int)
    adjacency_matrix = coo_matrix((data, (row, col)), shape=(num_vertices, num_vertices))
    adjacency_matrix = adjacency_matrix.tocsr()
    return adjacency_matrix


def construct_face_adjacency_matrix_naive(face_list):
    """
    Given a list of faces (each face is a 3-tuple of vertex indices),
    construct a face-based adjacency matrix of shape (num_faces, num_faces).
    Two faces are adjacent if they share an edge.

    If multiple connected components exist, dummy edges are added to 
    turn them into a single connected component. Edges are added naively by
    randomly selecting a face and connecting consecutive components -- (comp_i, comp_i+1) ...

    Parameters
    ----------
    face_list : list of tuples
        List of faces, each face is a tuple (v0, v1, v2) of vertex indices.

    Returns
    -------
    face_adjacency : scipy.sparse.csr_matrix
        A CSR sparse matrix of shape (num_faces, num_faces), 
        containing 1s for adjacent faces and 0s otherwise. 
        Additional edges are added if the faces are in multiple components.
    """

    num_faces = len(face_list)
    if num_faces == 0:
        # Return an empty matrix if no faces
        return csr_matrix((0, 0))

    # Step 1: Map each undirected edge -> list of face indices that contain that edge
    edge_to_faces = defaultdict(list)

    # Populate the edge_to_faces dictionary
    for f_idx, (v0, v1, v2) in enumerate(face_list):
        # For an edge, we always store its endpoints in sorted order
        # to avoid duplication (e.g. edge (2,5) is the same as (5,2)).
        edges = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0)))
        ]
        for e in edges:
            edge_to_faces[e].append(f_idx)

    # Step 2: Build the adjacency (row, col) lists among faces
    row = []
    col = []
    for e, faces_sharing_e in edge_to_faces.items():
        # If an edge is shared by multiple faces, make each pair of those faces adjacent
        f_indices = list(set(faces_sharing_e))  # unique face indices for this edge
        if len(f_indices) > 1:
            # For each pair of faces, mark them as adjacent
            for i in range(len(f_indices)):
                for j in range(i + 1, len(f_indices)):
                    f_i = f_indices[i]
                    f_j = f_indices[j]
                    row.append(f_i)
                    col.append(f_j)
                    row.append(f_j)
                    col.append(f_i)

    # Create a COO matrix, then convert it to CSR
    data = np.ones(len(row), dtype=np.int8)
    face_adjacency = coo_matrix(
        (data, (row, col)),
        shape=(num_faces, num_faces)
    ).tocsr()

    # Step 3: Ensure single connected component
    # Use connected_components to see how many components exist
    n_components, labels = connected_components(face_adjacency, directed=False)

    if n_components > 1:
        # We have multiple components; let's "connect" them via dummy edges
        # The simplest approach is to pick one face from each component
        # and connect them sequentially to enforce a single component.
        component_representatives = []

        for comp_id in range(n_components):
            # indices of faces in this component
            faces_in_comp = np.where(labels == comp_id)[0]
            if len(faces_in_comp) > 0:
                # take the first face in this component as a representative
                component_representatives.append(faces_in_comp[0])

        # Now, add edges between consecutive representatives
        dummy_row = []
        dummy_col = []
        for i in range(len(component_representatives) - 1):
            f_i = component_representatives[i]
            f_j = component_representatives[i + 1]
            dummy_row.extend([f_i, f_j])
            dummy_col.extend([f_j, f_i])

        if dummy_row:
            dummy_data = np.ones(len(dummy_row), dtype=np.int8)
            dummy_mat = coo_matrix(
                (dummy_data, (dummy_row, dummy_col)),
                shape=(num_faces, num_faces)
            ).tocsr()
            face_adjacency = face_adjacency + dummy_mat

    return face_adjacency

def construct_face_adjacency_matrix(face_list, vertices, k=10):
    """
    Given a list of faces (each face is a 3-tuple of vertex indices),
    construct a face-based adjacency matrix of shape (num_faces, num_faces).

    Two faces are adjacent if they share an edge (the "mesh adjacency").
    If multiple connected components remain, we:
      1) Compute the centroid of each face.
      2) Use a KNN graph (k=10) based on centroid distances.
      3) Compute MST of that KNN graph.
      4) Add MST edges that connect different components as "dummy" edges
         in the face adjacency matrix, ensuring one connected component.

    Parameters
    ----------
    face_list : list of tuples
        List of faces, each face is a tuple (v0, v1, v2) of vertex indices.
    vertices : np.ndarray of shape (num_vertices, 3)
        Array of vertex coordinates.
    k : int, optional
        Number of neighbors to use in centroid KNN. Default is 10.

    Returns
    -------
    face_adjacency : scipy.sparse.csr_matrix
        A CSR sparse matrix of shape (num_faces, num_faces),
        containing 1s for adjacent faces (shared-edge adjacency)
        plus dummy edges ensuring a single connected component.
    """
    num_faces = len(face_list)
    if num_faces == 0:
        # Return an empty matrix if no faces
        return csr_matrix((0, 0))

    #--------------------------------------------------------------------------
    # 1) Build adjacency based on shared edges.
    #    (Same logic as the original code, plus import statements.)
    #--------------------------------------------------------------------------
    edge_to_faces = defaultdict(list)
    uf = UnionFind(num_faces)
    for f_idx, (v0, v1, v2) in enumerate(face_list):
        # Sort each edge’s endpoints so (i, j) == (j, i)
        edges = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0)))
        ]
        for e in edges:
            edge_to_faces[e].append(f_idx)

    row = []
    col = []
    for edge, face_indices in edge_to_faces.items():
        unique_faces = list(set(face_indices))
        if len(unique_faces) > 1:
            # For every pair of distinct faces that share this edge,
            # mark them as mutually adjacent
            for i in range(len(unique_faces)):
                for j in range(i + 1, len(unique_faces)):
                    fi = unique_faces[i]
                    fj = unique_faces[j]
                    row.append(fi)
                    col.append(fj)
                    row.append(fj)
                    col.append(fi)
                    uf.union(fi, fj)

    data = np.ones(len(row), dtype=np.int8)
    face_adjacency = coo_matrix(
        (data, (row, col)), shape=(num_faces, num_faces)
    ).tocsr()

    #--------------------------------------------------------------------------
    # 2) Check if the graph from shared edges is already connected.
    #--------------------------------------------------------------------------
    n_components = 0
    for i in range(num_faces):
        if uf.find(i) == i:
            n_components += 1
    print("n_components", n_components)

    if n_components == 1:
        # Already a single connected component, no need for dummy edges
        return face_adjacency
    #--------------------------------------------------------------------------
    # 3) Compute centroids of each face for building a KNN graph.
    #--------------------------------------------------------------------------
    face_centroids = []
    for (v0, v1, v2) in face_list:
        centroid = (vertices[v0] + vertices[v1] + vertices[v2]) / 3.0
        face_centroids.append(centroid)
    face_centroids = np.array(face_centroids)

    #--------------------------------------------------------------------------
    # 4) Build a KNN graph (k=10) over face centroids using scikit‐learn
    #--------------------------------------------------------------------------
    knn = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn.fit(face_centroids)
    distances, indices = knn.kneighbors(face_centroids)
    # 'distances[i]' are the distances from face i to each of its 'k' neighbors
    # 'indices[i]' are the face indices of those neighbors

    #--------------------------------------------------------------------------
    # 5) Build a weighted graph in NetworkX using centroid-distances as edges
    #--------------------------------------------------------------------------
    G = nx.Graph()
    # Add each face as a node in the graph
    G.add_nodes_from(range(num_faces))

    # For each face i, add edges (i -> j) for each neighbor j in the KNN
    for i in range(num_faces):
        for j, dist in zip(indices[i], distances[i]):
            if i == j:
                continue  # skip self-loop
            # Add an undirected edge with 'weight' = distance
            # NetworkX handles parallel edges gracefully via last add_edge,
            # but it typically overwrites the weight if (i, j) already exists.
            G.add_edge(i, j, weight=dist)

    #--------------------------------------------------------------------------
    # 6) Compute MST on that KNN graph
    #--------------------------------------------------------------------------
    mst = nx.minimum_spanning_tree(G, weight='weight')
    # Sort MST edges by ascending weight, so we add the shortest edges first
    mst_edges_sorted = sorted(
        mst.edges(data=True), key=lambda e: e[2]['weight']
    )
    print("mst edges sorted", len(mst_edges_sorted))
    #--------------------------------------------------------------------------
    # 7) Use a union-find structure to add MST edges only if they
    #    connect two currently disconnected components of the adjacency matrix
    #--------------------------------------------------------------------------

    # Convert face_adjacency to LIL format for efficient edge addition
    adjacency_lil = face_adjacency.tolil()

    # Now, step through MST edges in ascending order
    for (u, v, attr) in mst_edges_sorted:
        if uf.find(u) != uf.find(v):
            # These belong to different components, so unify them
            uf.union(u, v)
            # And add a "dummy" edge to our adjacency matrix
            adjacency_lil[u, v] = 1
            adjacency_lil[v, u] = 1

    # Convert back to CSR format and return
    face_adjacency = adjacency_lil.tocsr()
    return face_adjacency


def construct_adjacency_matrix_face(face_list, num_faces):
    edge_to_faces = {}
    for f_idx, face in enumerate(face_list):
        v0, v1, v2 = face
        edges = [
            (min(v0, v1), max(v0, v1)),
            (min(v1, v2), max(v1, v2)),
            (min(v2, v0), max(v2, v0))
        ]
        for edge in edges:
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(f_idx)
    
    row = []
    col = []
    for edge, faces in edge_to_faces.items():
        if len(faces) > 1:
            for i in range(len(faces)):
                for j in range(i+1, len(faces)):
                    f1 = faces[i]
                    f2 = faces[j]
                    
                    
                    
                    row.extend([f1, f2])
                    col.extend([f2, f1])
    data = np.ones(len(row), dtype=int)
    adjacency_matrix = coo_matrix((data, (row, col)), shape=(num_faces, num_faces))
    adjacency_matrix = adjacency_matrix.tocsr()
    return adjacency_matrix




def normal_diff_matrix_spatial_sparse(mesh, radius=0.05):
    """
    Constructs a sparse matrix (shape: n_faces x n_faces) whose entries are the
    normal difference (clamped dot product) between faces whose centroids lie
    within the specified radius.

    normal_diff = max(dot(normal_i, normal_j), 0)

    For face pairs outside the search radius, the matrix entry is 0.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        A Trimesh object with triangular faces.
    radius : float, optional
        Maximum distance between face centroids for them to be considered neighbors.
        Default is 0.05.

    Returns
    -------
    normal_diff_sp : scipy.sparse.csr_matrix of shape (n_faces, n_faces), dtype=float
        Sparse matrix where entry (i, j) contains the clamped dot product between
        face i's normal and face j's normal, if the centroids of faces i and j
        lie within 'radius'. Otherwise, 0.
    """
    n_faces = len(mesh.faces)
    face_centroids = mesh.triangles_center  # (n_faces, 3)
    face_normals = mesh.face_normals        # (n_faces, 3)


    tree = KDTree(face_centroids)

    rows, cols, data = [], [], []
    for i in tqdm(range(n_faces)):
        centroid_i = face_centroids[i]
        normal_i = face_normals[i]
        

        neighbors = tree.query_ball_point(centroid_i, r=radius)
        

        for j in neighbors:
            if j == i:

                continue

            dot_ij = np.dot(normal_i, face_normals[j])
            if dot_ij > 0.0:
                rows.append(i)
                cols.append(j)
                # Store the clamped value
                data.append(dot_ij)

    normal_diff_sp = coo_matrix(
        (data, (rows, cols)),
        shape=(n_faces, n_faces),
        dtype=float
    ).tocsr()

    return normal_diff_sp



def construct_adjacency_matrix_face_with_normal( face_list, num_faces, mesh = None, point_feat=None):
    edge_to_faces = {}
    for f_idx, face in enumerate(face_list):
        v0, v1, v2 = face
        edges = [
            (min(v0, v1), max(v0, v1)),
            (min(v1, v2), max(v1, v2)),
            (min(v2, v0), max(v2, v0))
        ]
        for edge in edges:
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(f_idx)
    
    row = []
    col = []
    normals = []
    
    row_normal = []
    col_normal = []
    
    if point_feat is not None:  # num_faces x 448
        # allocate np list to num_faces x 1 x 448 and expand to num_faces x 4 x 448 with adjancency, then average
        new_point_feat = [feat for feat in point_feat]
    
    for edge, faces in edge_to_faces.items():
        if len(faces) > 1:
            for i in range(len(faces)):
                for j in range(i+1, len(faces)):
                    f1 = faces[i]
                    f2 = faces[j]
                    
                    if mesh is not None:
                        normal1 = mesh.face_normals[f1]
                        normal2 = mesh.face_normals[f2]
                        normal_diff = max(np.dot(normal1, normal2), 0)
                        if normal_diff > 0.65:
                            row.extend([f1, f2])
                            col.extend([f2, f1])
                            
                        row_normal.extend([f1, f2])
                        col_normal.extend([f2, f1])
                        normals.extend([normal_diff, normal_diff])

                    else:
                        row.extend([f1, f2])
                        col.extend([f2, f1])
                            
    data = np.ones(len(row), dtype=int)
    adjacency_matrix = coo_matrix((data, (row, col)), shape=(num_faces, num_faces))
    adjacency_matrix = adjacency_matrix.tocsr()
    
    
    start_time = time.time()
    normal_diff_matrix = normal_diff_matrix_spatial_sparse(mesh, radius=0.02)
    end_time = time.time()
    print("normal_diff_matrix computation time:", end_time - start_time, "seconds")
    # normal_diff_matrix = coo_matrix((normals, (row_normal, col_normal)), shape=(num_faces, num_faces))
    # normal_diff_matrix = normal_diff_matrix.tocsr()
    
    normal_diff_matrix.setdiag(1)

    weighted_sum = normal_diff_matrix.dot(point_feat)

    row_sums = np.asarray(normal_diff_matrix.sum(axis=1)).ravel()

    eps = 1e-9
    weighted_average = weighted_sum / (row_sums[:, None] + eps)


    
    
    return adjacency_matrix, weighted_average



class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

def hierarchical_clustering_labels(children, n_samples, max_cluster=20):
    # Union-Find structure to maintain cluster merges
    uf = UnionFind(2 * n_samples - 1)  # We may need to store up to 2*n_samples - 1 clusters
    
    current_cluster_count = n_samples
    
    # Process merges from the children array
    hierarchical_labels = []
    for i, (child1, child2) in enumerate(children):
        uf.union(child1, i + n_samples)
        uf.union(child2, i + n_samples)
        #uf.union(child1, child2)
        current_cluster_count -= 1  # After each merge, we reduce the cluster count
        
        if current_cluster_count <= max_cluster:
            labels = [uf.find(i) for i in range(n_samples)]
            hierarchical_labels.append(labels)
    
    return hierarchical_labels

# def solve_clustering(uid, view_id, save_dir="test_results1", max_cluster=20):
#     print(uid, view_id)
#     mesh = trimesh.load(f'{save_dir}/gt_sdf_{uid}_{view_id}.obj')
#     point_feat = np.load(f'{save_dir}/part_feat_{uid}_{view_id}.npy')

#     print(f'{save_dir}/gt_sdf_{uid}_{view_id}.obj')
#     point_feat = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)

#     os.makedirs(f'{save_dir}/__clustering_{uid}_{view_id}', exist_ok=True)
    
#     clustering_path =  f'{save_dir}/clustering_{uid}_{view_id}.pkl'
    
#     # Only keep the largest connected component for Agglomerative Clustering
#     # components_info = mesh.split(only_watertight=False)
#     # components = components_info[0]  # List of meshes
#     # vertex_maps = components_info[1]  # List of vertex indices for each component
    
#     component_mesh = max(mesh.split(only_watertight=False), key=lambda m: len(m.vertices))


#     # Get the face indices in the original mesh
#     component_vertices = component_mesh.vertices

#     # Compare with the original mesh's vertices
#     vertex_indices_in_original = []

#     indices = find_original_vertex_indices(mesh, component_mesh)

#     mesh = component_mesh
#     print("Original vertex indices for this component:", vertex_indices_in_original)

#     # if os.path.exists(clustering_path):
#     #     clustering = joblib.load(clustering_path)
#     # else:
#     point_feat = point_feat.astype(np.float32)
#     point_feat = point_feat[indices]
#     adj_matrix = construct_adjacency_matrix(mesh.faces, mesh.vertices.shape[0])
#     # breakpoint()
#     time1 = time.time()
#     clustering = AgglomerativeClustering(connectivity=adj_matrix,
#                                 n_clusters=1,
#                                 ).fit(point_feat)
#     print("Time:", time.time()-time1)
#     # breakpoint()
    
#     joblib.dump(clustering,clustering_path)
    
#     hierarchical_labels = hierarchical_clustering_labels(clustering.children_, point_feat.shape[0], max_cluster=max_cluster)
    
#     # assign a unique color to each cluster (max_cluster in total)
#     unique_labels = np.unique(hierarchical_labels[0])
#     num_colors = len(unique_labels)
#     cmap = plt.get_cmap('tab20') 
#     colors = cmap(np.linspace(0, 1, num_colors))

#     for n_cluster in range(max_cluster):
#         labels = hierarchical_labels[n_cluster]
#         color_matrix = np.zeros((len(labels), 3))

#         for i, label in enumerate(unique_labels):
#             color_matrix[labels == label] = colors[i, :3]  # Assign RGB values to each label

#         colored_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=color_matrix)
#         colored_mesh.export(f'{save_dir}/__clustering_{uid}_{view_id}/{max_cluster - n_cluster}.ply')


def get_tree_leaves(tree, node_key):
    descendants = []
    stack = [node_key]
    while stack:
        current = stack.pop()
        if current in tree:  # Check if it's a non-leaf node
            left, right = tree[current]['left'], tree[current]['right']
            # descendants.extend([left, right])
            stack.extend([left, right])
        else:
            descendants.append(current)
    return descendants

def build_tree(clustering, num_feat):
    ii = itertools.count(num_feat)
    # tree =[{'node_id': next(ii), 'left': x[0], 'right':x[1]} for x in clustering.children_]
    tree = { next(ii): {'left': x[0], 'right':x[1]} for x in clustering.children_}
    return tree, num_feat*2-2



# def solve_clustering_mesh(mesh, point_feat, clustering_path, num_bridge_faces=0, max_cluster=20, return_color_mesh=False, sample_on_faces=False,consider_normal=False):
    
# #@line_profiler.profile
def solve_clustering_mesh(    mesh,    point_feat,    clustering_path,    num_bridge_faces=0,    max_cluster=20,    return_color_mesh=False,    sample_on_faces=False,    pca_dim=None,):
    # adj_matrix = construct_adjacency_matrix(mesh.faces, mesh.vertices.shape[0])
    if clustering_path is None:
        return_color_mesh = False
    
    if type(point_feat) == torch.Tensor:
        # point_feat = point_feat.cpu().detach().numpy()
        point_feat = point_feat / torch.norm(point_feat, dim=-1, keepdim=True)
    else:
        point_feat = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)
    
    
    
    
    
    time1 = time.time()
    if pca_dim is not None:
        # point_feat = PCA(n_components=pca_dim).fit_transform(point_feat)
        point_feat, _, _ = pca_gpu(point_feat, num_components=pca_dim)
    print(f"PCA time: {time.time()-time1}")

    if type(point_feat) == torch.Tensor:
        point_feat = point_feat.cpu().detach().numpy()

    time1 = time.time() 
    if sample_on_faces:
        adj_matrix =  construct_face_adjacency_matrix_naive(mesh.faces)
    else:
        adj_matrix = construct_adjacency_matrix(mesh.faces, mesh.vertices.shape[0])
    print(f"Adjacency matrix construction time: {time.time()-time1}")
    time1 = time.time()
    # n_components, labels = connected_components(adj_matrix, directed=False)

    # if n_components > 1:
    #     raise Exception(f"Too many connected components ({n_components}). Clustering aborted.")
    clustering = AgglomerativeClustering(connectivity=adj_matrix,
                                n_clusters=1,
                                ).fit(point_feat)
    print("Clustering Time:", time.time()-time1)
    # indptr = adj_matrix.indptr    # length N+1
    # indices = adj_matrix.indices  # length nnz
    # adj_list = [
    #     indices[indptr[i] : indptr[i+1]].tolist()
    #     for i in range(adj_matrix.shape[0])
    # ]

    # clustering = agglo_cpp.AgglomerativeClustering(n_clusters=1)
    # clustering.fit(np.asfortranarray(point_feat, dtype=np.float64) , adj_list, min_clusters=10)

    
    if return_color_mesh:
        clustering_path = f'{clustering_path}/clustering_{point_feat.shape[0]}.pkl'

        os.makedirs(f'{os.path.dirname(clustering_path)}', exist_ok=True)

        if num_bridge_faces > 0:
            # clustering.children_ = clustering.children_[:-num_bridge_faces]
            mesh.faces = mesh.faces[:-num_bridge_faces]
        hierarchical_labels = hierarchical_clustering_labels(clustering.children_, point_feat.shape[0], max_cluster=max_cluster)
        unique_labels = np.unique(hierarchical_labels[0])
        num_colors = len(unique_labels)
        cmap = plt.get_cmap('tab20') 
        colors = cmap(np.linspace(0, 1, num_colors))

        for n_cluster in range(max_cluster):

            labels = hierarchical_labels[n_cluster]
            color_matrix = np.zeros((len(labels), 3))

            for i, label in enumerate(unique_labels):
                color_matrix[labels == label] = colors[i, :3]  # Assign RGB values to each label

            
            if sample_on_faces:
                colored_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
                colored_mesh.visual.face_colors = color_matrix[:mesh.faces.shape[0]]
            else:   
                
                colored_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=color_matrix)

            
            os.makedirs(f'{os.path.dirname(clustering_path)}/__clustering_', exist_ok=True)
            colored_mesh.export(f'{os.path.dirname(clustering_path)}/__clustering_/{max_cluster - n_cluster}.ply')
        print("Color mesh saved to ", f'{os.path.dirname(clustering_path)}/__clustering_/') 

    # return build_tree(clustering, point_feat.shape[0]- num_bridge_faces)
    time1 = time.time()
    tree, num_nodes = build_tree(clustering, point_feat.shape[0]- 0)
    print(f"Tree building time: {time.time()-time1}")
    return tree, num_nodes




# #@line_profiler.profile
def solve_clustering_mesh_pf20(    mesh,    point_feat,    output_path,    num_bridge_faces=0,    max_cluster=20,    return_color_mesh=False,    sample_on_faces=False,    consider_normal=False,    pca_dim=None,):
    # adj_matrix = construct_adjacency_matrix(mesh.faces, mesh.vertices.shape[0])
    os.makedirs(output_path, exist_ok=True)
    
    if type(point_feat) == torch.Tensor:
        # point_feat = point_feat.cpu().detach().numpy()
        point_feat = point_feat / torch.norm(point_feat, dim=-1, keepdim=True)
    else:
        point_feat = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)
    
    
    
    
    
    time1 = time.time()
    if pca_dim is not None:
        # point_feat = PCA(n_components=pca_dim).fit_transform(point_feat)
        point_feat, _, _ = pca_gpu(point_feat, num_components=pca_dim)
    print(f"PCA time: {time.time()-time1}")

    if type(point_feat) == torch.Tensor:
        point_feat = point_feat.cpu().detach().numpy()

    time1 = time.time() 
    if sample_on_faces:
        if consider_normal:
            adj_matrix, point_feat =  construct_adjacency_matrix_face_with_normal(mesh.faces,  mesh.faces.shape[0], mesh, point_feat)
        else:
            adj_matrix =  construct_face_adjacency_matrix_naive(mesh.faces)
            # adj_matrix =  construct_face_adjacency_matrix(mesh.faces, mesh.vertices)
        # adj_matrix,point_feat =  construct_adjacency_matrix_face_with_normal(mesh.faces,  mesh.faces.shape[0], mesh, point_feat)
        # adj_matrix =  construct_adjacency_matrix_face_with_normal(mesh.faces,  mesh.faces.shape[0])
        
    else:
        adj_matrix = construct_adjacency_matrix(mesh.faces, mesh.vertices.shape[0])
    print(f"Adjacency matrix construction time: {time.time()-time1}")
    time1 = time.time()
    # n_components, labels = connected_components(adj_matrix, directed=False)

    # if n_components > 1:
    #     raise Exception(f"Too many connected components ({n_components}). Clustering aborted.")
    clustering = AgglomerativeClustering(connectivity=adj_matrix,
                                n_clusters=1,
                                ).fit(point_feat)
    print("Clustering Time:", time.time()-time1)
    
    hierarchical_labels = hierarchical_clustering_labels(clustering.children_, point_feat.shape[0], max_cluster=max_cluster)
    unique_labels = np.unique(hierarchical_labels[0])
    labels = hierarchical_labels[0]
    

    for i, lbl in enumerate(unique_labels):
        face_idx       = np.where(labels == lbl)[0]        # indices of faces with this label
        part_mesh      = mesh.submesh([face_idx],
                                    append=True,
                                    repair=False)       # keep only those faces/vertices
        part_mesh.export(f'{output_path}/part_{i}.obj')
        
        
        
    # if return_color_mesh:
    #     clustering_path = f'{clustering_path}/clustering_{point_feat.shape[0]}.pkl'

    #     os.makedirs(f'{os.path.dirname(clustering_path)}', exist_ok=True)

    #     if num_bridge_faces > 0:
    #         # clustering.children_ = clustering.children_[:-num_bridge_faces]
    #         mesh.faces = mesh.faces[:-num_bridge_faces]
    #     hierarchical_labels = hierarchical_clustering_labels(clustering.children_, point_feat.shape[0], max_cluster=max_cluster)
    #     unique_labels = np.unique(hierarchical_labels[0])
    #     num_colors = len(unique_labels)
    #     cmap = plt.get_cmap('tab20') 
    #     colors = cmap(np.linspace(0, 1, num_colors))

    #     for n_cluster in range(max_cluster):

    #         labels = hierarchical_labels[n_cluster]
    #         color_matrix = np.zeros((len(labels), 3))

    #         for i, label in enumerate(unique_labels):
    #             color_matrix[labels == label] = colors[i, :3]  # Assign RGB values to each label

            
    #         if sample_on_faces:
    #             colored_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    #             colored_mesh.visual.face_colors = color_matrix[:mesh.faces.shape[0]]
    #         else:   
                
    #             colored_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=color_matrix)

            
    #         os.makedirs(f'{os.path.dirname(clustering_path)}/__clustering_', exist_ok=True)
    #         colored_mesh.export(f'{os.path.dirname(clustering_path)}/__clustering_/{max_cluster - n_cluster}.ply')
    #     print("Color mesh saved to ", f'{os.path.dirname(clustering_path)}/__clustering_/') 

    # return build_tree(clustering, point_feat.shape[0]- num_bridge_faces)
    time1 = time.time()
    tree, num_nodes = build_tree(clustering, point_feat.shape[0]- 0)
    print(f"Tree building time: {time.time()-time1}")
    return 


if __name__ == '__main__':
    # Supported formats include .stl, .obj, .ply, etc.
    mesh = trimesh.load('/ariesdv0/zhaoning/workspace/IUV/lscm/libigl-example-project/meshes/soilder_leg.obj')
    mesh = trimesh.load('/ariesdv0/zhaoning/workspace/IUV/lscm/libigl-example-project/meshes/soilder_leg_merged.obj')
    

    # Access the vertices and faces (if needed)
    vertices = mesh.vertices
    faces = mesh.faces

    # Access the face normals computed by trimesh
    face_normals = mesh.face_normals
    
    solve_clustering_mesh_pf20(mesh, mesh.face_normals, "test_pf20", num_bridge_faces=0, max_cluster=20, return_color_mesh=True, sample_on_faces=True, consider_normal=False)
    