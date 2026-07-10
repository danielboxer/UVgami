# from split_mesh import split_mesh   
import os
import sys

from os.path import join
import subprocess
import shutil
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from time import perf_counter

import trimesh
import numpy as np


import torch
from preprocess_utils.partfield_official.run_PF import PFInferenceModel

import struct



def PF_pipeline(pf_model, mesh_path, mesh = None, output_path="./pf_pipeline", save_features=False, save_segmentation=False, save_binary=False, sample_on_faces=10, sample_batch_size=100_000):

    os.makedirs(output_path, exist_ok=True)
    
    
    # Clear CUDA cache if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared")
        
    step = 0
    
    folder_path = output_path
    base_name = os.path.splitext(os.path.basename(mesh_path))[0]


    time1 = perf_counter()

    tree, root, mesh = pf_model.process_face(mesh_path, mesh=mesh, output_path=None,device=pf_model.device, save_features=save_features, save_segmentation=save_segmentation, pca_dim=None, sample_on_faces=sample_on_faces, sample_batch_size=sample_batch_size)
    
    time2 = perf_counter()
    print(f"Time taken to predict PF tree: {time2 - time1:.4f}s")
    
    # return 
    if save_binary:
        binary_file_path = os.path.join(folder_path, f"{base_name}.bin")
        
        # Open a binary file for writing in the same folder as mesh_path
        with open(binary_file_path, "wb") as f:
            # Write header: number of nodes (4 bytes, little-endian integer)
            keys = sorted(tree.keys())
            num_nodes = len(keys)
            f.write(struct.pack("<i", num_nodes))
            
            # Write each node as 3 little-endian integers: (id, left, right)
            for key in keys:
                left = tree[key]["left"]
                right = tree[key]["right"]
                f.write(struct.pack("<iii", key, left, right))
        return binary_file_path, tree
    else:
        return None, tree
            


    