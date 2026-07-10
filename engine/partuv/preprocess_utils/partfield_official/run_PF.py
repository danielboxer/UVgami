# load_checkpoint.py
import torch

import os
import torch
import numpy as np
from torch.nn.modules.loss import F
import trimesh
from sklearn.decomposition import PCA

from   .partfield.config import default_argument_parser, setup
from   .partfield.model_trainer_pvcnn_only_demo import Model  # Replace with the actual import path of your Model class
from   .AgglomerativeClustering import solve_clustering_mesh, get_tree_leaves, solve_clustering_mesh_pf20
# import line_profiler
import time

from pathlib import Path
from typing import Optional, Union
from importlib.resources import files, as_file


def _pkg_file(rel: str) -> Path:
    pkg = __package__  # e.g., 'partuv.preprocess_utils.partfield'
    res = files(pkg).joinpath(rel)
    # Materialize to a real path (temp when zipped wheels/pyz are used)
    return as_file(res).__enter__()  # caller must ensure it's used promptly

class PFInferenceModel:
    def __init__(self, 
                 cfg_path=None, 
                 checkpoint_path="model_objaverse.ckpt", 
                 device=None):
        parser = default_argument_parser()
        args = parser.parse_args([])
        
        if cfg_path is None:
            cfg_path = _pkg_file("configs/final/demo.yaml")
        
        args.config_file = cfg_path
        cfg = setup(args)
        self.cfg = cfg

        import time
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        # self.model : Model = Model.load_from_checkpoint(
        #     checkpoint_path=checkpoint_path,
        #     cfg=cfg,
        #     map_location=device
        # )

        
        self.model = Model(cfg, device=device)                   
        start_time = time.time()
        
        ckpt = torch.load(checkpoint_path, map_location=device,weights_only=False)
        state = ckpt.get('state_dict', ckpt)       # works for both PL and pure state_dict
        self.model.load_state_dict(state, strict=True)

        
        self.model.to(self.device)
        self.model.eval()
        end_time = time.time()
        print(f"Time to load model: {end_time - start_time:.4f} seconds")
        
        
        
    def postprocess_features(self, point_feat):
        data_scaled = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)
        pca = PCA(n_components=3)
        data_reduced = pca.fit_transform(data_scaled)*-1
        data_reduced = (data_reduced - data_reduced.min()) / (data_reduced.max() - data_reduced.min())
        colors_255 = (data_reduced * 255).astype(np.uint8)
        colors_255 = colors_255[:, [0,1,2]]  # Swap R and B channels
        return colors_255
    

    def generate_colored_mesh(self, obj_path, colors_255, output_path):
        if type(obj_path) == str:
            mesh = trimesh.load(obj_path, force='mesh')
        else:
            mesh = obj_path
        print(colors_255.shape)
        print(mesh.vertices.shape)
        colored_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, face_colors=colors_255)
        colored_mesh.export(output_path)

    def process(self, obj_path, output_path,max_cluster=20, udf_mesh=False, save_features = False, save_segmentation = True):
        point_feat, mesh, num_bridge_face  = self.model.run_inference(obj_path, return_udf_mesh=udf_mesh, combine_components=True)
        if save_features:
            colors_255 = self.postprocess_features(point_feat)
            self.generate_colored_mesh(mesh, colors_255, os.path.join(output_path, f'feat_pca_{os.path.basename(obj_path)}'))

        # if save_segmentation:
        tree, root =  solve_clustering_mesh(mesh, point_feat, output_path, max_cluster=max_cluster, return_color_mesh=save_segmentation,)
        return tree,root,mesh

    #@line_profiler.profile
    def process_face(self, obj_path, output_path,mesh = None, max_cluster=20, udf_mesh=False, save_features = False, save_segmentation = False, seed = 42, sample_on_faces=10, sample_batch_size=100_000, pca_dim=None, device="cuda"):
        print(f"Running PartField Official")
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        else:
            save_features = False
            save_segmentation = False
            
        time1 = time.time()
        print(f"Device: {device}")
        # point_feat, mesh, num_bridge_face = self.model.run_inference(obj_path, seed=seed, device=device)
        point_feat, mesh, num_bridge_face = self.model.run_inference(obj_path,mesh=mesh, sample_batch_size=sample_batch_size, sample_on_faces=sample_on_faces, seed=seed, device=device)
        
        time2 = time.time()
        print(f"Time for inference: {time2-time1}")
        if save_features:
            colors_255 = self.postprocess_features(point_feat.cpu().numpy())
            self.generate_colored_mesh(obj_path, colors_255, os.path.join(output_path, f'feat_pca_{os.path.basename(obj_path).split(".")[0]}.ply'))
        if num_bridge_face > 0:
            print(f"Bridge faces: {num_bridge_face}")
        # num_bridge_face = 0
        # if save_segmentation:
        tree, root =  solve_clustering_mesh(mesh, point_feat, output_path,num_bridge_faces=num_bridge_face, max_cluster=max_cluster, return_color_mesh=save_segmentation,sample_on_faces=True, pca_dim=pca_dim)
        time3 = time.time()
        print(f"Time for clustering: {time3-time2}")
        if num_bridge_face > 0:
            mesh.faces = mesh.faces[:-num_bridge_face]
        return tree,root,mesh
    
   

    def run_clustering(self, point_feat, mesh, output_path, max_cluster=10):
        
        tree, root = solve_clustering_mesh(mesh, point_feat, output_path, max_cluster=max_cluster)
        print(get_tree_leaves(tree, root))
            
            
    def visualize(self, obj_path, output_path):
        point_feat, mesh = self.model.run_inference(obj_path, return_udf_mesh=False, combine_components=True)
        colors_255 = self.postprocess_features(point_feat)
        self.generate_colored_mesh(obj_path, colors_255, output_path)
        