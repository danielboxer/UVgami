import torch
import lightning.pytorch as pl
# from .dataloader import Demo_Dataset, Demo_Remesh_Dataset, Correspondence_Demo_Dataset
from torch.utils.data import DataLoader
from .model.UNet.model import ResidualUNet3D
from .model.triplane import TriplaneTransformer, get_grid_coord #, sample_from_planes, Voxel2Triplane
from .model.model_utils import VanillaMLP
import torch.nn.functional as F
import torch.nn as nn
import os
import trimesh
import skimage
import numpy as np
import h5py
import torch.distributed as dist
from .model.PVCNN.encoder_pc import TriPlanePC2Encoder, sample_triplane_feat
import json
import gc
import time
from plyfile import PlyData, PlyElement


def sample_points_on_mesh_cuda(
        vertices: torch.Tensor,      # (V,3) float32/float16  – already on GPU
        faces:    torch.Tensor,      # (F,3) int64            – already on GPU
        k:        int,               # samples / face
        generator: torch.Generator   # for reproducibility
) -> torch.Tensor:
    """
    Return (F, k, 3) points sampled uniformly on each triangular face.
    """
    # Gather vertex triplets for every face: (F,3,3)
    v0 = vertices[faces[:, 0]]          # (F,3)
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # GPU uniform barycentric coordinates
    u1 = torch.rand((faces.shape[0], k), device=vertices.device, generator=generator)
    u2 = torch.rand(u1.shape, dtype=u1.dtype, device=u1.device, generator=generator)

    sqrt_u1 = u1.sqrt()                 # √u₁
    a = 1.0 - sqrt_u1                   # (F,k)
    b = sqrt_u1 * (1.0 - u2)
    c = sqrt_u1 * u2

    # Broadcast and mix the three corners → (F,k,3)
    samples = (
        a[..., None] * v0[:, None, :] +
        b[..., None] * v1[:, None, :] +
        c[..., None] * v2[:, None, :]
    ).to(torch.float16)                # keep everything FP16 to match triplane net
    return samples


class Model(pl.LightningModule):
    def __init__(self, cfg, device="cuda"):
        super().__init__()

        self.save_hyperparameters()
        self.cfg = cfg
        self.automatic_optimization = False
        self.triplane_resolution = cfg.triplane_resolution
        self.triplane_channels_low = cfg.triplane_channels_low
        self.triplane_transformer = TriplaneTransformer(
            input_dim=cfg.triplane_channels_low * 2,
            transformer_dim=1024,
            transformer_layers=6,
            transformer_heads=8,
            triplane_low_res=32,
            triplane_high_res=128,
            triplane_dim=cfg.triplane_channels_high,
        )
        self.sdf_decoder = VanillaMLP(input_dim=64,
                                      output_dim=1, 
                                      out_activation="tanh", 
                                      n_neurons=64, #64
                                      n_hidden_layers=6) #6
        self.use_pvcnn = cfg.use_pvcnnonly
        self.use_2d_feat = cfg.use_2d_feat
        if self.use_pvcnn:
            self.pvcnn = TriPlanePC2Encoder(
                cfg.pvcnn,
                device=device,
                shape_min=-1, 
                shape_length=2,
                use_2d_feat=self.use_2d_feat) #.cuda()
        self.logit_scale = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.grid_coord = get_grid_coord(256)
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss(reduction='none')

        if cfg.regress_2d_feat:
            self.feat_decoder = VanillaMLP(input_dim=64,
                                output_dim=192, 
                                out_activation="GELU", 
                                n_neurons=64, #64
                                n_hidden_layers=6) #6

    def predict_dataloader(self):
        if self.cfg.remesh_demo:
            dataset = Demo_Remesh_Dataset(self.cfg)        
        elif self.cfg.correspondence_demo:
            dataset = Correspondence_Demo_Dataset(self.cfg)
        else:
            dataset = Demo_Dataset(self.cfg)

        dataloader = DataLoader(dataset, 
                            num_workers=self.cfg.dataset.val_num_workers,
                            batch_size=self.cfg.dataset.val_batch_size,
                            shuffle=False, 
                            pin_memory=True,
                            drop_last=False)
        
        return dataloader           


    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        save_dir = f"exp_results/{self.cfg.result_name}"
        os.makedirs(save_dir, exist_ok=True)

        uid = batch['uid'][0]
        view_id = 0
        starttime = time.time()
        
        if uid == "car" or uid == "complex_car":
        # if uid == "complex_car":
            print("Skipping this for now.")
            print(uid)
            return

        ### Skip if model already processed
        if os.path.exists(f'{save_dir}/part_feat_{uid}_{view_id}.npy') or os.path.exists(f'{save_dir}/part_feat_{uid}_{view_id}_batch.npy'):
            print("Already processed "+uid)
            return

        N = batch['pc'].shape[0]
        assert N == 1

        if self.use_2d_feat: 
            print("ERROR. Dataloader not implemented with input 2d feat.")
            exit()
        else:
            pc_feat = self.pvcnn(batch['pc'], batch['pc'])

        planes = pc_feat
        planes = self.triplane_transformer(planes)
        sdf_planes, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)

        if self.cfg.is_pc:
            tensor_vertices = batch['pc'].reshape(1, -1, 3).cuda().to(torch.float16)
            point_feat = sample_triplane_feat(part_planes, tensor_vertices) # N, M, C
            point_feat = point_feat.cpu().detach().numpy().reshape(-1, 448)

            np.save(f'{save_dir}/part_feat_{uid}_{view_id}.npy', point_feat)
            print(f"Exported part_feat_{uid}_{view_id}.npy")

            ###########
            from sklearn.decomposition import PCA
            data_scaled = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)

            pca = PCA(n_components=3)

            data_reduced = pca.fit_transform(data_scaled)
            data_reduced = (data_reduced - data_reduced.min()) / (data_reduced.max() - data_reduced.min())
            colors_255 = (data_reduced * 255).astype(np.uint8)

            points = batch['pc'].squeeze().detach().cpu().numpy()

            if colors_255 is None:
                colors_255 = np.full_like(points, 255)  # Default to white color (255,255,255)
            else:
                assert colors_255.shape == points.shape, "Colors must have the same shape as points"
            
            # Convert to structured array for PLY format
            vertex_data = np.array(
                [(*point, *color) for point, color in zip(points, colors_255)],
                dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
            )

            # Create PLY element
            el = PlyElement.describe(vertex_data, "vertex")
            # Write to file
            filename = f'{save_dir}/feat_pca_{uid}_{view_id}.ply'
            PlyData([el], text=True).write(filename)
            print(f"Saved PLY file: {filename}")
            ############
        
        else:
            use_cuda_version = True
            if use_cuda_version:

                def sample_points(vertices, faces, n_point_per_face):
                    # Generate random barycentric coordinates
                    # borrowed from Kaolin https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/mesh/trianglemesh.py#L43
                    n_f = faces.shape[0]
                    u = torch.sqrt(torch.rand((n_f, n_point_per_face, 1),
                                                device=vertices.device,
                                                dtype=vertices.dtype))
                    v = torch.rand((n_f, n_point_per_face, 1),
                                    device=vertices.device,
                                    dtype=vertices.dtype)
                    w0 = 1 - u
                    w1 = u * (1 - v)
                    w2 = u * v

                    face_v_0 = torch.index_select(vertices, 0, faces[:, 0].reshape(-1))
                    face_v_1 = torch.index_select(vertices, 0, faces[:, 1].reshape(-1))
                    face_v_2 = torch.index_select(vertices, 0, faces[:, 2].reshape(-1))
                    points = w0 * face_v_0.unsqueeze(dim=1) + w1 * face_v_1.unsqueeze(dim=1) + w2 * face_v_2.unsqueeze(dim=1)
                    return points

                def sample_and_mean_memory_save_version(part_planes, tensor_vertices, n_point_per_face):
                    n_sample_each = self.cfg.n_sample_each # we iterate over this to avoid OOM
                    n_v = tensor_vertices.shape[1]
                    n_sample = n_v // n_sample_each + 1
                    all_sample = []
                    for i_sample in range(n_sample):
                        sampled_feature = sample_triplane_feat(part_planes, tensor_vertices[:, i_sample * n_sample_each: i_sample * n_sample_each + n_sample_each,])
                        assert sampled_feature.shape[1] % n_point_per_face == 0
                        sampled_feature = sampled_feature.reshape(1, -1, n_point_per_face, sampled_feature.shape[-1])
                        sampled_feature = torch.mean(sampled_feature, axis=-2)
                        all_sample.append(sampled_feature)
                    return torch.cat(all_sample, dim=1)
                
                if self.cfg.vertex_feature:
                    tensor_vertices = batch['vertices'][0].reshape(1, -1, 3).to(torch.float32)
                    point_feat = sample_and_mean_memory_save_version(part_planes, tensor_vertices, 1)
                else:
                    n_point_per_face = self.cfg.n_point_per_face
                    tensor_vertices = sample_points(batch['vertices'][0], batch['faces'][0], n_point_per_face)
                    tensor_vertices = tensor_vertices.reshape(1, -1, 3).to(torch.float32)
                    point_feat = sample_and_mean_memory_save_version(part_planes, tensor_vertices, n_point_per_face)  # N, M, C

                #### Take mean feature in the triangle
                print("Time elapsed for feature prediction: " + str(time.time() - starttime))
                point_feat = point_feat.reshape(-1, 448).cpu().numpy()
                np.save(f'{save_dir}/part_feat_{uid}_{view_id}_batch.npy', point_feat)
                print(f"Exported part_feat_{uid}_{view_id}.npy")

                ###########
                from sklearn.decomposition import PCA
                data_scaled = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)

                pca = PCA(n_components=3)

                data_reduced = pca.fit_transform(data_scaled)
                data_reduced = (data_reduced - data_reduced.min()) / (data_reduced.max() - data_reduced.min())
                colors_255 = (data_reduced * 255).astype(np.uint8)
                V = batch['vertices'][0].cpu().numpy()
                F = batch['faces'][0].cpu().numpy()
                if self.cfg.vertex_feature:
                    colored_mesh = trimesh.Trimesh(vertices=V, faces=F, vertex_colors=colors_255, process=False)
                else:
                    colored_mesh = trimesh.Trimesh(vertices=V, faces=F, face_colors=colors_255, process=False)
                colored_mesh.export(f'{save_dir}/feat_pca_{uid}_{view_id}.ply')
                ############
                torch.cuda.empty_cache()

            else:
                ### Mesh input (obj file)
                V = batch['vertices'][0].cpu().numpy()
                F = batch['faces'][0].cpu().numpy()

                ##### Loop through faces #####
                num_samples_per_face = self.cfg.n_point_per_face

                all_point_feats = []
                for face in F:
                    # Get the vertices of the current face
                    v0, v1, v2 = V[face]

                    # Generate random barycentric coordinates
                    u = np.random.rand(num_samples_per_face, 1)
                    v = np.random.rand(num_samples_per_face, 1)
                    is_prob = (u+v) >1
                    u[is_prob] = 1 - u[is_prob]
                    v[is_prob] = 1 - v[is_prob]
                    w = 1 - u - v
                    
                    # Calculate points in Cartesian coordinates
                    points = u * v0 + v * v1 + w * v2 

                    tensor_vertices = torch.from_numpy(points.copy()).reshape(1, -1, 3).cuda().to(torch.float32)
                    point_feat = sample_triplane_feat(part_planes, tensor_vertices) # N, M, C 

                    #### Take mean feature in the triangle
                    point_feat = torch.mean(point_feat, axis=1).cpu().detach().numpy()
                    all_point_feats.append(point_feat)                  
                ##############################
                
                all_point_feats = np.array(all_point_feats).reshape(-1, 448)
                
                point_feat = all_point_feats

                np.save(f'{save_dir}/part_feat_{uid}_{view_id}.npy', point_feat)
                print(f"Exported part_feat_{uid}_{view_id}.npy")
                
                ###########
                from sklearn.decomposition import PCA
                data_scaled = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)

                pca = PCA(n_components=3)

                data_reduced = pca.fit_transform(data_scaled)
                data_reduced = (data_reduced - data_reduced.min()) / (data_reduced.max() - data_reduced.min())
                colors_255 = (data_reduced * 255).astype(np.uint8)

                colored_mesh = trimesh.Trimesh(vertices=V, faces=F, face_colors=colors_255, process=False)
                colored_mesh.export(f'{save_dir}/feat_pca_{uid}_{view_id}.ply')
                ############

        print("Time elapsed: " + str(time.time()-starttime))
            
        return 

    # Inside class Model(pl.LightningModule):
    @torch.no_grad()
    # def run_inference(
    #     self,
    #     filename: str,
    #     device: str = "cuda",
    #     combine_components: bool = False,
    #     sample_on_faces: bool | int = True,   # True → use cfg.n_point_per_face; int → override
    #     vertex_feature: bool = False,         # sample at vertices (ignored if sampling faces)
    #     seed: int = 42,
    #     demo_pc_size: int | None = None,      # PVCNN input size (surface samples)
    # ):
    def run_inference(self, filename, mesh = None, device="cuda", sample_batch_size= 100_000, sample_on_faces=False, seed=42,):
        
        if mesh is None:
            mesh = trimesh.load(filename, force='mesh', process=False)
        else:
            mesh = mesh
        

        # normalize mesh
        mesh_scale = 0.9
        vertices = mesh.vertices
        bbmin = vertices.min(0)
        bbmax = vertices.max(0)
        center = (bbmin + bbmax) * 0.5

        scale = mesh_scale / (bbmax - bbmin).max()
        mesh.vertices = (vertices - center) * scale + 0.5
        pc, _ = trimesh.sample.sample_surface(mesh, 100000, seed=seed)

        pc = torch.tensor(pc, dtype=torch.float32, device=device).unsqueeze(0)  # (1, M, 3)

        pc_feat = self.pvcnn(pc, pc)                    # low-res tri-planes
        planes = self.triplane_transformer(pc_feat)     # high-res tri-planes
        sdf_planes, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)

        num_bridge_face = 0

        if sample_on_faces:
            batch_size =  sample_batch_size
            device      = device
            dtype       = torch.float16            # matches part_planes / triplane weights

            # One-time host→device copies (cost amortised over all batches)
            verts_t = torch.as_tensor(mesh.vertices, device=device, dtype=dtype)
            faces_t = torch.as_tensor(mesh.faces,    device=device, dtype=torch.long)
            planes_t = part_planes.to(device=device, dtype=dtype)

            # Optional: reproducible RNG
            g = torch.Generator(device).manual_seed(seed)

            all_face_feats = []                    # will stay on GPU until the very end

            for start in range(0, faces_t.shape[0], batch_size):
                end = min(start + batch_size, faces_t.shape[0])
                f_slice = faces_t[start:end]                       # (B,3)

                # (B, k, 3) → (1, B·k, 3) for the triplane network
                pts = sample_points_on_mesh_cuda(verts_t, f_slice, sample_on_faces, g)
                pts_flat = pts.reshape(1, -1, 3)

                # (1, B·k, C)   -- keeps FP16, stays on the same GPU
                # pts_flat = pts_flat.to(torch.float32)
                feats = sample_triplane_feat(planes_t, pts_flat)

                # Store on-GPU (or immediately move to CPU if RAM permits)
                all_face_feats.append((feats.view(-1, sample_on_faces, 448)).mean(dim=1))

            # Concatenate once at the end; choose where you need the result:
            all_sampled_points = torch.cat(all_face_feats, dim=0)        # GPU tensor
            face_feat = all_sampled_points

            return face_feat, mesh, num_bridge_face
        else:
            tensor_vertices = torch.from_numpy(mesh.vertices.copy()).reshape(1, -1, 3).cuda().to(torch.float16)
            point_feat = sample_triplane_feat(part_planes.to(torch.float16), tensor_vertices.to(torch.float16)) # N, M, C
            point_feat = point_feat.cpu().detach().numpy().reshape(-1, 448)
            return point_feat, mesh, num_bridge_face
        
    