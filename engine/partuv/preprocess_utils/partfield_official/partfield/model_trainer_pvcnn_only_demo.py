import torch
from .model.UNet.model import ResidualUNet3D
from .model.triplane import TriplaneTransformer, get_grid_coord #, sample_from_planes, Voxel2Triplane
from .model.model_utils import VanillaMLP
import torch.nn.functional as F
import torch.nn as nn
import os
import trimesh
import numpy as np
import torch.distributed as dist
from .model.PVCNN.encoder_pc import TriPlanePC2Encoder, sample_triplane_feat
import json
import gc
import time


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


class Model(nn.Module):
    def __init__(self, cfg, device="cuda"):
        super().__init__()

        self.cfg = cfg
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

    # Inside class Model(nn.Module):
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
            # copy so the normalization below doesn't mutate the caller's mesh,
            # which is unwrapped and saved with its original coordinates
            mesh = mesh.copy()


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
        
    