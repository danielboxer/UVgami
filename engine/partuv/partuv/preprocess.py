import subprocess, os, time
from .preprocess_utils.partfield_official.run_PF import PFInferenceModel
from .preprocess_utils.PartField_pipeline import PF_pipeline
from .preprocess_utils.manifold import fix_mesh_trimesh
from .preprocess_utils.merge_V_obj import load_mesh_and_merge
import shutil

import argparse
        
        
from pathlib import Path
import trimesh
import numpy as np

def preprocess(mesh_path, pf_model=None, output_path=None, save_tree_file=False, save_processed_mesh=False, sample_on_faces=10, sample_batch_size=100_000, merge_vertices_epsilon=1e-7):
    stem, _ = os.path.splitext(os.path.basename(mesh_path))
    if output_path is None:
        output_path = mesh_path
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(os.path.dirname(output_path), stem)
        # shutil.copy(mesh_path, output_path)
        

    preprocess_times = {}

    load_time = time.perf_counter()
    mesh = load_mesh_and_merge(mesh_path, epsilon=merge_vertices_epsilon)
    load_time = time.perf_counter() - load_time
    fix_time = time.perf_counter()
    mesh = fix_mesh_trimesh(mesh)   
    fix_time = time.perf_counter() - fix_time
    export_time = time.perf_counter()

    export_path = os.path.join(output_path, f"{stem}.obj")
    if save_processed_mesh:
        # Export mesh with adapted extension
        mesh.export(export_path)

    export_time = time.perf_counter() - export_time
    tree_dict = {}
    if pf_model is None:
        pf_model = PFInferenceModel(device="cuda")
    
    pf_time = time.perf_counter()

    bin_path = os.path.join(output_path, "bin")
    if save_tree_file:
        os.makedirs(bin_path, exist_ok=True)


    binary_file_path, tree_dict = PF_pipeline(
        pf_model=pf_model,
        mesh_path=export_path,
        mesh=mesh,
        output_path=bin_path,
        save_binary=save_tree_file,
        sample_on_faces=sample_on_faces,
        sample_batch_size=sample_batch_size,
    )
    pf_time = time.perf_counter() - pf_time

    preprocess_times["load"] = load_time
    preprocess_times["fix"] = fix_time
    preprocess_times["export"] = export_time
    preprocess_times["pf"] = pf_time
    
    return mesh, binary_file_path, tree_dict, preprocess_times

def _tm_mesh(V: np.ndarray, F: np.ndarray, UV: np.ndarray) -> trimesh.Trimesh:
    # Keep indices as-is; don't let trimesh merge/simplify
    vis = trimesh.visual.texture.TextureVisuals(
        uv=UV[:, :2]
    )
    return trimesh.Trimesh(vertices=V, faces=F, visual=vis, process=False)

def save_results(output_dir: str | Path, final_parts, individual_parts):
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    
    individual_output_dir = Path(output_dir) / "individual_parts"
    individual_output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    total_components = 0
    for i, part in enumerate(individual_parts):
        comp = part.to_components()
        submesh_path = individual_output_dir / f"part_{i}.obj"
        _tm_mesh(comp.V, comp.F, comp.UV).export(submesh_path, file_type="obj")
        written.append(submesh_path)

        total_components += part.num_components
        # if comp.distortion > 1.5:
        # print(f"Part {i} has {part.num_components} charts and distortion {comp.distortion}")
    with open(output_dir / "hierarchy.json", "w") as f:
        f.write(final_parts.hierarchy_json)
    UV_component = final_parts.to_components()
    print(f"# of V: {UV_component.V.shape[0]}")
    print(f"# of F: {UV_component.F.shape[0]}")
    print(f"# of UV: {UV_component.UV.shape[0]}")

    combined_mesh_path = output_dir / "final_components.obj"
    _tm_mesh(UV_component.V, UV_component.F, UV_component.UV).export(combined_mesh_path, file_type="obj", include_normals=False)
    written.append(combined_mesh_path)
    print(f"Wrote combined OBJ: {combined_mesh_path}")

    # print(f"Total # of Charts: {total_components}")
    return written
        
def main():
    preprocess("./archive/mug.obj", "./test_preprocess/mug_preprocessed.obj")

if __name__ == "__main__":
    main()