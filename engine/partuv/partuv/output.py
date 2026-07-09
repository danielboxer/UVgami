"""Result export helpers, kept free of torch imports for the [ai]-less install."""

from pathlib import Path

import numpy as np
import trimesh


def _tm_mesh(V: np.ndarray, F: np.ndarray, UV: np.ndarray) -> trimesh.Trimesh:
    # Keep indices as-is; don't let trimesh merge/simplify
    vis = trimesh.visual.texture.TextureVisuals(
        uv=UV[:, :2]
    )
    return trimesh.Trimesh(vertices=V, faces=F, visual=vis, process=False)


def save_results(output_dir: str | Path, final_parts, individual_parts):
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    UV_component = final_parts.to_components()
    combined_mesh_path = output_dir / "final_components.obj"
    _tm_mesh(UV_component.V, UV_component.F, UV_component.UV).export(combined_mesh_path, file_type="obj", include_normals=False)

    return [combined_mesh_path]
