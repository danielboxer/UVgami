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
