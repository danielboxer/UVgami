#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sys, glob
import numpy as np
import trimesh

# --------------- utilities -----------------

def load_single_mesh(path: str) -> trimesh.Trimesh:
    """Load an OBJ/GLB/PLY/etc. into a single Trimesh. Scenes are concatenated."""
    m = trimesh.load(path, force='scene' if path.lower().endswith(('.glb', '.gltf')) else None)
    if isinstance(m, trimesh.Scene):
        # dump() returns a list of meshes with transforms applied
        m = trimesh.util.concatenate(m.dump())
    if isinstance(m, list):  # rare: multiple meshes returned
        m = trimesh.util.concatenate(m)
    if not isinstance(m, trimesh.Trimesh):
        raise ValueError(f"Could not load mesh from {path}")
    return m

def strip_uvs(mesh: trimesh.Trimesh) -> None:
    """Remove UVs/materials to mirror the Blender script behavior."""
    try:
        # Reset to plain color visuals (no UVs/textures).
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
    except Exception:
        pass  # safe fallback; exporter will omit UVs if absent

def weld_vertices(mesh: trimesh.Trimesh, epsilon: float = 1e-5, keep: str = "first") -> trimesh.Trimesh:
    """
    Merge vertices whose positions are within 'epsilon'.

    Strategy:
      1) Quantize vertex positions into a 3D grid of size epsilon (fast bucket).
      2) Within each bucket, assign a representative (first or mean).
      3) Remap faces, drop degenerate faces, and clean up.

    This closely mimics Blender's 'remove doubles' distance threshold.
    """
    assert keep in ("first", "mean"), "keep must be 'first' or 'mean'"

    V = mesh.vertices
    F = mesh.faces
    if V.size == 0 or F.size == 0:
        return mesh.copy()

    # Quantize coordinates to integer bins
    bins = np.floor(V / float(epsilon) + 0.5).astype(np.int64)
    # Unique bins and inverse map old_vertex -> unique_bin_index
    # (unique rows on bins)
    uniq_bins, inv = np.unique(bins, axis=0, return_inverse=True)
    n_new = len(uniq_bins)

    # Build new vertex array
    if keep == "first":
        # First occurrence per bin
        first_idx = np.zeros(n_new, dtype=np.int64)
        # The first index for each bin is where inv == bin_id first occurs
        # Use argsort trick for speed/consistency
        order = np.argsort(inv, kind='stable')
        first_of_group = np.ones_like(order, dtype=bool)
        first_of_group[1:] = inv[order][1:] != inv[order][:-1]
        first_idx[inv[order][first_of_group]] = order[first_of_group]
        V_new = V[first_idx]
    else:
        # Mean of all vertices in each bin
        V_new = np.zeros((n_new, 3), dtype=V.dtype)
        counts = np.bincount(inv)
        np.add.at(V_new, inv, V)
        V_new = V_new / counts[:, None]

    # Remap faces to new vertex indices
    F_new = inv[F]

    # Remove degenerate faces (collapsed edges/zero area after welding)
    bad = (F_new[:,0] == F_new[:,1]) | (F_new[:,1] == F_new[:,2]) | (F_new[:,2] == F_new[:,0])
    F_new = F_new[~bad]
    if len(F_new) == 0:
        # Nothing left—return empty mesh with new vertices (rare)
        return trimesh.Trimesh(vertices=V_new, faces=np.empty((0,3), dtype=np.int64), process=False)

    m_out = trimesh.Trimesh(vertices=V_new, faces=F_new, process=False)
    # Clean any isolated vertices introduced by degenerate removal
    m_out.remove_unreferenced_vertices()
    # Optional: also remove duplicate/degenerate faces just in case
    m_out.remove_duplicate_faces()
    m_out.remove_degenerate_faces()
    return m_out

def find_inputs(path: str):
    """Yield files to process: either the single file, or all .obj/.glb under a directory (non-recursive)."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for ext in ("*.obj", "*.glb", "*.gltf", "*.ply", "*.stl"):
            for f in glob.glob(os.path.join(path, ext)):
                yield f
    else:
        raise FileNotFoundError(path)

def export_obj(mesh: trimesh.Trimesh, in_path: str, out_dir: str = None) -> str:
    """Export to OBJ next to input file, mirroring the original name but with .obj extension."""
    base = os.path.splitext(os.path.basename(in_path))[0] + ".obj"
    out_path = os.path.join(out_dir or os.path.dirname(in_path), base)
    # trimesh's OBJ exporter omits UVs if none exist; materials are not required
    mesh.export(out_path)
    return out_path

# --------------- main -----------------


def load_mesh_and_merge(mesh_path: str, epsilon: float = None, keep: str = "first") -> trimesh.Trimesh:
    mesh = load_single_mesh(mesh_path)
    strip_uvs(mesh)
    if epsilon is not None:
        welded = weld_vertices(mesh, epsilon=epsilon, keep=keep)
        return welded
    else:
        return mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True,
                    help="Path to a single mesh file or a directory containing meshes (.obj/.glb/...)")
    ap.add_argument("--epsilon", type=float, default=1e-5,
                    help="Distance threshold for welding (Blender 'remove doubles' equivalent).")
    ap.add_argument("--keep", choices=["first","mean"], default="first",
                    help="Representative vertex per cluster: 'first' (fast, Blender-like) or 'mean' (averaging).")
    ap.add_argument("--verbose", action="store_true", help="Verbose output")
    args = ap.parse_args()

    inputs = list(find_inputs(args.path))
    if args.verbose:
        print(f"Discovered {len(inputs)} file(s)")

    for in_file in inputs:
        if args.verbose:
            print(f"\nProcessing: {in_file}")

        mesh = load_single_mesh(in_file)

        # Remove UV layers to mirror Blender script behavior
        strip_uvs(mesh)

        # Weld vertices within epsilon
        welded = weld_vertices(mesh, epsilon=args.epsilon, keep=args.keep)

        # Export welded mesh as OBJ next to the input (mirrors Blender exporting to OBJ)
        out_obj = export_obj(welded, in_file)
        if args.verbose:
            print(f"  Exported welded mesh -> {out_obj}")
    if args.verbose:
        print("\nBatch processing completed!")

if __name__ == "__main__":
    main()
