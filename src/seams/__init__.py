"""Feature seams by strip merging, the hard surface Seams mode.

No bpy imports anywhere, so the algorithm stays testable outside blender.
Modules in pipeline order: mesh, regions, sweeps, cuts, boundaries,
islands, pipeline."""

from .boundaries import boundary_edges, flatten_teeth, reroute_boundaries
from .cuts import TURN_COST, crease_relief, cut_path, disk_cuts, path_cost, snap_paths
from .islands import crosses, island_ruined, split_islands, uv_topology
from .mesh import (
    LOW_ANGLE,
    build,
    cross,
    diagonal,
    face_edges,
    face_keys,
    island_groups,
    norm,
    pair,
    signed_area,
    turn_angle,
    uv_area_fit,
    uv_fit,
    uv_island_groups,
    vertex_components,
)
from .pipeline import is_hard_surface, seam_edges
from .regions import (
    CREASE_ANGLE,
    FLAT_ANGLE,
    absorb,
    close_rings,
    detect_width,
    merge_flat,
    merge_smooth,
    partition,
    region_topology,
)
from .sweeps import split_sweeps
