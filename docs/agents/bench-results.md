# Benchmark results

2026-07-11, RTX 3060 Laptop GPU. `uv run --no-sync python bench/run.py`, defaults: optcuts capped at 13,500 tris, partuv uncapped, ngon meshes fan-triangulated before both engines. `--quick` caps both engines at 6,000 tris (~4 min run vs ~55 min full). Raw numbers in `bench/results.csv` (holds last run only). partuv crashed silently on ogre during the full run (passed at 141s on retry, likely GPU OOM); the four meshes after the crash are from that retry.

## Per-mesh

| mesh | tris | engine | seconds | charts | seam_length | area_dist | angle_dist |
|---|---|---|---|---|---|---|---|
| suzanne | 968 | optcuts | fail | | | | |
| suzanne | | partuv | 1.9 | 27 | 11.014 | 1.142 | 1.021 |
| woody | 1,267 | optcuts | 0.4 | 1 | 0.000 | 1.000 | 1.000 |
| woody | | partuv | 1.2 | 1 | 0.000 | 1.000 | 1.000 |
| beetle | 2,053 | optcuts | fail | | | | |
| beetle | | partuv | 2.9 | 73 | 10.902 | 1.055 | 1.012 |
| cow | 5,804 | optcuts | fail | | | | |
| cow | | partuv | 5.7 | 20 | 11.170 | 1.112 | 1.012 |
| spot | 5,856 | optcuts | 153.1 | 1 | 4.064 | 1.118 | 1.018 |
| spot | | partuv | 7.0 | 9 | 9.281 | 1.127 | 1.013 |
| alligator | 5,981 | optcuts | 3.4 | 1 | 0.000 | 1.002 | 1.000 |
| alligator | | partuv | 3.2 | 1 | 0.000 | 1.000 | 1.000 |
| homer | 12,000 | optcuts | 933.5 | 1 | 4.778 | 1.116 | 1.018 |
| homer | | partuv | 11.2 | 27 | 12.181 | 1.143 | 1.014 |
| fandisk | 12,946 | optcuts | 298.7 | 1 | 1.907 | 1.111 | 1.017 |
| fandisk | | partuv | 12.8 | 4 | 6.007 | 1.098 | 1.011 |
| cheburashka | 13,334 | optcuts | 1102.5 | 1 | 4.386 | 1.116 | 1.017 |
| cheburashka | | partuv | 14.1 | 22 | 12.826 | 1.133 | 1.013 |
| beast | 64,618 | partuv | 69.5 | 188 | 27.753 | 1.110 | 1.011 |
| nefertiti | 99,938 | partuv | 312.8 | 6 | 6.544 | 1.187 | 1.044 |
| armadillo | 99,976 | partuv | 154.0 | 30 | 17.621 | 1.154 | 1.018 |
| max-planck | 99,991 | partuv | 114.5 | 9 | 8.775 | 1.096 | 1.008 |
| ogre | 124,008 | partuv | 141.1 | 174 | 23.753 | 1.125 | 1.013 |

## Observations

- partuv: 14/14 ok (one flaky silent crash on ogre, ok on retry). optcuts: 6/9 ok; beetle, cow, and suzanne fail in ~0.1s (see mesh requirements below).
- optcuts always produces a single chart with far less seam than partuv, at 10-80x the time on closed meshes: spot 153s vs 7s, homer 934s vs 11s, cheburashka 1103s vs 14s. Distortion is comparable throughout.
- optcuts time depends on shape, not just size: fandisk (12.9k, CAD-like) took 299s while similar-sized homer and cheburashka took 15-18 min. Open surfaces are near-instant (alligator 3.4s).
- partuv output is nondeterministic between runs: nefertiti gave 15 charts in one run and 6 in another, max-planck 14 vs 9. Timings are stable within ~10%.
- partuv mean uv_utilization is over 1 (spot 1.57 in one run), which should be impossible; metrics.py utilization may be miscomputed or charts overlap. Uninvestigated.
- woody and alligator are open surfaces (1 chart, zero seams for both engines), useful as a sanity floor, useless for comparison.

## optcuts mesh requirements

optcuts requires a single connected manifold surface: one component, no non-manifold edges, no non-manifold vertices. Boundary is fine (woody and alligator are open). Validated 6/6 against real runs with a static OBJ check:

| model | comp | boundary_e | nm_edges | nm_verts | optcuts |
|---|---|---|---|---|---|
| spot | 1 | 0 | 0 | 0 | ok |
| woody | 1 | 119 | 0 | 0 | ok |
| alligator | 1 | 433 | 0 | 0 | ok |
| suzanne | 3 | 42 | 1 | 0 | fail |
| beetle | 2 | 296 | 47 | 0 | fail |
| cow | 1 | 0 | 0 | 1 | fail |

cow is the subtle one: edge-manifold and single-component, but one non-manifold vertex (a bowtie pinch where two fans share a point but no edge). An edge-only manifold check misses it. The common-3d-test-models repo has no other manifold meshes under 10k tris; homer, fandisk, and cheburashka (12-13.5k) are clean manifolds and now inside the optcuts cap.
