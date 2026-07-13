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

## optcuts Windows vs WSL (engine 1.1.2)

2026-07-11, same machine as above. Both binaries from the engine-v1.1.2 release (Windows exe matches the bundled `engines/windows/uvgami.exe` by hash). Direct invocation with `-u 4.1 -s 100 -g`, models copied into each side's native filesystem, two runs each, no other load. `bench/run.py` can't drive WSL or a custom exe path, so these were timed by hand.

| mesh | Windows | WSL |
|---|---|---|
| spot | 179.9s / 193.3s | 73.9s / 85.0s |
| alligator | 3.5s / 3.5s | 1.5s / 1.7s |

WSL is ~2.2-2.4x faster on identical source. This is the bar for the Windows perf work (mimalloc, AVX2, source-level wins) whose goal is removing the addon's WSL path. The gap bundles GCC-vs-MSVC codegen and allocator differences, so no single change is expected to close it alone. The WSL numbers stay valid as a target while the shipped Linux binary is unchanged.

## optcuts perf branch ladder (spot)

2026-07-12, same machine and invocation as the Windows-vs-WSL table, two runs per config, Release/Ninja/MSVC, static CRT in all configs. Configs toggle `UVGAMI_ENABLE_AVX2` and `UVGAMI_USE_MIMALLOC`.

| config | run 1 | run 2 |
|---|---|---|
| baseline 1.1.2 | 179.9s | 193.3s |
| source changes only | 196.9s | 178.7s |
| source + AVX2 | 188.4s | 186.6s |
| source + mimalloc | 85.6s | 85.3s |
| source + AVX2 + mimalloc | 78.0s | 68.1s |

- mimalloc is the win on spot (~2.2x). AVX2 alone does nothing, but adds ~15% once mimalloc removes the allocator bottleneck. Source changes are spot-neutral (they help alligator: 3.32s vs 4.52s source-only, 1.50s full config).
- Run-to-run noise is ~10% (see source-only), so single-run deltas below that are meaningless.
- Full config beats the WSL target (73.9-85.0s) and closes the 2.3x gap.
- Correctness: full-config spot metrics (charts, seam, area/angle distortion, utilization) identical to baseline to 4 decimals; two runs byte-identical, so the parallel loops stay deterministic.
- AVX2 must be applied to every TU. Per-target `/arch:AVX2` changed `EIGEN_MAX_ALIGN_BYTES` in only some TUs and the linker folded Eigen's inline aligned malloc/free into mismatched pairs: allocation with the 32-byte offset-pointer variant, free with plain `free` (heap corruption, caught by ASan in `igl::unique_rows`). Global `add_compile_options` before the deps fixes it.

## optcuts perf branch full run

2026-07-12, `bench/run.py --engine optcuts --optcuts-path build-perf/uvgami.exe` (commit db3e3a8: parallel loops, global AVX2, mimalloc, static CRT), same machine as the baseline table. Big meshes ran solo with a small warmup mesh globbed in.

| mesh | baseline | new | speedup |
|---|---|---|---|
| woody | 0.4s | 0.3s | 1.3x |
| alligator | 3.4s | 1.5s | 2.3x |
| spot | 153.1s | 68.0s | 2.3x |
| fandisk | 298.7s | 53.6s | 5.6x |
| homer | 933.5s | 177.6s | 5.3x |
| cheburashka | 1102.5s | 212.8s | 5.2x |

- Chart count, seam length, and distortion metrics identical to the baseline run on every mesh. suzanne, beetle, cow still fail (non-manifold, expected).
- Speedup grows with mesh size (2.3x at 6k tris, ~5x at 12-13.5k): allocator pressure scales with the solver's working set, so mimalloc gains more on bigger meshes.
- The 12-13.5k meshes now beat the old WSL-side expectation, so the WSL path in the addon has no remaining reason to exist.

## optcuts cached sparse assembly (engine 1.2.2)

2026-07-12, same machine, direct invocation `-u 4.1 -s 100 -g`, baseline is the 1.2.1 master build (mimalloc + AVX2). The change removes the quadratic `conservativeResize` churn in triplet building and `set_pattern`, replaces the per-triplet `std::map` lookups with binary search over sorted CSR rows, and caches triplet destinations between `update_a` calls under the same pattern.

| mesh | baseline | new |
|---|---|---|
| alligator | 1.3s | 1.2s |
| spot | 76.3s / 75.8s | 65.2s / 67.8s |
| fandisk | 64.5s | 50.3s |

- ~13% on spot, ~22% on fandisk. mimalloc had already absorbed most of the allocation cost, so this is the remaining copy and map overhead, growing with mesh size.
- Output OBJs byte-identical to baseline on all three meshes: triplet order and summation order are unchanged, so this is a pure assembly-cost change.

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
