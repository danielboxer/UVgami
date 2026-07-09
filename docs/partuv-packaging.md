# PartUV packaging notes

Findings and open decisions for shipping the PartUV engine in the addon. Branch: `new-engine`.

## Model

PartUV runs in a managed Python 3.11 venv, as a subprocess, decoupled from Blender's Python. Reason: Blender 5.0/5.1 ship Python 3.13, and the partuv `ai` extra pins `torch==2.3.0` + `numpy<2`, neither of which has cp313 wheels. Installing into Blender's Python is dead. `pipeline_numpy` also does not release the GIL, so in-process would freeze the UI. The venv + CLI subprocess sidesteps both.

## Implemented (verified by inspection, not yet run end to end)

- `src/ops/install.py`: downloads pinned `uv` 0.11.25, `uv venv --python 3.11`, `uv pip install "partuv[ai] @ <wheel-url>" -f https://data.pyg.org/whl/torch-2.3.0+cu121.html` (AI) or `"partuv @ <url>"` (geometric). Both tiers run in the venv.
- `find_wheel_url` targets `cp311` (the venv), not Blender's Python, and matches `x86_64.whl` so manylinux names pass.
- Driver ships in the wheel as `python -m partuv` (`partuv/{cli,common,wsl,__main__}.py`), so it can't drift from the compiled core it calls. `uvgami_cli` stays in the repo as the dev-only optcuts CLI. The wheel also packages `config.yaml`, `output.py`, and `geometric.py` (the latter two were imported but missing from the install list). `src/engines.py` installed mode: `[venv_python, "-m", "partuv"]`; `build_env` sets only `UVGAMI_PARTUV_CHECKPOINT`.
- Checkpoint mirrored to our own release: `install.py` pulls from `releases/latest/download/model_objaverse.ckpt`; `.github/workflows/mirror-checkpoint.yml` copies it from HF (run once, rerun if upstream changes).
- `.github/workflows/partuv-build.yml`: Linux wheel `auditwheel repair` (bundles cgal/tbb/yaml-cpp/cudart, excludes big cuda math/dnn libs); matrix cut to cp311 only.
- Windows wheel repair: upstream LLVM libomp (pinned 22.1.8) renamed to `libomp140.x86_64.dll` and inserted into `partuv/.libs` with its license, via `wheel unpack/pack`. Both jobs end with an import smoke test in a fresh venv; the Windows job deletes the System32 libomp140 first (VS puts it there on runners too) so the test can't pass off the build environment.

## Windows DLL closure (measured, not guessed)

`dumpbin /dependents` on `_core.cp311-win_amd64.pyd` gives the real load-time imports:

- bundled already: `gmp-10.dll`, `mpfr-6.dll`, `yaml-cpp.dll` (CMake copies these to `partuv/.libs`, and `__init__.py` adds `.libs` to the DLL search path).
- system / redist: `KERNEL32`, `MSVCP140(*)`, `VCRUNTIME140(*)`, `api-ms-win-crt-*`, `python311`.
- not bundled, not covered: **`libomp140.x86_64.dll`**.
- NOT present: `tbb12.dll`, `cudart`, `easy_profiler.dll`, `gmpxx`. So despite the `TBB::tbb` link, tbb12 is not a runtime dependency, and cudart is confirmed statically linked.

### The libomp140 problem (resolved: bundle upstream LLVM libomp)

`libomp140.x86_64.dll` is the LLVM OpenMP runtime. VS installs it into System32 (that is why native import works on the dev machine), but it is **not** in the VC++ redistributable: it lives in VS `...\VC\Redist\MSVC\<ver>\debug_nonredist\x64`, non-redistributable while `/openmp:llvm` is experimental. A user with only the redist (or Blender) hits an import error.

Can't switch to the redistributable `vcomp140.dll`: `CMakeLists.txt:51` sets `OpenMP_RUNTIME_MSVC "llvm"` on purpose because `pipeline.cpp` uses OpenMP 3.x (`omp_get_level`, `parallel sections`) that classic vcomp (2.0) cannot compile.

Fix: CI inserts upstream LLVM `libomp.dll` (Apache-2.0 with LLVM exception, freely redistributable) into `partuv/.libs` under the import name. Verified locally (2026-07-09):

- all 30 omp imports of `_core.cp311-win_amd64.pyd` (`__kmpc_*`, `omp_*`) are in upstream libomp 22.1.8's exports (dumpbin).
- runtime test: renamed DLL placed in the installed venv's `.libs`, full geometric unwrap of `tests/fixtures/cube.obj` passed, nested parallel regions ran, and `GetModuleFileName` confirmed the loader picked the `.libs` copy over System32 (user dirs precede System32 in the default search order, so the bundled copy also wins when both exist).

## Open decisions

1. **WSL fallback.** Keep for now: it is the fallback when native import fails (older wheels without bundled libomp). Revisit once native Windows is proven on a user machine. The inner command is now `<venv>/bin/python -m partuv ...` (venv on ext4 via `UVGAMI_WSL_VENV` or `$HOME/uvgami-venv`); it no longer needs the repo or uv, since the driver ships in the wheel.

## Dissolved (no action)

- cudart symmetry: static on Windows, auditwheel handles Linux.
- tbb12 bundling: not a runtime dependency.

## Untested at runtime (can't build CUDA / run Blender here)

- cudart 13 (partuv build) vs torch 2.3 cudart 12.1 coexistence in one process (separate sonames, should be fine).
- AI tier needs cp311 wheels throughout; torch 2.3.0 has no cp313, which is why the venv is 3.11.
