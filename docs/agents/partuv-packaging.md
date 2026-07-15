# PartUV packaging notes

Findings and open decisions for shipping the PartUV engine in the addon. Branch: `new-engine`.

## Model

PartUV runs in a managed Python 3.11 venv, as a subprocess, decoupled from Blender's Python. Reason: Blender 5.0/5.1 ship Python 3.13, and the partuv `ai` extra pins `torch==2.3.0` + `numpy<2`, neither of which has cp313 wheels. Installing into Blender's Python is dead. `pipeline_numpy` also does not release the GIL, so in-process would freeze the UI. The venv + CLI subprocess sidesteps both.

## Implemented (verified by inspection, not yet run end to end)

- `src/engines/partuv/install.py`: downloads pinned `uv` 0.11.25, `uv venv --python 3.11`, `uv pip install "partuv[ai] @ <wheel-url>" -f https://data.pyg.org/whl/torch-2.3.0+cu121.html` (AI) or `"partuv @ <url>"` (geometric). Both tiers run in the venv.
- `find_wheel_url` targets `cp311` (the venv), not Blender's Python, and matches `x86_64.whl` so manylinux names pass.
- Driver ships in the wheel as `python -m partuv` (`partuv/{cli,common,__main__}.py`), so it can't drift from the compiled core it calls. `uvgami_cli` stays in the repo as the dev-only optcuts CLI. The wheel also packages `config.yaml`, `output.py`, and `geometric.py` (the latter two were imported but missing from the install list). `src/engines/partuv/__init__.py` installed mode: `[venv_python, "-m", "partuv"]`; `build_env` sets only `UVGAMI_PARTUV_CHECKPOINT`.
- Checkpoint mirrored to the fixed `checkpoint` release: `install.py` pulls from `releases/download/checkpoint/model_objaverse.ckpt`; `.github/workflows/mirror-checkpoint.yml` copies it from HF (run once, rerun if upstream changes).
- Geometric falls back to CPU when no nvidia driver is present: `cli.py` probes `nvcuda.dll`/`libcuda.so.1` and, if absent, rewrites the effective config with `pamo: false` so the cuda mesh simplifier is skipped. AI still requires CUDA (torch check).
- Checkpoint and uv downloads resume: `src/utils/download.py` retries with `Range` requests against a `.part` file and verifies the size, so a dropped connection doesn't restart the 1.24GB checkpoint.
- Wheel version pinned: `install.py` `PARTUV_VERSION` must match `engine/partuv/pyproject.toml` (a test enforces it). `install.py` reads the pinned `partuv-v{PARTUV_VERSION}` release by tag, so it can't grab a newer wheel with a drifted CLI; a missing wheel names that release in the error.
- Native load failure on a user machine raises "reinstall PartUV in the add-on preferences" with the core import error.
- A dead AI batch process no longer fails every mesh: never-started meshes are re-queued into a fresh batch when at least one mesh had started (`BatchProcess.should_retry`); a startup crash still fails everything so requeuing can't loop.
- `.github/workflows/partuv-build.yml`: Linux wheel `auditwheel repair` (bundles cgal/tbb/yaml-cpp/cudart, excludes big cuda math/dnn libs); matrix cut to cp311 only. Triggers: dispatch, or a master push touching `engine/partuv/pyproject.toml` (version bump) or `vcpkg.json`.
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

## Dissolved (no action)

- WSL fallback (removed): the Windows driver used to bridge to a provisioned WSL venv on native import failure. Dropped now that release wheels bundle libomp; native Windows is the only Windows path, and a load failure raises the reinstall hint.
- cudart symmetry: static on Windows, auditwheel handles Linux.
- tbb12 bundling: not a runtime dependency.

## Untested at runtime (can't build CUDA / run Blender here)

- cudart 13 (partuv build) vs torch 2.3 cudart 12.1 coexistence in one process (separate sonames, should be fine).
- AI tier needs cp311 wheels throughout; torch 2.3.0 has no cp313, which is why the venv is 3.11.
