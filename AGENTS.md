# UVgami

Blender addon that does automatic UV unwrapping. Three engines: optcuts and xatlas (bundled C++ binaries) and partuv (CUDA wheel in `engine/partuv`, runs as `python -m partuv`).

## Layout

- `src/`: the addon. `manager.py` runs the unwrap queue, `src/engines/` is one module per engine (optcuts, xatlas, partuv), listed in `src/engines/__init__.py`.
- `engine/partuv/`: the partuv wheel, C++ core plus python driver (`partuv/cli.py`).
- `dev/`: everything that isn't shipped. `dev/uvgami_cli/`: dev-only CLI driving every engine via `--engine`. `dev/tests/`: the test suite. `dev/bench/`: the benchmark.
- `docs/docs.md`: user guide. `docs/agents/development.md`: dev CLI, engine build steps, tests. `docs/agents/partuv-packaging.md`: partuv packaging decisions.
- `dev/bench/src/run.py`: benchmark harness. Append every run to `docs/agents/bench-results.md`. Models are split into `dev/bench/models/hard-surface/bevel/` (keeps the artist's uv map), `dev/bench/models/hard-surface/sharp/` (thingi10k plus six sketchfab fasteners, no uvs), `dev/bench/models/organic/` (characters and animals, artist uv map) and `dev/bench/models/no-uv/` (shape worth benching, uv map not). All are committed and licensed in `dev/bench/models/ATTRIBUTION.md`, `dev/bench/src/download.py` only re-fetches them. Fetching a sketchfab model needs `SKETCHFAB_API_TOKEN` in the environment. `dev/bench/models/edge-cases/` is generated instead, by `dev/bench/src/gen_edge_cases.py`: small meshes with one known defect each, for testing what an engine does with geometry it cannot unwrap. `dev/bench/models/regression/` holds meshes that once hung or failed in the wild, each with the sidecars its repro needs; wildcards skip it, run it with `--models "regression/*"` plus the route flags recorded in `bench-results.md`, and append new rows to `baseline.csv` by hand since a full run never covers it. The full set takes about an hour, `--models sample` is four models in 207s. A run scores itself against `dev/bench/baseline.csv` (committed) and prints what moved; `--save-baseline` on a full run makes the current numbers the new baseline. Results are keyed by `(mesh, engine, route)`, where route is `scratch`, `import` (`--import-uvs`) or `preseed`, so the same mesh through two routes is two rows.
- `docs/docs.md` and `README.md` are human only. Never edit them, propose the change instead. Anything an agent writes goes in `docs/agents/`.

## Commands

- Test: `uv run --no-sync pytest` (no GPU or Blender needed)
- Lint: `uv run --no-sync ruff check --fix` then `uv run --no-sync ruff format`

## Gotchas

- The dev venv is hand-built: the partuv CUDA stack was installed with `--extra partuv`, which is outside the default sync set. Bare `uv sync` uninstalls all of it, so sync with `uv sync --inexact`. Plain `uv run` is safe (inexact by default).
- The dev venv's editable partuv install holds copies of the python files, not links. After editing `engine/partuv/partuv/*.py` or `engine/partuv/preprocess_utils/*`, copy the file over `.venv/Lib/site-packages/partuv/` or pytest and `python -m partuv` run stale code (`CMakeLists.txt` installs preprocess_utils nested under `partuv/`).
- `preprocess_utils` resolves two ways and only one is the source. The engine imports it relatively as `partuv.preprocess_utils`, which is the venv copy; the top-level name `preprocess_utils` maps to the real source tree. A check script importing the top-level name can pass while the engine runs stale code, so verify through `partuv.preprocess` or copy first. Same trap when stashing for a before/after comparison: re-copy after both the stash and the pop.
- Rebuilding the compiled core needs a VS dev shell with CUDA and ninja on PATH. Steps in `docs/agents/development.md`.
- Never add a blocking stdin reader thread to the partuv CLI. On Windows a thread stuck reading stdin stalls native DLL imports.
- Engine stdout is a parsed protocol (`start:`/`done:`/`failed:`/`progress:` lines). Don't print extra lines to stdout in the engine path, use stderr.
- `src/` imports bpy, so only bpy-free modules (`src/batch.py`, `src/objfile.py`, the `src/seams` package, the partuv package) are unit-testable.
- Changing an engine's code needs that engine's version bumped: `engine/optcuts/VERSION`, `engine/xatlas/VERSION`, `engine/partuv/pyproject.toml` (mirrored in `src/engines/partuv/install.py` `PARTUV_VERSION`, `check-partuv-version.yml` fails on drift). That rebuilds the engine only. Releases trigger only from the version line in `blender_manifest.toml`.
- After building an engine, copy the binary to the dev engines folder or the addon runs the stale bundled one: `engine/optcuts/build-perf/optcuts.exe` and `engine/xatlas/build/Release/xatlas.exe` -> `engines/windows/` (per-platform, gitignored). Same rule after changing engine source: rebuild and copy before testing through the addon or CLI. If addon behavior contradicts the engine source, compare the binary's mtime to the latest engine commit first.
- GitHub Pages builds the site from the repo root, so anything tracked and not in the `exclude` list in `_config.yml` gets published. Adding a top level folder or a markdown file that isn't user docs means adding it there too.
