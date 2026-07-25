# UVgami

Blender addon that does automatic UV unwrapping. Three engines: optcuts and xatlas (bundled C++ binaries) and partuv (CUDA wheel in `engine/partuv`, runs as `python -m partuv`).

## Layout

- `src/`: the addon. `manager.py` runs the unwrap queue, `src/engines/` is one module per engine (optcuts, xatlas, partuv), listed in `src/engines/__init__.py`.
- `engine/partuv/`: the partuv wheel, C++ core plus python driver (`partuv/cli.py`).
- `uvgami_cli/`: dev-only CLI driving every engine via `--engine`, not shipped.
- `docs/docs.md`: user guide. `docs/agents/development.md`: dev CLI, engine build steps, tests. `docs/agents/partuv-packaging.md`: partuv packaging decisions.
- `bench/run.py`: benchmark harness. Append every run to `docs/agents/bench-results.md`.
- `docs/docs.md` and `README.md` are human only. Never edit them, propose the change instead. Anything an agent writes goes in `docs/agents/`.

## Commands

- Test: `uv run --no-sync pytest` (no GPU or Blender needed)
- Lint: `uv run --no-sync ruff check --fix` then `uv run --no-sync ruff format`

## Gotchas

- The dev venv is hand-built: the partuv CUDA stack was installed with `--extra partuv`, which is outside the default sync set. Bare `uv sync` uninstalls all of it, so sync with `uv sync --inexact`. Plain `uv run` is safe (inexact by default).
- The dev venv's editable partuv install holds copies of the python files, not links. After editing `engine/partuv/partuv/*.py`, copy the file over `.venv/Lib/site-packages/partuv/` or pytest and `python -m partuv` run stale code.
- Rebuilding the compiled core needs a VS dev shell with CUDA and ninja on PATH. Steps in `docs/agents/development.md`.
- Never add a blocking stdin reader thread to the partuv CLI. On Windows a thread stuck reading stdin stalls native DLL imports.
- Engine stdout is a parsed protocol (`start:`/`done:`/`failed:`/`progress:` lines). Don't print extra lines to stdout in the engine path, use stderr.
- `src/` imports bpy, so only bpy-free modules (`src/batch.py`, the partuv package) are unit-testable.
- Changing an engine's code needs that engine's version bumped: `engine/optcuts/VERSION`, `engine/xatlas/VERSION`, `engine/partuv/pyproject.toml` (mirrored in `src/engines/partuv/install.py` `PARTUV_VERSION`, `check-partuv-version.yml` fails on drift). That rebuilds the engine only. Releases trigger only from the version line in `blender_manifest.toml`.
- After building an engine, copy the binary to the dev engines folder or the addon runs the stale bundled one: `engine/optcuts/build-perf/optcuts.exe` and `engine/xatlas/build/Release/xatlas.exe` -> `engines/windows/` (per-platform, gitignored).
- GitHub Pages builds the site from the repo root, so anything tracked and not in the `exclude` list in `_config.yml` gets published. Adding a top level folder or a markdown file that isn't user docs means adding it there too.
