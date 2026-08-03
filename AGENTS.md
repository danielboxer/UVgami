# UVgami

Blender addon that does automatic UV unwrapping. Three engines: optcuts (C++ binary), xatlas (C++ binary) and partuv (CUDA wheel in `engine/partuv`, runs as `python -m partuv`).

## Layout

- `src/`: the addon. `manager.py` runs the unwrap queue, `src/engines/` is one module per engine (optcuts, xatlas, partuv), listed in `src/engines/__init__.py`.
- `engine/partuv/`: the partuv wheel, C++ core plus python driver (`partuv/cli.py`).
- `dev/`: everything that isn't shipped. `dev/uvgami_cli/`: dev-only CLI driving every engine via `--engine`. `dev/tests/`: the test suite. `dev/bench/`: the benchmark.
- `docs/docs.md`: user guide. `docs/agents/development.md`: dev CLI, tests, benchmarks. `docs/agents/partuv.md`: everything partuv, build setup, options, packaging.
- Benchmarks: run steps, model sets and the baseline workflow are in the Benchmarks section of `docs/agents/development.md`. Append every run to `docs/agents/bench-results.md`.
- `docs/docs.md` and `README.md` are human only. Never edit them, propose the change instead. Anything an agent writes goes in `docs/agents/`.

## Commands

- Test: `uv run --no-sync pytest` (no GPU or Blender needed)
- Lint: `uv run --no-sync ruff check --fix` then `uv run --no-sync ruff format`

## Gotchas

- The dev venv is hand-built: the partuv CUDA stack was installed with `--extra partuv`, which is outside the default sync set. Bare `uv sync` uninstalls all of it, so sync with `uv sync --inexact`. Plain `uv run` is safe (inexact by default).
- The dev venv's editable partuv install holds copies of the python files, not links. After editing `engine/partuv/partuv/*.py` or `engine/partuv/preprocess_utils/*`, copy the file over `.venv/Lib/site-packages/partuv/` or pytest, `python -m partuv` and the engine run stale code. Same after a stash and after the pop. Verify fixes through `partuv.preprocess`: the top-level `preprocess_utils` name maps to the source tree, so a check importing it can pass while the engine runs the stale copy.
- Engine stdout is a parsed protocol (`start:`/`done:`/`failed:`/`progress:` lines). Don't print extra lines to stdout in the engine path, use stderr.
- `src/` imports bpy, so only bpy-free modules (`src/batch.py`, `src/objfile.py`, the `src/seams` package, the partuv package) are unit-testable.
- The addon zip ships no engines. Each one downloads from its own GitHub release on first use, driven by `src/engines/binary_engine.py` (optcuts, xatlas) and `src/engines/partuv/install.py`.
- Changing an engine's code needs that engine's version bumped: `engine/optcuts/VERSION`, `engine/xatlas/VERSION`, `engine/partuv/pyproject.toml`. CI fails if the version constants mirrored in the addon drift. That rebuilds the engine only, releases trigger only from the version line in `blender_manifest.toml`.
- After building an engine, copy the binary into `engines/windows/` (per-platform, gitignored) or the addon and CLI run the stale one. That local copy also beats the downloaded engine, so it's how a dev build gets tested in Blender. If addon behavior contradicts the engine source, compare the binary's mtime to the latest engine commit first.
- GitHub Pages builds the site from the repo root, so anything tracked and not in the `exclude` list in `_config.yml` gets published. Adding a top level folder or a markdown file that isn't user docs means adding it there too.
