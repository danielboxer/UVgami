# UVgami

Blender addon that auto unwraps UVs. Two engines: optcuts (bundled C++ binary) and partuv (CUDA wheel in `engine/partuv`, runs as `python -m partuv`).

## Layout

- `src/`: the addon. `manager.py` runs the unwrap queue, `engines.py` has per-engine logic.
- `engine/partuv/`: the partuv wheel, C++ core plus python driver (`partuv/cli.py`).
- `uvgami_cli/`: dev-only CLI driving both engines via `--engine`, not shipped.
- `docs/docs.md`: user guide, with a development section at the end. `docs/agents/partuv-packaging.md`: packaging decisions and open questions.

## Commands

- Test: `uv run --no-sync pytest` (no GPU or Blender needed)
- Lint: `uv run --no-sync ruff check --fix` then `uv run --no-sync ruff format`
- Use uv for everything, never pip.

## Gotchas

- The dev venv is hand-built: the partuv CUDA stack was installed with `--extra partuv`, which is outside the default sync set. Bare `uv sync` uninstalls all of it; sync with `uv sync --inexact`. Plain `uv run` is safe (inexact by default).
- The dev venv's editable partuv install serves copies of the python files. After editing `engine/partuv/partuv/*.py`, copy the file over `.venv/Lib/site-packages/partuv/` or pytest and `python -m partuv` run stale code.
- Rebuilding the compiled core needs a VS dev shell with CUDA and ninja on PATH. See `docs/agents/partuv-packaging.md`.
- Never add a blocking stdin reader thread to the partuv CLI. On Windows a thread stuck reading stdin stalls native DLL imports.
- Engine stdout is a parsed protocol (`start:`/`done:`/`failed:`/`progress:` lines). Don't print extra lines to stdout in the engine path, use stderr.
- `src/` imports bpy, so only bpy-free modules (`src/batch.py`, the partuv package) are unit-testable.
- When you change an engine's code, bump that engine's version: optcuts in `engine/optcuts/VERSION`, partuv in `engine/partuv/pyproject.toml` (mirror it in `src/ops/install.py` `PARTUV_VERSION`, `check-partuv-version.yml` fails the build if they drift). That rebuilds the engine only. It does not cut an addon release: a release triggers solely from bumping the version line in `blender_manifest.toml`.
