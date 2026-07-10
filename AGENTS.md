# UVgami

Blender addon that auto unwraps UVs. Two engines: optcuts (bundled C++ binary) and partuv (CUDA wheel in `engine/partuv`, runs as `python -m partuv`).

## Layout

- `src/`: the addon. `manager.py` runs the unwrap queue, `engines.py` has per-engine logic.
- `engine/partuv/`: the partuv wheel, C++ core plus python driver (`partuv/cli.py`).
- `uvgami_cli/`: dev-only optcuts CLI, not shipped.
- `docs/docs.md`: user guide, with a development section at the end. `docs/agents/partuv-packaging.md`: packaging decisions and open questions.

## Commands

- Test: `uv run --no-sync pytest` (no GPU or Blender needed)
- Lint: `uv run --no-sync ruff check --fix` then `uv run --no-sync ruff format`
- Use uv for everything, never pip.

## Gotchas

- The dev venv's editable partuv install serves copies of the python files. After editing `engine/partuv/partuv/*.py`, copy the file over `.venv/Lib/site-packages/partuv/` or pytest and `python -m partuv` run stale code.
- Rebuilding the compiled core needs a VS dev shell with CUDA and ninja on PATH. See `docs/agents/partuv-packaging.md`.
- Never add a blocking stdin reader thread to the partuv CLI. On Windows a thread stuck reading stdin stalls native DLL imports.
- Engine stdout is a parsed protocol (`start:`/`done:`/`failed:`/`progress:` lines). Don't print extra lines to stdout in the engine path, use stderr.
- `src/` imports bpy, so only bpy-free modules (`src/batch.py`, the partuv package) are unit-testable.
