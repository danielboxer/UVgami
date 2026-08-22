"""Run the bpy half of the test suite inside a real Blender.

Most of src/ imports bpy, so the dev venv's pytest can only reach the modules
that don't, and operators, selection state and packing were hand-clicking only.
This runs pytest over this folder inside a background Blender that has the
checkout enabled as an extension.

    uv run --no-sync python dev/tests/blender/run.py [pytest args]

Set BLENDER to pick a specific executable.
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
# blender's --python runs the script without its folder on the path
sys.path.append(str(HERE))

import background_blender  # noqa: E402

if __name__ == "__main__":
    background_blender.main(HERE, __file__)
