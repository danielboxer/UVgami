import shutil
import subprocess
import sys
import types

import pytest
from uvgami_cli._pytest_plugin import FIXTURES
from uvgami_cli.common import REPO_ROOT

from no_bpy.engine_run import ToolsCommand

SOURCE = REPO_ROOT / "src"

# the addon package imports bpy, so a bare package holds its bpy-free modules
addon_package = types.ModuleType("addon_src")
addon_package.__path__ = [str(SOURCE)]
sys.modules.setdefault("addon_src", addon_package)


@pytest.fixture
def triangle(tmp_path):
    path = tmp_path / "triangle.obj"
    shutil.copyfile(FIXTURES / "triangle.obj", path)
    return path


@pytest.fixture
def cube(tmp_path):
    path = tmp_path / "cube.obj"
    shutil.copyfile(FIXTURES / "cube.obj", path)
    return path


@pytest.fixture
def unit_cube():
    verts = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ]
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return verts, faces


@pytest.fixture
def tools(engine_binary):
    return ToolsCommand([engine_binary("uvgami-tools")], None)


@pytest.fixture
def fake_engine(tmp_path):
    path = tmp_path / "engine.exe"
    path.write_text("fake")
    return path


@pytest.fixture
def popen_recorder(monkeypatch):
    # each process records the argv it was started with
    def record(process_class, **process_kwargs):
        calls = []

        def fake_popen(argv, **kwargs):
            process = process_class(argv, **process_kwargs)
            calls.append(process)
            return process

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        return calls

    return record
