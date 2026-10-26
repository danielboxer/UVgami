import importlib.util
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
from uvgami_cli.common import REPO_ROOT, engine_binaries

FIXTURES = Path(__file__).parent / "fixtures"
SOURCE = REPO_ROOT / "src"

# the addon package imports bpy, so seams loads on its own and a bare package holds the rest
seams_spec = importlib.util.spec_from_file_location(
    "seams",
    SOURCE / "seams" / "__init__.py",
    submodule_search_locations=[str(SOURCE / "seams")],
)
sys.modules["seams"] = importlib.util.module_from_spec(seams_spec)
seams_spec.loader.exec_module(sys.modules["seams"])
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


# the same lookup as the addon and the cli, so UVGAMI_ENGINE_DIR counts
@pytest.fixture
def engine_binary():
    def resolve(name):
        path = engine_binaries.get_local_engine_path(name)
        if path is None:
            pytest.skip(f"no local {name} build")
        return path

    return resolve


@pytest.fixture
def fixture_mesh():
    def read(stem):
        verts, faces = [], []
        for line in (FIXTURES / f"{stem}.obj").read_text().splitlines():
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
        return verts, faces

    return read


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
def fake_engine(tmp_path):
    path = tmp_path / "engine.exe"
    path.write_text("fake")
    return path


# each process records the argv it was started with
@pytest.fixture
def popen_recorder(monkeypatch):
    def record(process_class, **process_kwargs):
        calls = []

        def fake_popen(argv, **kwargs):
            process = process_class(argv, **process_kwargs)
            calls.append(process)
            return process

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        return calls

    return record
