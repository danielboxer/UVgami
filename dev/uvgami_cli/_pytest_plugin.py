import importlib.util
import sys

import numpy
import pytest

from uvgami_cli.common import REPO_ROOT, engine_binaries

FIXTURES = REPO_ROOT / "dev" / "tests" / "fixtures"
ONE_TRIANGLE_OBJ = "v 0 0 0\nvt 0 0\nf 1/1 1/1 1/1\n"


def corners_and_totals(faces):
    corners = numpy.array([v for face in faces for v in face], dtype=numpy.int64)
    totals = numpy.array([len(face) for face in faces], dtype=numpy.int64)
    return corners, totals


def quad_grid(size, spacing=1.0):
    side = size + 1
    verts = [(x * spacing, y * spacing, 0.0) for y in range(side) for x in range(side)]
    faces = [
        [y * side + x, y * side + x + 1, (y + 1) * side + x + 1, (y + 1) * side + x]
        for y in range(size)
        for x in range(size)
    ]
    return verts, faces


def grid_uvs(verts, faces):
    return [[(verts[v][0], verts[v][1]) for v in face] for face in faces]


NO_BPY_FOLDER = REPO_ROOT / "src" / "no_bpy"

# the addon package imports bpy, so no_bpy loads on its own as a top-level package
no_bpy_spec = importlib.util.spec_from_file_location(
    "no_bpy",
    NO_BPY_FOLDER / "__init__.py",
    submodule_search_locations=[str(NO_BPY_FOLDER)],
)
sys.modules["no_bpy"] = importlib.util.module_from_spec(no_bpy_spec)
no_bpy_spec.loader.exec_module(sys.modules["no_bpy"])


# the repo root __init__.py is the blender addon entry, not a python package
def pytest_collect_directory(path, parent):
    if path == parent.config.rootpath:
        return pytest.Dir.from_parent(parent, path=path)
    return None


# the same lookup as the addon and the cli, so UVGAMI_ENGINE_DIR counts
@pytest.fixture
def engine_binary():
    def resolve(name):
        path = engine_binaries.get_dev_build_path(name)
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
