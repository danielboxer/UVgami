import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


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
