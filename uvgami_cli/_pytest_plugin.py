import pytest


# the repo root __init__.py is the blender addon entry, not a python package;
# collect the root as a plain directory so pytest never imports it.
# registered with "-p uvgami_cli._pytest_plugin" because a conftest.py in the
# root would itself be imported as part of that fake package.
def pytest_collect_directory(path, parent):
    if path == parent.config.rootpath:
        return pytest.Dir.from_parent(parent, path=path)
    return None
