from uvgami_cli.common import load_addon_module

# install.py imports bpy, this module does not
venv_commands = load_addon_module("src/engines/partuv/partfield/venv_commands.py")

PARTFIELD_URL = "https://example.invalid/partuv_partfield-0.2.0-py3-none-any.whl"


def build(create_venv=False):
    return venv_commands.build_install_commands(
        "uv", "venv/python.exe", "venv", PARTFIELD_URL, create_venv=create_venv
    )


def find_argument(command, flag):
    return command[command.index(flag) + 1]


def test_installs_torch_from_the_cuda_index():
    torch_install = next(
        command for command in build() if command[-1].startswith("torch==")
    )
    assert "download.pytorch.org/whl/cu121" in find_argument(
        torch_install, "--index-url"
    )
    # the pypi wheel has no local version
    assert torch_install[-1].endswith("+cu121")


def test_installs_the_partfield_wheel_last():
    assert build()[-1][-1] == PARTFIELD_URL


def test_points_torch_scatter_at_its_own_index():
    assert "data.pyg.org" in find_argument(build()[-1], "-f")


def test_an_existing_venv_is_not_recreated():
    assert not any("venv" in command for command in build(create_venv=False))
    assert "venv" in build(create_venv=True)[0]


def test_every_command_ignores_user_config():
    for command in build(create_venv=True):
        assert "--no-config" in command
