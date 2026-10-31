from uvgami_cli.common import load_addon_module

# install.py imports bpy, this module does not
venv_commands = load_addon_module("src/engines/partuv/venv_commands.py")

WHEEL_URL = "https://example.invalid/partuv-0.1.4-cp311-cp311-win_amd64.whl"


def build(ai, create_venv=False):
    return venv_commands.build_install_commands(
        "uv", "venv/python.exe", "venv", WHEEL_URL, ai=ai, create_venv=create_venv
    )


def find_argument(command, flag):
    return command[command.index(flag) + 1]


def test_ai_installs_torch_from_the_cuda_index():
    torch_install = next(
        command for command in build(ai=True) if command[-1].startswith("torch==")
    )
    assert "download.pytorch.org/whl/cu121" in find_argument(
        torch_install, "--index-url"
    )
    # the pypi wheel has no local version
    assert torch_install[-1].endswith("+cu121")


def test_ai_installs_the_wheel_with_its_extra():
    assert build(ai=True)[-1][-1] == f"partuv[ai] @ {WHEEL_URL}"


def test_ai_points_torch_scatter_at_its_own_index():
    assert "data.pyg.org" in find_argument(build(ai=True)[-1], "-f")


def test_geometric_installs_no_torch():
    commands = build(ai=False)
    assert not any("torch" in argument for command in commands for argument in command)
    assert commands[-1][-1] == f"partuv @ {WHEEL_URL}"


def test_an_existing_venv_is_not_recreated():
    assert not any("venv" in command for command in build(ai=True, create_venv=False))
    assert "venv" in build(ai=True, create_venv=True)[0]


def test_every_command_ignores_user_config():
    for command in build(ai=True, create_venv=True):
        assert "--no-config" in command
