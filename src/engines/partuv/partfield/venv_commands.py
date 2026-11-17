# must match partuv-partfield's pyproject.toml
TORCH_VERSION = "2.3.0"
# pypi's windows torch wheel is cpu-only
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu121"
# without this page torch-scatter builds from source
TORCH_SCATTER_FIND_LINKS = f"https://data.pyg.org/whl/torch-{TORCH_VERSION}+cu121.html"
VENV_PYTHON = "3.11"


def _uv(uv, *args):
    # uv.toml in a user's config can pin an index or an exclude-newer cutoff
    return [uv, "--no-config", *args]


def build_install_commands(uv, venv_python, venv_path, wheel_url, *, create_venv):
    commands = []
    if create_venv:
        commands.append(_uv(uv, "venv", "--python", VENV_PYTHON, venv_path))
    # cuda torch from this index takes precedence over partuv-partfield's cpu pin
    commands.append(
        _uv(
            uv,
            "pip",
            "install",
            "--python",
            venv_python,
            "--index-url",
            TORCH_CUDA_INDEX,
            f"torch=={TORCH_VERSION}+cu121",
        )
    )
    commands.append(
        _uv(
            uv,
            "pip",
            "install",
            "--python",
            venv_python,
            "--upgrade",
            "-f",
            TORCH_SCATTER_FIND_LINKS,
            wheel_url,
        )
    )
    return commands
