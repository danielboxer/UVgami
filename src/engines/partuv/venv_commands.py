# must match the ai extra in partuv's pyproject.toml
TORCH_VERSION = "2.3.0"
# pypi's windows torch wheel is cpu-only
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu121"
# without this page torch-scatter builds from source
TORCH_SCATTER_FIND_LINKS = f"https://data.pyg.org/whl/torch-{TORCH_VERSION}+cu121.html"
VENV_PYTHON = "3.11"


def _uv(uv, *args):
    # uv.toml in a user's config can pin an index or an exclude-newer cutoff
    return [uv, "--no-config", *args]


def build_install_commands(uv, venv_python, venv_path, wheel_url, *, ai, create_venv):
    commands = []
    if create_venv:
        commands.append(_uv(uv, "venv", "--python", VENV_PYTHON, venv_path))
    if ai:
        # cuda torch from this index takes precedence over the ai extra's cpu pin
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
        requirement = f"partuv[ai] @ {wheel_url}"
        extra_args = ["-f", TORCH_SCATTER_FIND_LINKS]
    else:
        requirement = f"partuv @ {wheel_url}"
        extra_args = []
    commands.append(
        _uv(
            uv,
            "pip",
            "install",
            "--python",
            venv_python,
            "--upgrade",
            *extra_args,
            requirement,
        )
    )
    return commands
