import functools
import pathlib
import shutil
import subprocess
import tarfile
import zipfile

import bpy

from ....utils.engine_binaries import get_platform_tag
from ....utils.paths import get_dir_path
from ...download import ENGINE_DOWNLOADS_URL, download_file
from ...install_task import (
    DOWNLOADED_MESSAGE,
    offline_error,
    report_progress,
    task_state,
    version_below,
)
from ..install import PARTUV, PARTUV_VERSION, PartuvTask
from ..paths import (
    CHECKPOINT_NAME,
    get_partfield_checkpoint_path,
    get_partuv_venv_path,
    get_partuv_venv_python,
    get_uv_path,
)
from .venv_commands import VENV_PYTHON, build_install_commands

# must match partuv-partfield's pyproject.toml
PARTFIELD_WHEEL = f"partuv_partfield-{PARTUV_VERSION}-py3-none-any.whl"
# Hugging Face's copy changes with its main branch
CHECKPOINT_URL = "https://github.com/DanielBoxer/UVgami/releases/download/checkpoint/model_objaverse.ckpt"
AI_DOWNLOAD_SIZE = "5 GB"
UV_VERSION = "0.11.25"
UV_ARCHIVES = {
    "windows": "uv-x86_64-pc-windows-msvc.zip",
    "linux": "uv-x86_64-unknown-linux-gnu.tar.gz",
}


@functools.cache
def find_partfield_dev_repo():
    repo = get_dir_path()
    if (
        (repo / "dev" / "uvgami_cli").is_dir()
        and (repo / ".venv").is_dir()
        and shutil.which("uv") is not None
    ):
        return repo
    return None


@functools.cache
def get_installed_partfield_version():
    venv = get_partuv_venv_path()
    if get_platform_tag() == "windows":
        site = venv / "Lib" / "site-packages"
    else:
        site = venv / "lib" / f"python{VENV_PYTHON}" / "site-packages"
    dist_info = next(site.glob("partuv_partfield-*.dist-info"), None)
    if dist_info is None:
        return None
    return dist_info.name[len("partuv_partfield-") : -len(".dist-info")]


def is_partfield_installed():
    return (
        get_installed_partfield_version() is not None
        and get_partfield_checkpoint_path().is_file()
    )


def is_ai_segmentation_available():
    return find_partfield_dev_repo() is not None or is_partfield_installed()


def partfield_update_pending():
    return is_partfield_installed() and version_below(
        get_installed_partfield_version(), PARTUV_VERSION
    )


def build_partfield_args(shared, partuv_path):
    repo = find_partfield_dev_repo()
    if repo is None:
        command = [str(get_partuv_venv_python()), "-m", "partuv_partfield"]
        checkpoint = get_partfield_checkpoint_path()
    else:
        command = [
            "uv",
            "run",
            "--project",
            str(repo),
            "--no-sync",
            "python",
            "-m",
            "partuv_partfield",
        ]
        checkpoint = repo / "engine-builds" / CHECKPOINT_NAME
    return [
        *command,
        *shared,
        "--checkpoint",
        str(checkpoint),
        "--partuv-path",
        str(partuv_path),
    ]


def ensure_uv():
    uv = get_uv_path()
    if uv.is_file():
        return uv
    archive_name = UV_ARCHIVES[get_platform_tag()]
    url = (
        f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{archive_name}"
    )
    uv.parent.mkdir(parents=True, exist_ok=True)
    archive_path = uv.parent / archive_name
    task_state["phase"] = "Downloading uv"
    download_file(url, archive_path, progress=report_progress)
    # uv archives nest the binary in a per-target folder
    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            member = next(
                name for name in archive.namelist() if name.endswith("uv.exe")
            )
            with archive.open(member) as source, open(uv, "wb") as destination:
                shutil.copyfileobj(source, destination)
    else:
        with tarfile.open(archive_path) as archive:
            member = next(
                entry for entry in archive.getmembers() if entry.name.endswith("/uv")
            )
            with archive.extractfile(member) as source, open(uv, "wb") as destination:
                shutil.copyfileobj(source, destination)
        uv.chmod(0o755)
    archive_path.unlink()
    return uv


def install_partuv_with_partfield():
    PARTUV.install()
    uv = ensure_uv()
    # uv prints no byte counts
    task_state["phase"] = "Installing packages"
    task_state["bytes_total"] = None
    venv_python = get_partuv_venv_python()
    commands = build_install_commands(
        str(uv),
        str(venv_python),
        str(get_partuv_venv_path()),
        f"{ENGINE_DOWNLOADS_URL}/{PARTFIELD_WHEEL}",
        create_venv=not venv_python.is_file(),
    )
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            tail = "\n".join(result.stderr.strip().splitlines()[-3:])
            raise RuntimeError(f"{pathlib.Path(command[0]).name} failed: {tail}")
    target = get_partfield_checkpoint_path()
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    task_state["phase"] = "Downloading AI checkpoint"
    download_file(CHECKPOINT_URL, target, progress=report_progress)


class UVGAMI_OT_install_partfield(PartuvTask, bpy.types.Operator):
    bl_idname = "uvgami.install_partfield"
    bl_label = "Download PartUV AI"
    bl_description = (
        f"Download the engine with AI segmentation, ~{AI_DOWNLOAD_SIZE}."
        " Includes geometric and tree. Needs an NVIDIA GPU"
    )
    done_message = DOWNLOADED_MESSAGE

    def precheck(self):
        return super().precheck() or offline_error()

    def invoke(self, context, event):
        update = PARTUV.update_available() or partfield_update_pending()
        action = "Update" if update else "Download"
        return context.window_manager.invoke_confirm(
            self,
            event,
            title=f"{action} PartUV AI",
            message=f"{AI_DOWNLOAD_SIZE}. The NVIDIA AI model is non-commercial only",
            confirm_text=action,
        )

    def build_task(self):
        return install_partuv_with_partfield
