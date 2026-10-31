import functools
import json
import pathlib
import platform
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile

import bpy

from ...utils.download import download_file, with_retries
from ..install_task import (
    DELETE_DESCRIPTION,
    DELETED_MESSAGE,
    DOWNLOADED_MESSAGE,
    InstallTask,
    offline_error,
    report_progress,
    task_state,
    version_below,
)
from .paths import (
    get_partfield_checkpoint_path,
    get_partuv_venv_path,
    get_partuv_venv_python,
    get_uv_path,
)
from .venv_commands import VENV_PYTHON, build_install_commands

# must match partuv's pyproject.toml
PARTUV_VERSION = "0.1.6"
PARTUV_MINIMUM_VERSION = "0.1.0"
PARTUV_RELEASE_API = f"https://api.github.com/repos/DanielBoxer/UVgami/releases/tags/partuv-v{PARTUV_VERSION}"
# Hugging Face's copy changes with its main branch
CHECKPOINT_URL = "https://github.com/DanielBoxer/UVgami/releases/download/checkpoint/model_objaverse.ckpt"
# partuv is an nvidia cuda wheel
PARTUV_PLATFORMS = ("Windows", "Linux")
GEOMETRIC_DOWNLOAD_SIZE = "200 MB"
AI_DOWNLOAD_SIZE = "5 GB"
PARTUV_PY_TAG = "cp311"
UV_VERSION = "0.11.25"
UV_ARCHIVES = {
    "Windows": "uv-x86_64-pc-windows-msvc.zip",
    "Linux": "uv-x86_64-unknown-linux-gnu.tar.gz",
}


@functools.cache
def get_installed_partuv_version():
    venv = get_partuv_venv_path()
    if platform.system() == "Windows":
        site = venv / "Lib" / "site-packages"
    else:
        site = venv / "lib" / f"python{VENV_PYTHON}" / "site-packages"
    dist_info = next(site.glob("partuv-*.dist-info"), None)
    if dist_info is None:
        return None
    return dist_info.name[len("partuv-") : -len(".dist-info")]


def partuv_update_pending():
    return version_below(get_installed_partuv_version(), PARTUV_VERSION)


def partuv_too_old():
    return version_below(get_installed_partuv_version(), PARTUV_MINIMUM_VERSION)


def find_wheel_url():
    def fetch_release():
        request = urllib.request.Request(
            PARTUV_RELEASE_API, headers={"Accept": "application/vnd.github+json"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    release = with_retries(PARTUV_RELEASE_API, fetch_release)
    # linux wheels ship as manylinux after auditwheel repair
    wheel_platform = "win_amd64" if platform.system() == "Windows" else "x86_64"
    for asset in release.get("assets", []):
        name = asset["name"]
        if (
            name.startswith(f"partuv-{PARTUV_VERSION}-")
            and PARTUV_PY_TAG in name
            and name.endswith(f"{wheel_platform}.whl")
        ):
            return asset["browser_download_url"]
    raise RuntimeError(
        f"no partuv {PARTUV_VERSION} wheel for {PARTUV_PY_TAG} {wheel_platform} in the"
        f" partuv-v{PARTUV_VERSION} release"
    )


def ensure_uv():
    uv = get_uv_path()
    if uv.is_file():
        return uv
    archive_name = UV_ARCHIVES[platform.system()]
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


def run_venv_install(wheel_url, *, ai):
    uv = ensure_uv()
    # uv prints no byte counts
    task_state["phase"] = "Installing packages"
    task_state["bytes_total"] = None
    venv_python = get_partuv_venv_python()
    commands = build_install_commands(
        str(uv),
        str(venv_python),
        str(get_partuv_venv_path()),
        wheel_url,
        ai=ai,
        create_venv=not venv_python.is_file(),
    )
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            tail = "\n".join(result.stderr.strip().splitlines()[-3:])
            raise RuntimeError(f"{pathlib.Path(command[0]).name} failed: {tail}")


def install_partuv(*, ai):
    run_venv_install(find_wheel_url(), ai=ai)
    target = get_partfield_checkpoint_path()
    if not ai or target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    task_state["phase"] = "Downloading AI checkpoint"
    download_file(CHECKPOINT_URL, target, progress=report_progress)


def uninstall_partuv():
    task_state["phase"] = "Deleting engine"
    venv = get_partuv_venv_path()
    if venv.is_dir():
        shutil.rmtree(venv)
    get_partfield_checkpoint_path().unlink(missing_ok=True)
    uv_dir = get_uv_path().parent
    if uv_dir.is_dir():
        shutil.rmtree(uv_dir)


class PartuvTask(InstallTask):
    owner = "partuv"

    def precheck(self):
        if platform.system() not in PARTUV_PLATFORMS:
            return "PartUV is only available on Windows and Linux"
        return None


class UVGAMI_OT_install_partuv(PartuvTask, bpy.types.Operator):
    bl_idname = "uvgami.install_partuv"
    bl_label = "Download PartUV Engine"
    done_message = DOWNLOADED_MESSAGE

    def precheck(self):
        return super().precheck() or offline_error()

    @classmethod
    def description(cls, context, properties):
        if properties.tier == "AI":
            return (
                f"Download the engine with AI segmentation, ~{AI_DOWNLOAD_SIZE}."
                " Includes geometric. Needs an NVIDIA GPU"
            )
        return (
            "Download the engine with geometric segmentation only, a much smaller"
            " download. Needs an NVIDIA GPU"
        )

    def invoke(self, context, event):
        ai = self.tier == "AI"
        action = "Update" if partuv_update_pending() else "Download"
        return context.window_manager.invoke_confirm(
            self,
            event,
            title=f"{action} PartUV AI" if ai else f"{action} PartUV",
            message=f"{AI_DOWNLOAD_SIZE}. The NVIDIA AI model is non-commercial only"
            if ai
            else GEOMETRIC_DOWNLOAD_SIZE,
            confirm_text=action,
        )

    tier: bpy.props.EnumProperty(
        items=(
            ("GEOMETRIC", "Geometric", ""),
            ("AI", "AI", ""),
        ),
        default="GEOMETRIC",
        options={"HIDDEN"},
    )

    def build_task(self):
        ai = self.tier == "AI"
        return lambda: install_partuv(ai=ai)


class UVGAMI_OT_uninstall_partuv(PartuvTask, bpy.types.Operator):
    bl_idname = "uvgami.uninstall_partuv"
    bl_label = "Delete PartUV Engine"
    bl_description = DELETE_DESCRIPTION
    done_message = DELETED_MESSAGE

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def build_task(self):
        return uninstall_partuv
