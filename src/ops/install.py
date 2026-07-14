import json
import platform
import shutil
import subprocess
import tarfile
import threading
import urllib.request
import zipfile

import bpy

from ..utils.download import download_file
from ..utils.paths import (
    get_partuv_checkpoint_path,
    get_partuv_venv_path,
    get_partuv_venv_python,
    get_uv_path,
)

# must match engine/partuv/pyproject.toml
PARTUV_VERSION = "0.1.3.0"
PARTUV_RELEASE_API = f"https://api.github.com/repos/DanielBoxer/UVgami/releases/tags/partuv-v{PARTUV_VERSION}"
# mirrored onto the fixed `checkpoint` release so installs don't depend on the HF
# repo staying up or its main branch not moving (see mirror-checkpoint.yml)
CHECKPOINT_URL = "https://github.com/DanielBoxer/UVgami/releases/download/checkpoint/model_objaverse.ckpt"
# the ai extra pins torch 2.3.0 (cu121); torch-scatter has no matching pypi
# wheel, so pip must be pointed at the pyg wheel index to avoid a source build
TORCH_SCATTER_FIND_LINKS = "https://data.pyg.org/whl/torch-2.3.0+cu121.html"
# partuv runs in a managed 3.11 venv, decoupled from blender's python version
VENV_PYTHON = "3.11"
PARTUV_PY_TAG = "cp311"
# pinned so the venv is reproducible; the archive names are stable across releases
UV_VERSION = "0.11.25"
UV_ARCHIVES = {
    "Windows": "uv-x86_64-pc-windows-msvc.zip",
    "Linux": "uv-x86_64-unknown-linux-gnu.tar.gz",
}

# written by the install thread, read by the preferences ui
install_state = {
    "running": False,
    "error": None,
    "phase": "",
    "bytes_done": 0,
    "bytes_total": None,
}


def _report_progress(done, total):
    install_state["bytes_done"] = done
    install_state["bytes_total"] = total


def find_wheel_url():
    request = urllib.request.Request(
        PARTUV_RELEASE_API, headers={"Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)
    # the venv is 3.11 regardless of blender's python; linux wheels ship as
    # manylinux after auditwheel repair, so match the arch tail
    plat = "win_amd64" if platform.system() == "Windows" else "x86_64"
    for asset in release.get("assets", []):
        name = asset["name"]
        if (
            name.startswith(f"partuv-{PARTUV_VERSION}-")
            and PARTUV_PY_TAG in name
            and name.endswith(f"{plat}.whl")
        ):
            return asset["browser_download_url"]
    raise RuntimeError(
        f"no partuv {PARTUV_VERSION} wheel for {PARTUV_PY_TAG} {plat} in the"
        f" partuv-v{PARTUV_VERSION} release"
    )


def _run(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-3:])
        raise RuntimeError(f"{args[0]} failed: {tail}")


def ensure_uv():
    """Download the standalone uv binary if it isn't already present."""
    uv = get_uv_path()
    if uv.is_file():
        return uv
    archive_name = UV_ARCHIVES[platform.system()]
    url = (
        f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{archive_name}"
    )
    uv.parent.mkdir(parents=True, exist_ok=True)
    tmp = uv.parent / archive_name
    install_state["phase"] = "Downloading uv"
    download_file(url, tmp, progress=_report_progress)
    # the archives nest the binary in a per-target folder, extract just uv
    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(tmp) as archive:
            member = next(n for n in archive.namelist() if n.endswith("uv.exe"))
            with archive.open(member) as src, open(uv, "wb") as dst:
                shutil.copyfileobj(src, dst)
    else:
        with tarfile.open(tmp) as archive:
            member = next(m for m in archive.getmembers() if m.name.endswith("/uv"))
            with archive.extractfile(member) as src, open(uv, "wb") as dst:
                shutil.copyfileobj(src, dst)
        uv.chmod(0o755)
    tmp.unlink()
    return uv


def run_venv_install(wheel_url, ai):
    uv = ensure_uv()
    # uv's subprocess output is opaque, so no byte progress here
    install_state["phase"] = "Installing packages"
    install_state["bytes_total"] = None
    venv_python = get_partuv_venv_python()
    if not venv_python.is_file():
        # uv fetches a managed cpython 3.11 if the system has none
        _run([str(uv), "venv", "--python", VENV_PYTHON, str(get_partuv_venv_path())])
    if ai:
        requirement = f"partuv[ai] @ {wheel_url}"
        extra_args = ["-f", TORCH_SCATTER_FIND_LINKS]
    else:
        requirement = f"partuv @ {wheel_url}"
        extra_args = []
    _run(
        [
            str(uv),
            "pip",
            "install",
            "--python",
            str(venv_python),
            "--upgrade",
            *extra_args,
            requirement,
        ]
    )


def download_checkpoint():
    target = get_partuv_checkpoint_path()
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    install_state["phase"] = "Downloading AI checkpoint"
    download_file(CHECKPOINT_URL, target, progress=_report_progress)


class UVGAMI_OT_install_partuv(bpy.types.Operator):
    bl_idname = "uvgami.install_partuv"
    bl_label = "Install PartUV Engine"

    @classmethod
    def description(cls, context, properties):
        if properties.tier == "AI":
            return (
                "Install the PartUV engine with AI segmentation: the engine, the"
                " PyTorch stack and the PartField model, ~5 GB total. Includes"
                " geometric segmentation. Needs an NVIDIA GPU with CUDA."
                " If already installed, reinstalls it"
            )
        return (
            "Install the PartUV engine with geometric segmentation only, which"
            " splits meshes by shape without an AI model, a much smaller download."
            " Needs an NVIDIA GPU with CUDA. If already installed, reinstalls it"
        )

    tier: bpy.props.EnumProperty(
        items=(
            ("GEOMETRIC", "Geometric", ""),
            ("AI", "AI", ""),
        ),
        default="GEOMETRIC",
        options={"HIDDEN"},
    )

    def execute(self, context):
        if install_state["running"]:
            self.report({"WARNING"}, "PartUV install is already running")
            return {"CANCELLED"}
        if platform.system() not in ("Windows", "Linux"):
            self.report({"ERROR"}, "PartUV is only available on Windows and Linux")
            return {"CANCELLED"}

        install_state["running"] = True
        install_state["error"] = None
        install_state["phase"] = ""
        install_state["bytes_done"] = 0
        install_state["bytes_total"] = None
        threading.Thread(target=self._install, args=(self.tier,), daemon=True).start()

        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    @staticmethod
    def _install(tier):
        try:
            ai = tier == "AI"
            run_venv_install(find_wheel_url(), ai)
            if ai:
                download_checkpoint()
        except Exception as error:
            install_state["error"] = str(error)
        finally:
            install_state["running"] = False

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if install_state["running"]:
            # preferences can live in its own window, redraw them all so the bar animates
            for window in context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == "PREFERENCES":
                        area.tag_redraw()
            return {"PASS_THROUGH"}
        context.window_manager.event_timer_remove(self._timer)
        for area in context.screen.areas:
            area.tag_redraw()
        if install_state["error"] is not None:
            self.report({"ERROR"}, f"PartUV install failed: {install_state['error']}")
            return {"CANCELLED"}
        self.report({"INFO"}, "PartUV engine installed")
        return {"FINISHED"}
