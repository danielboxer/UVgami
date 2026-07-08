# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import json
import platform
import subprocess
import sys
import threading
import urllib.request

import bpy

from ..utils.paths import get_partuv_libs_path

RELEASES_API = "https://api.github.com/repos/DanielBoxer/UVgami/releases/latest"

# written by the install thread, read by the preferences ui
install_state = {"running": False, "error": None}


def find_wheel_url():
    request = urllib.request.Request(
        RELEASES_API, headers={"Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    plat = "win_amd64" if platform.system() == "Windows" else "linux_x86_64"
    for asset in release.get("assets", []):
        name = asset["name"]
        if name.startswith("partuv-") and py_tag in name and name.endswith(f"{plat}.whl"):
            return asset["browser_download_url"]
    raise RuntimeError(f"the latest release has no partuv wheel for {py_tag} {plat}")


def run_pip_install(wheel_url):
    python = sys.executable
    has_pip = (
        subprocess.run([python, "-m", "pip", "--version"], capture_output=True)
    ).returncode == 0
    if not has_pip:
        subprocess.run([python, "-m", "ensurepip", "--upgrade"], capture_output=True)
    result = subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--target",
            str(get_partuv_libs_path()),
            wheel_url,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-3:])
        raise RuntimeError(f"pip install failed: {tail}")


class UVGAMI_OT_install_partuv(bpy.types.Operator):
    bl_idname = "uvgami.install_partuv"
    bl_label = "Install PartUV Engine"
    bl_description = (
        "Download the PartUV engine from the latest UVgami release and install it"
        " with Blender's Python. Needs an NVIDIA GPU with CUDA"
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
        threading.Thread(target=self._install, daemon=True).start()

        self._timer = context.window_manager.event_timer_add(
            0.5, window=context.window
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    @staticmethod
    def _install():
        try:
            run_pip_install(find_wheel_url())
        except Exception as error:
            install_state["error"] = str(error)
        finally:
            install_state["running"] = False

    def modal(self, context, event):
        if event.type != "TIMER" or install_state["running"]:
            return {"PASS_THROUGH"}
        context.window_manager.event_timer_remove(self._timer)
        for area in context.screen.areas:
            area.tag_redraw()
        if install_state["error"] is not None:
            self.report({"ERROR"}, f"PartUV install failed: {install_state['error']}")
            return {"CANCELLED"}
        self.report({"INFO"}, "PartUV engine installed")
        return {"FINISHED"}
