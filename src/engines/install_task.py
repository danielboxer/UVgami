import threading

import bpy

from ..utils.ui import tag_redraw

# shared across engines, only one install task runs at a time
task_state = {
    "running": False,
    "owner": "",
    "error": None,
    "phase": "",
    "bytes_done": 0,
    "bytes_total": None,
}


# the unwrap panels show what is installed
INSTALL_AREAS = ("PREFERENCES", "VIEW_3D", "IMAGE_EDITOR")

DOWNLOADED_MESSAGE = "Engine downloaded"
DELETED_MESSAGE = "Engine deleted"
DELETE_DESCRIPTION = "Delete the engine"
NOT_DOWNLOADED_ERROR = "Engine not downloaded. Download it in the add-on preferences"


def parse_version(version):
    parts = version.split(".")
    if len(parts) < 3 or not all(part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


# False when nothing is installed
def version_below(version, bound):
    return version is not None and parse_version(version) < parse_version(bound)


# Blender requires add-ons to check this before any download
def offline_error():
    if bpy.app.online_access:
        return None
    if bpy.app.online_access_override:
        return "Blender was started in offline mode"
    return "Turn on Allow Online Access in the preferences"


# True when it drew, meaning no download can run yet
def draw_online_access(layout):
    if bpy.app.online_access:
        return False
    row = layout.row()
    row.scale_y = 1.5
    # the tooltip has the reason the button is greyed out in offline mode
    row.operator(
        "extensions.userpref_allow_online", text="Allow Online Access", icon="URL"
    )
    return True


def report_progress(done, total):
    task_state["bytes_done"] = done
    task_state["bytes_total"] = total


def _run_task(task):
    try:
        task()
    except Exception as error:
        task_state["error"] = str(error)
    finally:
        task_state["running"] = False


class InstallTask:
    done_message = ""
    owner = ""

    def build_task(self):
        raise NotImplementedError

    def precheck(self):
        return None

    def execute(self, context):
        # imported late, this module loads while the engines package is importing
        from ..manager import manager

        # an install or delete replaces the binary a running unwrap uses
        if manager.is_active:
            self.report({"ERROR"}, "Finish or cancel the current unwrap first")
            return {"CANCELLED"}
        if task_state["running"]:
            self.report({"WARNING"}, "An engine install or delete is already running")
            return {"CANCELLED"}
        error = self.precheck()
        if error is not None:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}

        task_state["running"] = True
        task_state["owner"] = self.owner
        task_state["error"] = None
        task_state["phase"] = ""
        task_state["bytes_done"] = 0
        task_state["bytes_total"] = None
        # built here so the thread never reads operator properties
        threading.Thread(
            target=_run_task, args=(self.build_task(),), daemon=True
        ).start()

        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if task_state["running"]:
            tag_redraw(area_types=INSTALL_AREAS)
            return {"PASS_THROUGH"}
        context.window_manager.event_timer_remove(self._timer)
        # imported late, the engines package imports this module while loading
        from . import invalidate_engine_caches

        invalidate_engine_caches()
        tag_redraw(area_types=INSTALL_AREAS)
        if task_state["error"] is not None:
            self.report({"ERROR"}, f"{self.bl_label} failed: {task_state['error']}")
            return {"CANCELLED"}
        self.report({"INFO"}, self.done_message)
        return {"FINISHED"}


UPDATE_ICON = "FILE_REFRESH"
UPDATE_LABEL_SPLIT = 0.9


# the download button is in the preferences, not here
def draw_update_row(layout, owner, default_phase, text, required):
    if task_state["running"] and task_state["owner"] == owner:
        draw_progress(layout, default_phase)
        return
    if text is None:
        return
    split = layout.split(factor=UPDATE_LABEL_SPLIT)
    split.label(text=text, icon="ERROR" if required else "INFO")
    split.row().operator("uvgami.open_preferences", text="", icon="PREFERENCES")


# the sidebar redraws constantly and the check reads the engine folders
class CachedUpdateNotice:
    # the task_state owner of the engine's install and delete
    install_owner = ""
    install_phase = ""
    # (text, required), read again after invalidate_caches
    _update_notice = None

    def invalidate_caches(self):
        super().invalidate_caches()
        self._update_notice = None

    def draw_update_notice(self, layout):
        if self._update_notice is None:
            required = self.install_too_old()
            state = "required" if required else "available"
            text = f"{self.label} update {state}" if self.update_pending() else None
            self._update_notice = (text, required)
        draw_update_row(
            layout, self.install_owner, self.install_phase, *self._update_notice
        )


# only the first line, a label cannot wrap
def draw_error(layout, owner):
    if task_state["error"] is None or task_state["owner"] != owner:
        return
    layout.row().label(text=task_state["error"].partition("\n")[0], icon="ERROR")


def draw_progress(layout, default_phase):
    row = layout.row()
    phase = task_state["phase"] or default_phase
    total = task_state["bytes_total"]
    if total:
        factor = task_state["bytes_done"] / total
        row.progress(factor=factor, type="BAR", text=f"{phase}  {factor * 100:.0f}%")
    else:
        row.label(text=phase, icon="SORTTIME")
