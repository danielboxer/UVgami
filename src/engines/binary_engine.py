import shutil
import zipfile

import bpy

from ..utils.download import download_file, verify_sha256
from ..utils.engine_binaries import (
    get_engine_binary_name,
    get_local_engine_path,
    get_platform_tag,
)
from ..utils.paths import get_extension_dir_path
from . import Engine
from .install_task import (
    DELETE_DESCRIPTION,
    DELETED_MESSAGE,
    DOWNLOADED_MESSAGE,
    NOT_DOWNLOADED_ERROR,
    UPDATE_ICON,
    CachedUpdateNotice,
    InstallTask,
    draw_error,
    draw_online_access,
    draw_progress,
    offline_error,
    parse_version,
    report_progress,
    task_state,
    version_below,
)

RELEASES_URL = "https://github.com/DanielBoxer/UVgami/releases"

DOWNLOAD_PHASE = "Downloading engine"
LICENSE_FILE = "LICENSE.txt"
LICENSES_FOLDER = "licenses"


def engine_install_root(name):
    return get_extension_dir_path() / name


# the engine zips nest the licenses at different depths
def license_target(install_dir, member):
    parts = member.split("/")
    if parts[-1] == LICENSE_FILE:
        return install_dir / LICENSE_FILE
    if LICENSES_FOLDER in parts[:-1] and parts[-1]:
        return install_dir / LICENSES_FOLDER / parts[-1]
    return None


class EngineRelease:
    def __init__(
        self, name, label, version, minimum_version, download_size, archive_sha256s
    ):
        self.name = name
        self.label = label
        self.version = version
        self.minimum_version = minimum_version
        self.download_size = download_size
        self.archive_sha256s = archive_sha256s
        self.install_op = f"uvgami.install_{name}"

    def install_dir(self):
        # named by version, installed_version reads it back
        return engine_install_root(self.name) / self.version

    # true when a download exists, even while a local build is the one that runs
    def is_downloaded(self):
        root = engine_install_root(self.name)
        return root.is_dir() and any(d.is_dir() for d in root.iterdir())

    # a dead download leaves the version folder without a binary
    def installed_version(self):
        root = engine_install_root(self.name)
        if not root.is_dir():
            return None
        binary_name = get_engine_binary_name(self.name)
        names = [
            d.name
            for d in root.iterdir()
            if parse_version(d.name) is not None and (d / binary_name).is_file()
        ]
        return max(names, key=parse_version, default=None)

    def installed_path(self):
        version = self.installed_version()
        if version is None or self.install_too_old():
            return None
        return (
            engine_install_root(self.name) / version / get_engine_binary_name(self.name)
        )

    def installed_below(self, bound):
        return version_below(self.installed_version(), bound)

    def install_too_old(self):
        return self.installed_below(self.minimum_version)

    def update_available(self):
        return self.installed_below(self.version)

    def install(self):
        if self.installed_version() == self.version:
            return
        install_dir = self.install_dir()
        install_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.fetch_binary(install_dir)
        except Exception:
            # an empty version dir counts as downloaded
            shutil.rmtree(install_dir, ignore_errors=True)
            raise

        # drop engines left behind by older addon versions
        for old in install_dir.parent.iterdir():
            if old.is_dir() and old != install_dir:
                shutil.rmtree(old)

    def fetch_binary(self, install_dir):
        platform_tag = get_platform_tag()
        asset = f"{self.name}-engine-{self.version}-{platform_tag}.zip"
        url = f"{RELEASES_URL}/download/{self.name}-v{self.version}/{asset}"
        archive_path = install_dir / asset
        task_state["phase"] = DOWNLOAD_PHASE
        download_file(url, archive_path, progress=report_progress)
        verify_sha256(archive_path, self.archive_sha256s[platform_tag])

        binary = install_dir / get_engine_binary_name(self.name)
        with zipfile.ZipFile(archive_path) as archive:
            # the cpack zip nests the binary in a package folder
            member = next(
                (
                    name
                    for name in archive.namelist()
                    if name.rsplit("/", 1)[-1] == binary.name
                ),
                None,
            )
            if member is None:
                raise RuntimeError(f"{asset} has no {binary.name}")
            with archive.open(member) as src, open(binary, "wb") as dst:
                shutil.copyfileobj(src, dst)
            for name in archive.namelist():
                target = license_target(install_dir, name)
                if target is None:
                    continue
                target.parent.mkdir(exist_ok=True)
                with archive.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        archive_path.unlink()
        binary.chmod(0o755)


class UVGAMI_OT_delete_engine(InstallTask, bpy.types.Operator):
    bl_idname = "uvgami.delete_engine"
    bl_label = "Delete Engine"
    done_message = DELETED_MESSAGE
    bl_description = DELETE_DESCRIPTION

    engine_id: bpy.props.StringProperty(options={"HIDDEN"})

    # the preferences match the progress and errors to the release name
    @property
    def owner(self):
        # imported late, the engines package imports this module while loading
        from . import get_engine

        return get_engine(self.engine_id).release.name

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def build_task(self):
        root = engine_install_root(self.owner)

        def delete():
            task_state["phase"] = "Deleting engine"
            shutil.rmtree(root, ignore_errors=True)

        return delete


class InstallEngineTask(InstallTask):
    done_message = DOWNLOADED_MESSAGE
    release = None

    @classmethod
    def description(cls, context, properties):
        action = "Update" if cls.release.update_available() else "Download"
        return f"{action} the engine"

    def precheck(self):
        if get_platform_tag() is None:
            return f"{self.release.label} has no build for this platform"
        return offline_error()

    def invoke(self, context, event):
        action = "Update" if self.release.update_available() else "Download"
        return context.window_manager.invoke_confirm(
            self,
            event,
            title=f"{action} {self.release.label}",
            message=self.release.download_size,
            confirm_text=action,
        )

    def build_task(self):
        return self.release.install


class BinaryEngine(CachedUpdateNotice, Engine):
    release = None
    has_install_controls = True
    missing_label = "Engine"
    install_phase = DOWNLOAD_PHASE

    @property
    def install_owner(self):
        return self.release.name

    def validate(self, prefs):
        release = self.release
        path = get_local_engine_path(release.name) or release.installed_path()
        if path is not None:
            return path, None
        if release.install_too_old():
            return (
                None,
                "This update needs a newer engine."
                " Download it in the add-on preferences",
            )
        return None, NOT_DOWNLOADED_ERROR

    def install_too_old(self):
        return self.release.install_too_old()

    def describe(self):
        if get_local_engine_path(self.release.name) is not None:
            return f"{self.label} v{self.release.version} (local build)"
        version = self.release.installed_version() or self.release.version
        return f"{self.label} v{version}"

    def update_pending(self):
        release = self.release
        return (
            get_local_engine_path(release.name) is None and release.update_available()
        )

    def draw_not_installed(self, layout, waiting_for=None):
        box = layout.box()
        if task_state["running"] and waiting_for in (None, task_state["owner"]):
            draw_progress(box, DOWNLOAD_PHASE)
            return
        outdated = self.install_too_old()
        row = box.row()
        row.alignment = "CENTER"
        if outdated:
            row.label(text=f"{self.missing_label} update required", icon="FILE_REFRESH")
        else:
            row.label(text=f"{self.missing_label} not downloaded", icon="INFO")
        if draw_online_access(box):
            return
        row = box.row()
        row.scale_y = 1.5
        # skip the confirmation, this is the only way to get an engine
        row.operator_context = "EXEC_DEFAULT"
        row.operator(
            self.release.install_op,
            text="Update Engine" if outdated else "Download Engine",
            icon=UPDATE_ICON if outdated else "IMPORT",
        )

    def draw_prefs(self, layout, prefs):
        release = self.release
        if task_state["running"] and task_state["owner"] == release.name:
            draw_progress(layout, DOWNLOAD_PHASE)
            return

        _, error = self.validate(prefs)
        needs_download = error is not None
        update_pending = self.update_pending()
        installed = release.installed_version()
        change = f"v{installed} → v{release.version}"
        if self.install_too_old():
            status = (f"Engine update required: {change}", "ERROR")
        elif update_pending:
            status = (f"Engine update available: {change}", "FILE_REFRESH")
        elif needs_download:
            status = ("Not downloaded", "X")
        elif get_local_engine_path(release.name) is not None:
            status = ("Local build", "CHECKMARK")
        else:
            status = (f"Downloaded v{installed}", "CHECKMARK")

        layout.row().label(text=status[0], icon=status[1])

        wants_download = needs_download or update_pending
        if wants_download or release.is_downloaded():
            row = layout.row()
            row.scale_y = 1.5
            if wants_download and not draw_online_access(row):
                row.operator(
                    release.install_op,
                    text="Update Engine" if update_pending else "Download Engine",
                    icon=UPDATE_ICON if update_pending else "IMPORT",
                )
            if release.is_downloaded():
                delete = row.operator(
                    "uvgami.delete_engine", text="Delete Engine", icon="TRASH"
                )
                delete.engine_id = self.id

        draw_error(layout, release.name)
