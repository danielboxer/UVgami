import importlib
import inspect
import pickle
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

INPUT_NAME = "input"
RESULT_NAME = "result"
CANCEL_NAME = "cancel"
STDOUT_NAME = "stdout"
STDERR_NAME = "stderr"
PROGRESS_PREFIX = "progress: "
STDERR_TAIL_LINES = 10
# how often the worker looks for the cancel file
CANCEL_CHECK_SECONDS = 0.05

# a killed process keeps its log files locked a few milliseconds past its exit
LOG_UNLOCK_SECONDS = 0.1
LOG_UNLOCK_POLL_SECONDS = 0.001

SEAMS_PACKAGE = "seams"
SOURCE_FOLDER = Path(__file__).parents[1]
# the bare interpreter can't import the addon package
BOOTSTRAP = (
    "import sys;"
    "sys.path.insert(0, sys.argv[1]);"
    "from seams.worker import run_in_worker;"
    "run_in_worker(sys.argv[2])"
)


class WorkerError(RuntimeError):
    pass


def remove_folder(folder):
    deadline = time.monotonic() + LOG_UNLOCK_SECONDS
    shutil.rmtree(folder, ignore_errors=True)
    while Path(folder).exists() and time.monotonic() < deadline:
        time.sleep(LOG_UNLOCK_POLL_SECONDS)
        shutil.rmtree(folder, ignore_errors=True)


def last_progress(stdout_path):
    # the last line can be half written
    whole_lines = Path(stdout_path).read_text().rpartition("\n")[0]
    for line in reversed(whole_lines.splitlines()):
        if line.startswith(PROGRESS_PREFIX):
            try:
                return float(line.split()[1])
            except (IndexError, ValueError):
                pass
    return 0.0


def stderr_tail(stderr_path):
    lines = Path(stderr_path).read_text(errors="replace").splitlines()
    return " ".join(lines[-STDERR_TAIL_LINES:]).strip()


class _ResultUnpickler(pickle.Unpickler):
    # the worker pickled seams classes under the top-level package name
    def find_class(self, module, name):
        if module == SEAMS_PACKAGE or module.startswith(SEAMS_PACKAGE + "."):
            module = __package__ + module[len(SEAMS_PACKAGE) :]
        return super().find_class(module, name)


# runs a seams function in its own python process
class WorkerProcess:
    # args and kwargs cross to the worker pickled
    def __init__(self, python, function, *args, **kwargs):
        self._folder = Path(tempfile.mkdtemp(prefix="uvgami-worker-"))
        module = SEAMS_PACKAGE + function.__module__[len(__package__) :]
        call = (module, function.__name__, args, kwargs)
        (self._folder / INPUT_NAME).write_bytes(
            pickle.dumps(call, pickle.HIGHEST_PROTOCOL)
        )
        executable, python_arguments = python
        command = [
            executable,
            *python_arguments,
            "-c",
            BOOTSTRAP,
            str(SOURCE_FOLDER),
            str(self._folder),
        ]
        try:
            with (
                (self._folder / STDOUT_NAME).open("wb") as stdout,
                (self._folder / STDERR_NAME).open("wb") as stderr,
            ):
                self.process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
        except OSError:
            remove_folder(self._folder)
            raise

    def done(self):
        return self.process.poll() is not None

    @property
    def progress(self):
        return last_progress(self._folder / STDOUT_NAME)

    # raises what the function raised
    def result(self):
        code = self.process.wait()
        try:
            result_path = self._folder / RESULT_NAME
            if code != 0 or not result_path.is_file():
                detail = stderr_tail(self._folder / STDERR_NAME)
                raise WorkerError(f"worker exited {code}: {detail}")
            with result_path.open("rb") as file:
                succeeded, value = _ResultUnpickler(file).load()
            traceback_text = (self._folder / STDERR_NAME).read_text(errors="replace")
        finally:
            self.close()
        if not succeeded:
            value.add_note(f"worker traceback: {traceback_text}")
            raise value
        return value

    # only a function that takes cancelled sees this
    def cancel(self):
        if self._folder.is_dir():
            (self._folder / CANCEL_NAME).touch()

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        remove_folder(self._folder)


def _cancel_requested(folder):
    cancel_path = folder / CANCEL_NAME
    state = {"cancelled": False, "checked_at": float("-inf")}

    def cancelled():
        now = time.monotonic()
        if not state["cancelled"] and now - state["checked_at"] > CANCEL_CHECK_SECONDS:
            state["checked_at"] = now
            state["cancelled"] = cancel_path.exists()
        return state["cancelled"]

    return cancelled


def _print_progress(fraction):
    print(f"{PROGRESS_PREFIX}{fraction}", flush=True)


# the worker process's side of WorkerProcess
def run_in_worker(folder):
    folder = Path(folder)
    module, name, args, kwargs = pickle.loads((folder / INPUT_NAME).read_bytes())
    function = getattr(importlib.import_module(module), name)
    parameters = inspect.signature(function).parameters
    if "cancelled" in parameters:
        kwargs["cancelled"] = _cancel_requested(folder)
    if "progress" in parameters:
        kwargs["progress"] = _print_progress
    try:
        outcome = (True, function(*args, **kwargs))
    except Exception as error:
        traceback.print_exc()
        outcome = (False, error)
    (folder / RESULT_NAME).write_bytes(pickle.dumps(outcome, pickle.HIGHEST_PROTOCOL))
