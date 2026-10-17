import shutil
import subprocess
import time
from pathlib import Path

from .cancel import Cancelled
from .worker import remove_folder, stderr_tail


class EngineError(RuntimeError):
    pass


# how often a cancellable engine run checks on the engine
POLL_INTERVAL = 0.05


# result() also removes the workdir
class EngineRun:
    def __init__(self, process, workdir, out_path, face_count, read):
        self.process = process
        self.workdir = workdir
        self.out_path = out_path
        self.face_count = face_count
        self.read = read

    def poll(self):
        return self.process.poll()

    def result(self):
        code = self.process.wait()
        try:
            if code != 0:
                detail = stderr_tail(stderr_path(self.workdir))
                raise EngineError(f"engine exited {code}: {detail}")
            return self.read(self.out_path, self.face_count)
        finally:
            shutil.rmtree(self.workdir, ignore_errors=True)

    def stop(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        remove_folder(self.workdir)


def write_obj(path, verts, faces):
    with Path(path).open("w") as f:
        for x, y, z in verts:
            f.write(f"v {x} {y} {z}\n")
        for face in faces:
            f.write("f " + " ".join(str(v + 1) for v in face) + "\n")


# run needs poll, stop and result
def poll_run(run, cancelled):
    while run.poll() is None:
        if cancelled is not None and cancelled():
            run.stop()
            raise Cancelled
        time.sleep(POLL_INTERVAL)
    return run.result()


def stdout_path(workdir):
    return workdir / "stdout"


def stderr_path(workdir):
    return workdir / "stderr"


def launch(workdir, args, env):
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        # the engine stalls when nothing reads its stdout, a file never fills
        with (
            stdout_path(workdir).open("wb") as stdout,
            stderr_path(workdir).open("wb") as stderr,
        ):
            return subprocess.Popen(
                args,
                stdout=stdout,
                stderr=stderr,
                env=env,
                creationflags=creationflags,
            )
    except OSError as error:
        shutil.rmtree(workdir, ignore_errors=True)
        raise EngineError(f"engine failed to start: {error}") from error
