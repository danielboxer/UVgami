import itertools
import tempfile
import time
from pathlib import Path

import numpy

from .cancel import Cancelled
from .worker import (
    STDERR_NAME,
    STDOUT_NAME,
    remove_folder,
    start_logged,
    stderr_tail,
    stop_process,
)


class EngineError(RuntimeError):
    pass


# how often a cancellable engine run checks on the engine
POLL_INTERVAL = 0.05

PACKED_INPUT_NAME = "input.bin"
PACKED_OUTPUT_NAME = "output.bin"
INTEGER = numpy.dtype("<i8")
FLOAT = numpy.dtype("<f8")


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
            remove_folder(self.workdir)

    def stop(self):
        stop_process(self.process, self.workdir)


def face_arrays(uvs):
    sizes = numpy.fromiter((len(face) for face in uvs), INTEGER, len(uvs))
    flat = itertools.chain.from_iterable(itertools.chain.from_iterable(uvs))
    return sizes, numpy.fromiter(flat, FLOAT, 2 * int(sizes.sum()))


# arrays one after another, each length known from the ones before it
def packed_reader(data):
    offset = 0

    def take(dtype, count):
        nonlocal offset
        values = numpy.frombuffer(data, dtype, count, offset)
        offset += values.nbytes
        return values

    return take


# the arrays go in as the command's input, read gets its output's bytes
def start_packed(command, workdir, subcommand, arrays, read):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(dir=workdir))
    with (folder / PACKED_INPUT_NAME).open("wb") as file:
        for array in arrays:
            file.write(array.tobytes())
    paths = [folder / PACKED_INPUT_NAME, folder / PACKED_OUTPUT_NAME]
    args = [str(arg) for arg in [*command, subcommand, *paths]]
    return EngineRun(
        launch(folder, args, None),
        folder,
        paths[1],
        0,
        lambda path, _: read(path.read_bytes()),
    )


def write_obj(path, verts, faces):
    with Path(path).open("w") as f:
        # one %-format over every vertex runs in C, an f-string each is far slower
        f.write(
            ("v %s %s %s\n" * len(verts)) % tuple(itertools.chain.from_iterable(verts))
        )
        f.writelines("f " + " ".join(str(v + 1) for v in face) + "\n" for face in faces)


# run needs poll, stop and result
def poll_run(run, cancelled):
    while run.poll() is None:
        if cancelled is not None and cancelled():
            run.stop()
            raise Cancelled
        time.sleep(POLL_INTERVAL)
    return run.result()


def stdout_path(workdir):
    return workdir / STDOUT_NAME


def stderr_path(workdir):
    return workdir / STDERR_NAME


def launch(workdir, args, env):
    try:
        return start_logged(args, workdir, env)
    except OSError as error:
        raise EngineError(f"engine failed to start: {error}") from error
