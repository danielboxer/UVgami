import itertools
import tempfile
import time
from collections import namedtuple
from pathlib import Path

import numpy

from .worker import (
    DONE_PREFIX,
    STDERR_NAME,
    STDOUT_NAME,
    last_progress,
    remove_folder,
    start_logged,
    stderr_tail,
    stop_process,
)


class EngineError(RuntimeError):
    pass


class Cancelled(Exception):
    pass


def check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise Cancelled


POLL_INTERVAL = 0.05

PACKED_INPUT_NAME = "input.bin"
PACKED_OUTPUT_NAME = "output.bin"
INTEGER = numpy.dtype("<i8")
FLOAT = numpy.dtype("<f8")

ToolsCommand = namedtuple("ToolsCommand", "args env")


class EngineRun:
    def __init__(self, process, workdir, out_path, face_count, read):
        self.process = process
        self.workdir = workdir
        self.out_path = out_path
        self.face_count = face_count
        self.read = read

    def poll(self):
        return self.process.poll()

    def done(self):
        return self.poll() is not None

    @property
    def progress(self):
        return last_progress(stdout_path(self.workdir))

    def result(self):
        code = self.process.wait()
        try:
            if code != 0:
                detail = stderr_tail(self.workdir / STDERR_NAME)
                raise EngineError(f"engine exited {code}: {detail}")
            return self.read(self.out_path, self.face_count)
        finally:
            remove_folder(self.workdir)

    def close(self):
        stop_process(self.process, self.workdir)


def face_arrays(uvs):
    sizes = numpy.fromiter((len(face) for face in uvs), INTEGER, len(uvs))
    flat = itertools.chain.from_iterable(itertools.chain.from_iterable(uvs))
    return sizes, numpy.fromiter(flat, FLOAT, 2 * int(sizes.sum()))


def corner_array(faces, sizes):
    return numpy.fromiter(
        itertools.chain.from_iterable(faces), INTEGER, int(sizes.sum())
    )


def packed_reader(data):
    offset = 0

    def take(dtype, count):
        nonlocal offset
        values = numpy.frombuffer(data, dtype, count, offset)
        offset += values.nbytes
        return values

    return take


def new_run_folder(workdir):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=workdir))


def start_run(folder, args, env, out_path, face_count, read):
    return EngineRun(launch(folder, args, env), folder, out_path, face_count, read)


def start_packed(command, workdir, subcommand, arrays, read):
    folder = new_run_folder(workdir)
    with (folder / PACKED_INPUT_NAME).open("wb") as file:
        for array in arrays:
            file.write(array.tobytes())
    paths = [folder / PACKED_INPUT_NAME, folder / PACKED_OUTPUT_NAME]
    return start_run(
        folder,
        [*command.args, subcommand, *paths],
        command.env,
        paths[1],
        0,
        lambda path, _: read(path.read_bytes()),
    )


def write_obj(path, positions, faces):
    with Path(path).open("w") as file:
        # one %-format over every vertex runs in C, an f-string each is far slower
        file.write(
            ("v %s %s %s\n" * len(positions))
            % tuple(itertools.chain.from_iterable(positions))
        )
        file.writelines(
            "f " + " ".join(str(vertex + 1) for vertex in face) + "\n" for face in faces
        )


def poll_run(run, cancelled):
    while run.poll() is None:
        if cancelled is not None and cancelled():
            run.close()
            raise Cancelled
        time.sleep(POLL_INTERVAL)
    return run.result()


def stdout_path(workdir):
    return workdir / STDOUT_NAME


def done_text(workdir):
    lines = stdout_path(workdir).read_text().splitlines()
    done = [
        line.removeprefix(DONE_PREFIX) for line in lines if line.startswith(DONE_PREFIX)
    ]
    if len(done) != 1:
        raise EngineError(f"engine printed {len(done)} done lines, expected one")
    return done[0]


def launch(workdir, args, env):
    try:
        return start_logged([str(arg) for arg in args], workdir, env)
    except OSError as error:
        raise EngineError(f"engine failed to start: {error}") from error
