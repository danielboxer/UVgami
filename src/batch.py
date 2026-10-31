import subprocess
import tempfile
from pathlib import Path

from .no_bpy.worker import (
    DONE_PREFIX,
    FAILED_PREFIX,
    PROGRESS_PREFIX,
    START_PREFIX,
    STDERR_NAME,
    STDERR_TAIL_LINES,
    STDOUT_NAME,
    remove_folder,
)


class EngineProcess:
    def __init__(self, args, env=None):
        self._log_dir = Path(tempfile.mkdtemp(prefix="uvgami-engine-"))
        # partuv stops once its unread stdout buffer is full
        stdout_path = self._log_dir / STDOUT_NAME
        self._stderr_path = self._log_dir / STDERR_NAME
        with stdout_path.open("wb") as stdout, self._stderr_path.open("wb") as stderr:
            self.process = subprocess.Popen(
                args,
                stdout=stdout,
                stdin=subprocess.PIPE,
                stderr=stderr,
                universal_newlines=True,
                env=env,
            )
        self._stdout = stdout_path.open("rb")
        self._unfinished_line = b""

    def read_lines(self):
        data = self._unfinished_line + self._stdout.read()
        *lines, self._unfinished_line = data.split(b"\n")
        return [line.decode(errors="replace").rstrip("\r") + "\n" for line in lines]

    def stderr_tail(self):
        text = self._stderr_path.read_text(errors="replace")
        return text.splitlines()[-STDERR_TAIL_LINES:]

    def close(self):
        self._stdout.close()
        remove_folder(self._log_dir)


def last_meaningful_line(tail):
    for line in reversed(tail):
        # tqdm and torch write carriage-return progress bars to stderr
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


class EngineOutput:
    def __init__(self, sink=None):
        self.sink = sink
        self._in_visual = False

    def feed(self, line):
        sink = self.sink
        if sink is None:
            return
        if line.startswith(PROGRESS_PREFIX):
            sink.progress_data.append(line.removeprefix(PROGRESS_PREFIX))
        elif line == "visual_begin:\n":
            sink.uv_co.clear()
            sink.uv_indices.clear()
            sink.is_uv_data_ready = False
            self._in_visual = True
        elif line == "visual_end:\n":
            sink.is_uv_data_ready = True
            self._in_visual = False
        elif self._in_visual:
            if line.startswith("vt"):
                numbers = line[3:].split()
                sink.uv_co.append((float(numbers[0]), float(numbers[1])))
            elif line.startswith("f"):
                numbers = line[2:].split()
                sink.uv_indices.append(
                    (int(numbers[0]), int(numbers[1]), int(numbers[2]))
                )


class BatchProcess:
    def __init__(self, args, env=None, sinks=None):
        self.args = args
        self.sinks = sinks or {}
        self.engine_process = EngineProcess(args, env)
        self.process = self.engine_process.process
        self._started = set()
        self._results = {}
        self._sent = None
        self._parser = EngineOutput()

    def started_stems(self):
        self.read_output()
        return self._started

    def read_output(self):
        parser = self._parser
        for line in self.engine_process.read_lines():
            if line.startswith(START_PREFIX):
                stem = line.removeprefix(START_PREFIX).strip()
                self._started.add(stem)
                parser.sink = self.sinks.get(stem)
            elif line.startswith(DONE_PREFIX):
                self._results[line.removeprefix(DONE_PREFIX).strip()] = 0
                parser.sink = None
            elif line.startswith(FAILED_PREFIX):
                stem, _, code = line.removeprefix(FAILED_PREFIX).strip().rpartition(" ")
                self._results[stem] = int(code)
                parser.sink = None
            else:
                parser.feed(line)

    def send(self, path, sink):
        self.sinks[path.stem] = sink
        self._sent = path.stem
        try:
            print(f"unwrap {path}", file=self.process.stdin, flush=True)
        except OSError:
            # writes to an exited engine fail
            pass

    def is_idle(self):
        if self.process.poll() is not None:
            return False
        self.read_output()
        return self._sent is None or self._sent in self._results

    def close(self):
        try:
            # partuv exits when its stdin closes
            self.process.stdin.close()
        except OSError:
            # flushing a dead engine's stdin in close raises
            pass
        self.process.wait()
        self.engine_process.close()

    def should_retry(self, stem):
        exited = self.process.poll() is not None
        started = self.started_stems()
        return (
            exited
            and stem not in started
            and stem not in self._results
            # without len(started) a startup crash is requeued forever
            and len(started) > 0
        )

    def poll_result(self, stem):
        # polling after the read drops the last markers of a process that just exited
        exit_code = self.process.poll()
        self.read_output()
        code = self._results.get(stem)
        if code is not None:
            return code
        if exit_code is None:
            return None
        return exit_code if exit_code != 0 else 1
