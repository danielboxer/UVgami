import subprocess
import tempfile
import time
from pathlib import Path

STDERR_TAIL_LINES = 10

# a killed engine keeps its log files locked a few milliseconds past its exit
LOG_UNLOCK_SECONDS = 0.1
LOG_UNLOCK_POLL_SECONDS = 0.001


# stdout and stderr go to files, a pipe needs a thread to keep it from filling
class EngineProcess:
    def __init__(self, args, env=None):
        self._log_dir = tempfile.TemporaryDirectory(
            prefix="uvgami-engine-", ignore_cleanup_errors=True
        )
        stdout_path = Path(self._log_dir.name) / "stdout"
        self._stderr_path = Path(self._log_dir.name) / "stderr"
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

    # every whole stdout line written since the last call
    def read_lines(self):
        data = self._unfinished_line + self._stdout.read()
        *lines, self._unfinished_line = data.split(b"\n")
        return [line.decode(errors="replace").rstrip("\r") + "\n" for line in lines]

    def stderr_tail(self):
        text = self._stderr_path.read_text(errors="replace")
        return text.splitlines()[-STDERR_TAIL_LINES:]

    # windows can't delete a file the engine still holds open
    def close(self):
        self._stdout.close()
        deadline = time.monotonic() + LOG_UNLOCK_SECONDS
        self._log_dir.cleanup()
        while Path(self._log_dir.name).exists() and time.monotonic() < deadline:
            time.sleep(LOG_UNLOCK_POLL_SECONDS)
            self._log_dir.cleanup()


# tqdm and torch write carriage-return progress bars to stderr
def last_meaningful_line(tail):
    for line in reversed(tail):
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
        if line.startswith("progress: "):
            sink.progress_data.append(line[10:])
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
                uv_co = line[3:].split()
                sink.uv_co.append((float(uv_co[0]), float(uv_co[1])))
            elif line.startswith("f"):
                uv_indices = line[2:].split()
                sink.uv_indices.append(
                    (int(uv_indices[0]), int(uv_indices[1]), int(uv_indices[2]))
                )


class BatchProcess:
    def __init__(self, args, env=None, sinks=None):
        self.sinks = sinks or {}
        self._engine_process = EngineProcess(args, env)
        self.process = self._engine_process.process
        self._started = set()
        self._results = {}
        # None for an argv batch
        self._sent = None
        self._parser = EngineOutput()

    @property
    def started(self):
        self.read_output()
        return self._started

    def read_output(self):
        parser = self._parser
        for line in self._engine_process.read_lines():
            if line.startswith("start: "):
                stem = line[7:].strip()
                self._started.add(stem)
                parser.sink = self.sinks.get(stem)
            elif line.startswith("done: "):
                self._results[line[6:].strip()] = 0
                parser.sink = None
            elif line.startswith("failed: "):
                stem, _, code = line[8:].strip().rpartition(" ")
                try:
                    self._results[stem] = int(code)
                except ValueError:
                    pass
                parser.sink = None
            else:
                parser.feed(line)

    # the exit code reports a write to a dead process
    def send(self, path, sink):
        self.sinks[path.stem] = sink
        self._sent = path.stem
        try:
            print(f"unwrap {path}", file=self.process.stdin, flush=True)
        except OSError:
            pass

    @property
    def is_idle(self):
        if self.process.poll() is not None:
            return False
        self.read_output()
        return self._sent is None or self._sent in self._results

    # a closed stdin is the process's signal to exit
    def close(self):
        try:
            self.process.stdin.close()
        except OSError:
            # a dead engine refuses the flush in close
            pass
        self.process.wait()
        self._engine_process.close()

    # one tail shared by every mesh in the batch
    def stderr_lines(self):
        return self._engine_process.stderr_tail()

    # len(started) keeps a startup crash from requeuing forever
    def should_retry(self, stem):
        return (
            self.process.poll() is not None
            and stem not in self.started
            and stem not in self._results
            and len(self.started) > 0
        )

    # None while pending, 0 when unwrapped, nonzero exit code on failure
    def poll_result(self, stem):
        # a poll after the read can miss a dead process's last markers
        ret = self.process.poll()
        self.read_output()
        code = self._results.get(stem)
        if code is not None:
            return code
        if ret is None:
            return None
        # the process ended without reporting this mesh
        return ret if ret != 0 else 1
