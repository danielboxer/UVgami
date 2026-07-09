# no bpy imports here so the classes stay testable outside blender

import subprocess
import threading


class EngineOutput:
    """Parses engine stdout lines into an unwrap-like sink.

    The optcuts engine answers stdin snapshot requests with a full uv map
    (visual_begin:/vt/f/visual_end:); both engines emit progress: lines."""

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
    """One engine process unwrapping many meshes sequentially.

    Tracks per-mesh state from the cli's start/done/failed stdout markers,
    keyed by input file stem."""

    def __init__(self, args, env=None, sinks=None):
        # unwrap-like sinks keyed by stem, engine output routes to the sink
        # of the mesh currently being unwrapped; passed in here because the
        # reader thread may see the first start marker right away
        self.sinks = sinks or {}
        self.process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            universal_newlines=True,
            env=env,
        )
        self.started = set()
        self._results = {}
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()

    def _read_output(self):
        parser = EngineOutput()
        for line in iter(self.process.stdout.readline, ""):
            if line.startswith("start: "):
                stem = line[7:].strip()
                self.started.add(stem)
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

    def should_retry(self, stem):
        """True when the process has died with this mesh never started and no
        result, but at least one other mesh did start. A mid-batch death has
        intact remaining work, while a startup crash starts nothing and must
        keep failing so requeuing can't loop forever."""
        return (
            self.process.poll() is not None
            and stem not in self.started
            and stem not in self._results
            and len(self.started) > 0
        )

    def poll_result(self, stem):
        """None while pending, 0 when unwrapped, nonzero exit code on failure."""
        code = self._results.get(stem)
        if code is not None:
            return code
        # only fall back to the process exit code once stdout is drained,
        # otherwise a marker could still be in flight
        if self.process.poll() is None or self._reader.is_alive():
            return None
        ret = self.process.poll()
        # the process ended without reporting this mesh
        return ret if ret != 0 else 1
