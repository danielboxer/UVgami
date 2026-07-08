# no bpy imports here so the class stays testable outside blender

import subprocess
import threading


class BatchProcess:
    """One engine process unwrapping many meshes sequentially.

    Tracks per-mesh state from the cli's start/done/failed stdout markers,
    keyed by input file stem."""

    def __init__(self, args, env=None):
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
        for line in iter(self.process.stdout.readline, ""):
            if line.startswith("start: "):
                self.started.add(line[7:].strip())
            elif line.startswith("done: "):
                self._results[line[6:].strip()] = 0
            elif line.startswith("failed: "):
                stem, _, code = line[8:].strip().rpartition(" ")
                try:
                    self._results[stem] = int(code)
                except ValueError:
                    pass

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
