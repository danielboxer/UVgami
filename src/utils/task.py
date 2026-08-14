import threading


class BackgroundTask:
    """A callable run on a worker thread, holding its result until the main
    thread collects it.

    Blender's data is not thread safe, so the work must touch no bpy data and
    the caller polls done() from a timer. A thread cannot be killed, so
    cancel() only sets the flag the work is handed."""

    def __init__(self, work):
        self._box = {}
        self._cancelled = False
        self._thread = threading.Thread(target=self._run, args=(work,), daemon=True)
        self._thread.start()

    def _run(self, work):
        try:
            self._box["result"] = work(self.is_cancelled)
        except BaseException as error:
            self._box["error"] = error

    def done(self):
        return not self._thread.is_alive()

    def result(self):
        """What the work returned, or what it raised, raised again here."""
        if "error" in self._box:
            raise self._box["error"]
        return self._box["result"]

    def cancel(self):
        self._cancelled = True

    def is_cancelled(self):
        return self._cancelled
