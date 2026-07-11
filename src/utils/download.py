# Copyright (C) 2022 Daniel Boxer
# See __init__.py and LICENSE for more information

import http.client
import time
import urllib.error
import urllib.request
from pathlib import Path

# github releases honor Range, so a dropped transfer resumes from the .part
# file instead of restarting from zero (the ai checkpoint is over a gigabyte)
_CHUNK = 1 << 20
_ATTEMPTS = 4
_BACKOFF = 2.0


class DownloadError(RuntimeError):
    pass


def download_file(
    url, dest, attempts=_ATTEMPTS, timeout=30, backoff=_BACKOFF, progress=None
):
    """Download url to dest, resuming a partial .part across retries.

    streams into <dest>.part and atomically replaces dest on success. each
    retry resumes with a Range request when bytes are already on disk; a 200
    reply to that range restarts the file from scratch. progress, if given, is
    called as progress(done_bytes, total_bytes_or_None) with done_bytes counting
    from the start of the file, including any resumed offset.
    """
    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")
    error = None
    for attempt in range(attempts):
        if attempt:
            time.sleep(backoff * attempt)
        try:
            _fetch(url, part, timeout, progress)
            part.replace(dest)
            return
        except (OSError, http.client.IncompleteRead, DownloadError) as e:
            error = e
            # a full-size .part yields an unsatisfiable range; drop it to refetch
            if isinstance(e, urllib.error.HTTPError) and e.code == 416:
                part.unlink(missing_ok=True)
    raise DownloadError(f"failed to download {url} after {attempts} attempts: {error}")


def _fetch(url, part, timeout, progress=None):
    resume_from = part.stat().st_size if part.is_file() else 0
    request = urllib.request.Request(url)
    if resume_from:
        request.add_header("Range", f"bytes={resume_from}-")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        resumed = response.status == 206
        if not resumed:
            # server ignored the range (or none was sent), start the file over
            resume_from = 0
        length = response.getheader("Content-Length")
        expected = resume_from + int(length) if length is not None else None
        done = resume_from
        with open(part, "ab" if resumed else "wb") as file:
            while chunk := response.read(_CHUNK):
                file.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, expected)
    written = part.stat().st_size
    if expected is not None and written != expected:
        raise DownloadError(f"expected {expected} bytes from {url}, got {written}")
