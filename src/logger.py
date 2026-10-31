import platform
import time
import traceback

import bpy

from .utils.paths import get_extension_dir_path, read_manifest

HEADER_PREFIX = "UVgami "
LOG_LINE_LIMIT = 500
UNSAVED_FILE_NAME = "unsaved"


def get_log_path():
    return get_extension_dir_path() / "log.txt"


def block_header():
    name = bpy.path.basename(bpy.data.filepath) or UNSAVED_FILE_NAME
    return (
        f"{HEADER_PREFIX}{read_manifest()['version']} | Blender {bpy.app.version_string}"
        f" | {platform.system()} | {time.strftime('%Y-%m-%d %H:%M:%S')} | {name}"
    )


def split_blocks(content):
    blocks = []
    for line in content.splitlines():
        if line.startswith(HEADER_PREFIX):
            blocks.append([])
        if blocks and line:
            blocks[-1].append(line)
    return blocks


def trim_blocks(blocks):
    kept = []
    lines = 0
    for block in reversed(blocks):
        if kept and lines + len(block) > LOG_LINE_LIMIT:
            break
        kept.append(block)
        lines += len(block)
    kept.reverse()
    return kept


def _logged_blocks_with(header, lines):
    path = get_log_path()
    blocks = split_blocks(path.read_text(encoding="utf-8")) if path.is_file() else []
    if not lines:
        return blocks
    if not blocks or blocks[-1][0] != header:
        blocks.append([header])
    blocks[-1].extend(lines)
    return blocks


class Info:
    def __init__(self):
        self.started = time.strftime("%H:%M:%S")
        self.time = 0
        self.pre_processing_time = 0
        self.unwrap_time = 0
        self.post_processing_time = 0
        self.errors = []
        self.status = "In Progress"
        self.written_to_file = False
        self.objects = []
        self.piece_counts = {}
        self.engine = ""
        self.settings = ""

    def get_time(self):
        text = f"{self.time:.2f}s"
        if self.unwrap_time:
            text += (
                f" (pre-processing {self.pre_processing_time:.2f}s,"
                f" unwrap {self.unwrap_time:.2f}s,"
                f" post-processing {self.post_processing_time:.2f}s)"
            )
        return text

    def get_objects(self):
        names = []
        for name, count in self.objects:
            if count >= 1_000_000:
                size = f"{count / 1_000_000:.1f}M"
            elif count >= 1000:
                size = f"{count / 1000:.1f}k"
            else:
                size = str(count)
            names.append(f"{name} ({size} tris)")
        return ", ".join(names)

    def get_pieces(self):
        counts = ", ".join(
            f"{count} {result.value}"
            for result, count in self.piece_counts.items()
            if count
        )
        return f"Pieces: {counts}" if counts else ""

    def get_info(self):
        fields = [self.started, self.status, self.get_time()]
        if self.engine:
            fields.append(self.engine)
        fields.append(self.get_objects())
        pieces = self.get_pieces()
        if pieces:
            fields.append(pieces)
        if self.settings:
            fields.append(f"Settings: {self.settings}")
        if self.errors:
            fields.append(f"Errors: {'; '.join(self.errors)}")
        return " | ".join(fields)


class Logger:
    def __init__(self):
        self.unwrap_info = []
        self.start_time = 0
        self.unwrap_start = None
        self.unwrap_end = None
        self.header = None

    def reset(self):
        self.unwrap_info.clear()
        self.header = None

    def new_info(self):
        info = Info()
        self.unwrap_info.append(info)
        self.start_time = time.perf_counter()
        self.unwrap_start = None
        self.unwrap_end = None
        return info

    def discard_refused_run(self):
        self.unwrap_info.pop()

    def add_error(self, text):
        self.get_latest().errors.append(text)

    def log_traceback(self, heading):
        self.add_error(heading)
        lines = traceback.format_exc().split("\n")[:-1]
        for line in lines:
            self.add_error(line)
            print(line)
        return lines

    def get_latest(self):
        return self.unwrap_info[-1]

    def get_all(self):
        return [info.get_info() for info in self.unwrap_info]

    def write_latest(self):
        if self.header is None:
            self.header = block_header()
        info = self.get_latest()
        blocks = _logged_blocks_with(self.header, [info.get_info()])
        info.written_to_file = True
        body = "\n\n".join("\n".join(block) for block in trim_blocks(blocks))
        # crashes and file loads lose the in-memory list
        get_log_path().write_text(body + "\n", encoding="utf-8")

    def log_with_unfinished_runs(self):
        pending = [
            info.get_info() for info in self.unwrap_info if not info.written_to_file
        ]
        blocks = _logged_blocks_with(self.header or block_header(), pending)
        return "\n\n".join("\n".join(block) for block in blocks)

    def mark_unwrapping(self):
        self.unwrap_end = time.perf_counter()
        if self.unwrap_start is None:
            self.unwrap_start = self.unwrap_end

    def update_time(self):
        info = self.get_latest()
        info.time = time.perf_counter() - self.start_time
        if self.unwrap_start is None:
            info.pre_processing_time = info.time
            return
        info.pre_processing_time = self.unwrap_start - self.start_time
        info.unwrap_time = self.unwrap_end - self.unwrap_start
        info.post_processing_time = (
            info.time - info.pre_processing_time - info.unwrap_time
        )


logger = Logger()
