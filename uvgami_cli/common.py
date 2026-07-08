import shutil
import sys
import traceback
from pathlib import Path

EXIT_INVALID_INPUT = 2
EXIT_MISSING_RUNTIME = 3
EXIT_ENGINE_FAILURE = 4
EXIT_BAD_OUTPUT = 5

REPO_ROOT = Path(__file__).resolve().parents[1]


class UnwrapError(Exception):
    def __init__(self, exit_code, message):
        super().__init__(message)
        self.exit_code = exit_code


def log(message):
    # stdout is reserved for --json output and batch markers
    print(message, file=sys.stderr, flush=True)


def emit(message):
    """Machine-readable stdout marker consumed by the Blender add-on."""
    print(message, flush=True)


def unwrap_all(pairs, unwrap_one):
    """Unwrap each (input, output) pair and return the first failing exit code.

    With multiple pairs, failures are isolated per mesh and start/done/failed
    markers are emitted so a caller can track progress per mesh. Deleting an
    input file mid-batch cancels that mesh: it fails fast with no compute.
    No stdin watcher for this on purpose: a thread blocked reading a stdin
    pipe stalls native module imports for minutes on windows."""
    batch = len(pairs) > 1
    first_code = 0
    for input_path, output_path in pairs:
        if batch:
            emit(f"start: {input_path.stem}")
        try:
            if not input_path.is_file():
                raise UnwrapError(
                    EXIT_INVALID_INPUT, f"input not found: {input_path}"
                )
            unwrap_one(input_path, output_path)
        except Exception as error:
            if not batch:
                raise
            if isinstance(error, UnwrapError):
                code = error.exit_code
            else:
                code = EXIT_ENGINE_FAILURE
                log(traceback.format_exc())
            log(f"error: {input_path.name}: {error}")
            emit(f"failed: {input_path.stem} {code}")
            if first_code == 0:
                first_code = code
        else:
            if batch:
                emit(f"done: {input_path.stem}")
    return first_code


def validate_uv_obj(path):
    if not path.is_file():
        raise UnwrapError(EXIT_BAD_OUTPUT, f"engine produced no output: {path}")
    found = {"v": False, "vt": False, "f": False}
    with open(path) as file:
        for line in file:
            key = line.split(maxsplit=1)[0] if line.strip() else ""
            if key in found:
                found[key] = True
                if all(found.values()):
                    return
    missing = ", ".join(key for key, seen in found.items() if not seen)
    raise UnwrapError(EXIT_BAD_OUTPUT, f"output OBJ is missing data: {missing}")


def deliver(result_path, output_path):
    """Validate the engine result and move it to the requested output."""
    validate_uv_obj(result_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(result_path), str(output_path))
