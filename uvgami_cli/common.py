import shutil
import sys
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
    # stdout is reserved for --json output
    print(message, file=sys.stderr, flush=True)


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
