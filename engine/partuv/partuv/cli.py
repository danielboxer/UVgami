import argparse
import ctypes
import functools
import os
import platform
import re
import tempfile
import time
from pathlib import Path

from .common import (
    EXIT_INVALID_INPUT,
    EXIT_MISSING_RUNTIME,
    UnwrapError,
    deliver,
    emit,
    log,
    unwrap_all,
)

# packaged next to this file by the wheel (see CMakeLists install)
DEFAULT_CONFIG = Path(__file__).parent / "config" / "config.yaml"
# engine/partuv/model_objaverse.ckpt in the source tree, absent in the wheel
DEFAULT_CHECKPOINT = Path(__file__).parents[1] / "model_objaverse.ckpt"
CHECKPOINT_URL = "https://huggingface.co/mikaelaangel/partfield-ckpt"


def build_parser():
    parser = argparse.ArgumentParser(
        prog="partuv", description="UV unwrap OBJ files with the PartUV engine"
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="*",
        help="input OBJ files (combined with --input-list if given)",
    )
    parser.add_argument(
        "--input-list",
        type=Path,
        help="text file with one input OBJ path per line, appended to the inputs",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        action="append",
        help="output file, repeat once per input, default: <input stem>_uv.obj",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write each output as <input stem>.obj in this directory",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing output"
    )
    parser.add_argument(
        "--threshold", type=float, help="distortion threshold, default: 1.25"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="PartField model checkpoint, default: $UVGAMI_PARTUV_CHECKPOINT"
        " or engine/partuv/model_objaverse.ckpt",
    )
    parser.add_argument("--config", type=Path, help="default: packaged config.yaml")
    parser.add_argument(
        "--segmentation",
        choices=["ai", "geometric"],
        help="part segmentation: ai (PartField, needs checkpoint + torch)"
        " or geometric (normals-based, no checkpoint), default: ai",
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        default=None,
        help="stream finished charts and progress on stdout for live viewing",
    )
    return parser


def read_input_list(path):
    """Input OBJ paths from a list file, one per line, blank lines skipped.
    Read here on the main thread, not via a stdin reader: a thread blocked on a
    stdin pipe stalls native module imports for minutes on windows."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise UnwrapError(
            EXIT_INVALID_INPUT, f"input list not found: {path}"
        ) from error
    return [Path(line.strip()) for line in text.splitlines() if line.strip()]


def validate(args):
    inputs = list(args.input or [])
    if args.input_list is not None:
        inputs += read_input_list(args.input_list)
    args.input = inputs
    if not inputs:
        raise UnwrapError(
            EXIT_INVALID_INPUT, "no input: pass an OBJ file or --input-list"
        )
    for input_path in args.input:
        # in a batch a missing input fails per mesh instead, so a cancelled
        # mesh (its input file is deleted) doesn't abort the rest
        if len(args.input) == 1 and not input_path.is_file():
            raise UnwrapError(EXIT_INVALID_INPUT, f"input not found: {input_path}")
        if input_path.suffix.lower() != ".obj":
            raise UnwrapError(
                EXIT_INVALID_INPUT, f"input must be an OBJ file: {input_path}"
            )
    if args.output and args.output_dir:
        raise UnwrapError(
            EXIT_INVALID_INPUT, "-o and --output-dir are mutually exclusive"
        )
    if args.output and len(args.output) != len(args.input):
        raise UnwrapError(EXIT_INVALID_INPUT, "-o must be given once per input")
    if args.output_dir is not None:
        args.outputs = [args.output_dir / f"{p.stem}.obj" for p in args.input]
    elif args.output:
        args.outputs = list(args.output)
    else:
        args.outputs = [p.with_name(f"{p.stem}_uv.obj") for p in args.input]
    if len(set(args.outputs)) != len(args.outputs):
        raise UnwrapError(EXIT_INVALID_INPUT, "output paths collide, rename the inputs")
    for output_path in args.outputs:
        if output_path.exists() and not args.overwrite:
            raise UnwrapError(
                EXIT_INVALID_INPUT, f"output exists (use --overwrite): {output_path}"
            )

    args.segmentation = args.segmentation or "ai"
    if args.segmentation == "ai":
        args.checkpoint = resolve_checkpoint(args.checkpoint)
    elif args.checkpoint is not None:
        raise UnwrapError(
            EXIT_INVALID_INPUT, "--checkpoint only applies to --segmentation ai"
        )
    args.threshold = args.threshold if args.threshold is not None else 1.25
    args.visual = bool(args.visual)


def resolve_checkpoint(flag_value):
    """Kept as str: a raw path in $UVGAMI_PARTUV_CHECKPOINT would be mangled by
    WindowsPath."""
    if flag_value is not None:
        return str(flag_value)
    named = os.environ.get("UVGAMI_PARTUV_CHECKPOINT")
    if named:
        return named
    if DEFAULT_CHECKPOINT.is_file():
        return str(DEFAULT_CHECKPOINT)
    raise UnwrapError(
        EXIT_INVALID_INPUT,
        "no PartField checkpoint: pass --checkpoint, set UVGAMI_PARTUV_CHECKPOINT,"
        f" or place model_objaverse.ckpt in engine/partuv/ (download: {CHECKPOINT_URL})",
    )


def _native_available():
    from . import _CORE_ERROR

    return _CORE_ERROR is None


@functools.lru_cache(maxsize=1)
def _cuda_available():
    """Probe the nvidia driver directly so the geometric path can decide on a
    cpu fallback without importing torch."""
    if platform.system() == "Windows":
        loader, name = ctypes.WinDLL, "nvcuda.dll"
    else:
        loader, name = ctypes.CDLL, "libcuda.so.1"
    try:
        return loader(name).cuInit(0) == 0
    except OSError:
        return False


def _config_without_pamo(config):
    """Write a copy of config with the cuda mesh simplifier off, returning the
    temp path. We own the format, so patch the pamo line without a yaml dep."""
    text = re.sub(r"(?m)^(\s*pamo:\s*)true\b", r"\1false", config.read_text())
    handle, name = tempfile.mkstemp(prefix="uvgami-config-", suffix=".yaml")
    with os.fdopen(handle, "w") as file:
        file.write(text)
    return Path(name)


def run(pairs, checkpoint, config, threshold, segmentation="ai", visual=False):
    """Unwrap (input, output) pairs, returning the first failing exit code."""
    system = platform.system()
    if system == "Windows":
        if not _native_available():
            from . import _CORE_ERROR

            log(f"native partuv failed to load: {_CORE_ERROR}")
            raise UnwrapError(
                EXIT_MISSING_RUNTIME,
                f"the PartUV engine failed to load natively ({_CORE_ERROR});"
                " reinstall PartUV in the add-on preferences",
            )
    elif system != "Linux":
        raise UnwrapError(
            EXIT_MISSING_RUNTIME,
            "PartUV requires Windows or Linux with CUDA",
        )
    config = Path(config) if config is not None else DEFAULT_CONFIG
    if not config.is_file():
        raise UnwrapError(EXIT_MISSING_RUNTIME, f"PartUV config not found: {config}")

    try:
        import numpy as np

        from ._core import pipeline_numpy
        from .output import save_results
    except ImportError as error:
        raise UnwrapError(
            EXIT_MISSING_RUNTIME,
            f"partuv is not installed ({error}), reinstall PartUV in the add-on"
            " preferences, or in dev: uv sync --extra partuv",
        ) from error

    # the geometric path runs the cuda mesh simplifier (pamo) on big components;
    # with no nvidia driver disable it so unwrapping stays on the cpu branch.
    # the ai path is already covered by the torch.cuda check above.
    cpu_config = None
    if segmentation == "geometric" and not _cuda_available():
        log("no NVIDIA GPU detected, unwrapping on CPU (slower)")
        config = cpu_config = _config_without_pamo(config)

    model = None
    if segmentation == "ai":
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise UnwrapError(
                EXIT_MISSING_RUNTIME, f"checkpoint not found: {checkpoint}"
            )
        try:
            import torch
        except ImportError as error:
            raise UnwrapError(
                EXIT_MISSING_RUNTIME,
                "torch is not installed, reinstall PartUV in the add-on preferences,"
                " or in dev: uv sync --extra partuv (or use --segmentation geometric)",
            ) from error
        if not torch.cuda.is_available():
            raise UnwrapError(EXIT_MISSING_RUNTIME, "CUDA is not available")
        # import the function explicitly: the submodule import shadows the
        # package-level lazy `preprocess` attribute with the module itself
        from .preprocess import PFInferenceModel
        from .preprocess import preprocess as pf_preprocess

        # loaded once, shared by every mesh in the batch
        log("loading PartField model", style="step")
        if visual:
            # tiny blue nudges so the bar leaves its full-red not-started state
            # long before the engine reports real face fractions
            emit("progress: 0.01 0 0.99")
        model = PFInferenceModel(checkpoint_path=str(checkpoint), device="cuda")

    def unwrap_one(input_path, output_path):
        with tempfile.TemporaryDirectory(prefix="uvgami-") as tmp:
            work = Path(tmp)
            log(f"preprocessing {input_path.name}", style="step")
            if visual:
                emit("progress: 0.02 0 0.98")
            if segmentation == "geometric":
                from .geometric import preprocess_geometric

                mesh, tree_dict = preprocess_geometric(str(input_path))
            else:
                mesh, _, tree_dict, _ = pf_preprocess(
                    str(input_path),
                    pf_model=model,
                    output_path=str(work / "pre" / input_path.name),
                )

            log("running PartUV pipeline", style="step")
            if visual:
                emit("progress: 0.05 0 0.95")
            source_V = np.asarray(mesh.vertices, dtype=np.float64)
            source_F = np.asarray(mesh.faces, dtype=np.int32)
            final_part, individual_parts = pipeline_numpy(
                source_V,
                source_F,
                tree_dict,
                str(config),
                threshold,
                visual=visual,
            )
            save_results(work, final_part, individual_parts, source_V, source_F)

            deliver(work / "final_components.obj", output_path)

    try:
        return unwrap_all(pairs, unwrap_one)
    finally:
        if cpu_config is not None:
            cpu_config.unlink(missing_ok=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        validate(args)
        pairs = list(zip(args.input, args.outputs))
        start = time.perf_counter()
        code = run(
            pairs,
            args.checkpoint,
            args.config,
            args.threshold,
            args.segmentation,
            args.visual,
        )
        elapsed = time.perf_counter() - start
    except UnwrapError as error:
        log(f"error: {error}", style="error")
        return error.exit_code

    if len(pairs) == 1:
        log(f"wrote {args.outputs[0]} in {elapsed:.1f}s", style="success")
    else:
        log(
            f"batch finished in {elapsed:.1f}s, {code.ok} ok, {code.failed} failed",
            style="error" if code.failed else "success",
        )
    return code
