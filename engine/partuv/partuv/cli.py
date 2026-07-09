import argparse
import os
import platform
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
    parser.add_argument("input", type=Path, nargs="+", help="input OBJ files")
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
    parser.add_argument("--overwrite", action="store_true", help="replace existing output")
    parser.add_argument("--threshold", type=float, help="distortion threshold, default: 1.25")
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


def validate(args):
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
    """Kept as str: a WSL-side checkpoint path would be mangled by WindowsPath."""
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


def run(pairs, checkpoint, config, threshold, segmentation="ai", visual=False):
    """Unwrap (input, output) pairs, returning the first failing exit code."""
    system = platform.system()
    if system == "Windows":
        # UVGAMI_PARTUV_WSL=1 forces the bridge even when the native build exists
        if os.environ.get("UVGAMI_PARTUV_WSL") or not _native_available():
            from . import wsl

            return wsl.run(pairs, checkpoint, config, threshold, segmentation, visual)
        log("using native Windows partuv")
    elif system != "Linux":
        raise UnwrapError(
            EXIT_MISSING_RUNTIME,
            "PartUV requires Linux with CUDA (use WSL on Windows)",
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

    model = None
    if segmentation == "ai":
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise UnwrapError(EXIT_MISSING_RUNTIME, f"checkpoint not found: {checkpoint}")
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
        log("loading PartField model")
        if visual:
            # tiny blue nudges so the bar leaves its full-red not-started state
            # long before the engine reports real face fractions
            emit("progress: 0.01 0 0.99")
        model = PFInferenceModel(checkpoint_path=str(checkpoint), device="cuda")

    def unwrap_one(input_path, output_path):
        with tempfile.TemporaryDirectory(prefix="uvgami-") as tmp:
            work = Path(tmp)
            log(f"preprocessing {input_path.name}")
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

            log("running PartUV pipeline")
            if visual:
                emit("progress: 0.05 0 0.95")
            final_part, individual_parts = pipeline_numpy(
                np.asarray(mesh.vertices, dtype=np.float64),
                np.asarray(mesh.faces, dtype=np.int32),
                tree_dict,
                str(config),
                threshold,
                visual=visual,
            )
            save_results(work, final_part, individual_parts)

            deliver(work / "final_components.obj", output_path)

    return unwrap_all(pairs, unwrap_one)


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
        log(f"error: {error}")
        return error.exit_code

    if len(pairs) == 1:
        log(f"wrote {args.outputs[0]} in {elapsed:.1f}s")
    else:
        log(f"batch finished in {elapsed:.1f}s")
    return code
