import argparse
import json
import time
from pathlib import Path

from .common import EXIT_INVALID_INPUT, UnwrapError, log, unwrap_all

ENGINE_FLAGS = {
    "optcuts": ("quality", "import_uvs", "seam_weights", "seam_weight", "optcuts_path"),
    "partuv": ("threshold", "checkpoint", "config", "segmentation", "visual"),
}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="uvgami", description="UV unwrap OBJ files with the UVgami engines"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    unwrap = subparsers.add_parser("unwrap", help="unwrap OBJ files")
    unwrap.add_argument("input", type=Path, nargs="+", help="input OBJ files")
    unwrap.add_argument("--engine", choices=["optcuts", "partuv"], required=True)
    unwrap.add_argument(
        "-o",
        "--output",
        type=Path,
        action="append",
        help="output file, repeat once per input, default: <input stem>_uv.obj",
    )
    unwrap.add_argument(
        "--output-dir",
        type=Path,
        help="write each output as <input stem>.obj in this directory",
    )
    unwrap.add_argument("--overwrite", action="store_true", help="replace existing output")
    unwrap.add_argument(
        "--json",
        action="store_true",
        help="print a JSON result on stdout (single input only)",
    )

    # engine flag defaults are applied in validate() so that flags passed to the
    # wrong engine can be detected
    optcuts = unwrap.add_argument_group("optcuts options")
    optcuts.add_argument("--quality", choices=["high", "medium", "low"], help="default: medium")
    optcuts.add_argument(
        "--import-uvs", action="store_true", default=None, help="keep existing UVs as a starting point"
    )
    optcuts.add_argument("--seam-weights", type=Path, help="vertex weights file")
    optcuts.add_argument(
        "--seam-weight", type=int, choices=range(1, 6), help="seam weight level, default: 3"
    )
    optcuts.add_argument("--optcuts-path", type=Path, help="default: bundled binary")

    partuv = unwrap.add_argument_group("partuv options")
    partuv.add_argument("--threshold", type=float, help="distortion threshold, default: 1.25")
    partuv.add_argument(
        "--checkpoint",
        type=Path,
        help="PartField model checkpoint, default: $UVGAMI_PARTUV_CHECKPOINT"
        " or engine/partuv/model_objaverse.ckpt",
    )
    partuv.add_argument("--config", type=Path, help="default: engine/partuv/config/config.yaml")
    partuv.add_argument(
        "--segmentation",
        choices=["ai", "geometric"],
        help="part segmentation: ai (PartField, needs checkpoint + torch)"
        " or geometric (normals-based, no checkpoint), default: ai",
    )
    partuv.add_argument(
        "--visual",
        action="store_true",
        default=None,
        help="stream finished charts and progress on stdout for live viewing",
    )
    return parser


def validate(args):
    other = "partuv" if args.engine == "optcuts" else "optcuts"
    for name in ENGINE_FLAGS[other]:
        if getattr(args, name) is not None:
            flag = "--" + name.replace("_", "-")
            raise UnwrapError(
                EXIT_INVALID_INPUT, f"{flag} is only valid with --engine {other}"
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
    if args.json and len(args.input) > 1:
        raise UnwrapError(EXIT_INVALID_INPUT, "--json only supports a single input")
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

    if args.engine == "optcuts":
        args.quality = args.quality or "medium"
        args.import_uvs = bool(args.import_uvs)
        args.seam_weight = args.seam_weight or 3
        if args.seam_weights is not None and not args.seam_weights.is_file():
            raise UnwrapError(
                EXIT_INVALID_INPUT, f"seam weights file not found: {args.seam_weights}"
            )
    else:
        from . import partuv

        args.segmentation = args.segmentation or "ai"
        if args.segmentation == "ai":
            args.checkpoint = partuv.resolve_checkpoint(args.checkpoint)
        elif args.checkpoint is not None:
            raise UnwrapError(
                EXIT_INVALID_INPUT, "--checkpoint only applies to --segmentation ai"
            )
        args.threshold = args.threshold if args.threshold is not None else 1.25
        args.visual = bool(args.visual)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        validate(args)
        pairs = list(zip(args.input, args.outputs))
        start = time.perf_counter()
        if args.engine == "optcuts":
            from . import optcuts

            def unwrap_one(input_path, output_path):
                optcuts.run(
                    input_path,
                    output_path,
                    args.quality,
                    args.import_uvs,
                    args.seam_weights,
                    args.seam_weight,
                    args.optcuts_path,
                )

            code = unwrap_all(pairs, unwrap_one)
        else:
            from . import partuv

            code = partuv.run(
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
        if args.json:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "exit_code": error.exit_code,
                        "message": str(error),
                    }
                )
            )
        return error.exit_code

    if len(pairs) == 1:
        log(f"wrote {args.outputs[0]} in {elapsed:.1f}s")
        if args.json:
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "engine": args.engine,
                        "input": str(args.input[0]),
                        "output": str(args.outputs[0]),
                        "seconds": round(elapsed, 2),
                    }
                )
            )
    else:
        log(f"batch finished in {elapsed:.1f}s")
    return code
