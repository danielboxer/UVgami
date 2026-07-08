import argparse
import json
import time
from pathlib import Path

from .common import EXIT_INVALID_INPUT, UnwrapError, log

ENGINE_FLAGS = {
    "optcuts": ("quality", "import_uvs", "seam_weights", "seam_weight", "optcuts_path"),
    "partuv": ("threshold", "checkpoint", "config", "segmentation"),
}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="uvgami", description="UV unwrap OBJ files with the UVgami engines"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    unwrap = subparsers.add_parser("unwrap", help="unwrap an OBJ file")
    unwrap.add_argument("input", type=Path, help="input OBJ file")
    unwrap.add_argument("--engine", choices=["optcuts", "partuv"], required=True)
    unwrap.add_argument("-o", "--output", type=Path, help="default: <input stem>_uv.obj")
    unwrap.add_argument("--overwrite", action="store_true", help="replace existing output")
    unwrap.add_argument("--json", action="store_true", help="print a JSON result on stdout")

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
    return parser


def validate(args):
    other = "partuv" if args.engine == "optcuts" else "optcuts"
    for name in ENGINE_FLAGS[other]:
        if getattr(args, name) is not None:
            flag = "--" + name.replace("_", "-")
            raise UnwrapError(
                EXIT_INVALID_INPUT, f"{flag} is only valid with --engine {other}"
            )

    if not args.input.is_file():
        raise UnwrapError(EXIT_INVALID_INPUT, f"input not found: {args.input}")
    if args.input.suffix.lower() != ".obj":
        raise UnwrapError(EXIT_INVALID_INPUT, f"input must be an OBJ file: {args.input}")
    if args.output is None:
        args.output = args.input.with_name(f"{args.input.stem}_uv.obj")
    if args.output.exists() and not args.overwrite:
        raise UnwrapError(
            EXIT_INVALID_INPUT, f"output exists (use --overwrite): {args.output}"
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


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        validate(args)
        start = time.perf_counter()
        if args.engine == "optcuts":
            from . import optcuts

            optcuts.run(
                args.input,
                args.output,
                args.quality,
                args.import_uvs,
                args.seam_weights,
                args.seam_weight,
                args.optcuts_path,
            )
        else:
            from . import partuv

            partuv.run(
                args.input,
                args.output,
                args.checkpoint,
                args.config,
                args.threshold,
                args.segmentation,
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

    log(f"wrote {args.output} in {elapsed:.1f}s")
    if args.json:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "engine": args.engine,
                    "input": str(args.input),
                    "output": str(args.output),
                    "seconds": round(elapsed, 2),
                }
            )
        )
    return 0
