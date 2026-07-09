import argparse
import json
import time
from pathlib import Path

from .common import EXIT_INVALID_INPUT, UnwrapError, log, unwrap_all


def build_parser():
    parser = argparse.ArgumentParser(
        prog="uvgami", description="UV unwrap OBJ files with the UVgami OptCuts engine"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    unwrap = subparsers.add_parser("unwrap", help="unwrap OBJ files")
    unwrap.add_argument("input", type=Path, nargs="+", help="input OBJ files")
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

    args.quality = args.quality or "medium"
    args.import_uvs = bool(args.import_uvs)
    args.seam_weight = args.seam_weight or 3
    if args.seam_weights is not None and not args.seam_weights.is_file():
        raise UnwrapError(
            EXIT_INVALID_INPUT, f"seam weights file not found: {args.seam_weights}"
        )


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        validate(args)
        pairs = list(zip(args.input, args.outputs))
        start = time.perf_counter()
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
        elapsed = time.perf_counter() - start
    except UnwrapError as error:
        log(f"error: {error}", style="error")
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
        log(f"wrote {args.outputs[0]} in {elapsed:.1f}s", style="success")
        if args.json:
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "engine": "optcuts",
                        "input": str(args.input[0]),
                        "output": str(args.outputs[0]),
                        "seconds": round(elapsed, 2),
                    }
                )
            )
    else:
        log(
            f"batch finished in {elapsed:.1f}s, {code.ok} ok, {code.failed} failed",
            style="error" if code.failed else "success",
        )
    return code
