import os
import platform
import tempfile
from pathlib import Path

from .common import (
    EXIT_INVALID_INPUT,
    EXIT_MISSING_RUNTIME,
    REPO_ROOT,
    UnwrapError,
    deliver,
    log,
    unwrap_all,
)

DEFAULT_CONFIG = REPO_ROOT / "engine" / "partuv" / "config" / "config.yaml"
DEFAULT_CHECKPOINT = REPO_ROOT / "engine" / "partuv" / "model_objaverse.ckpt"
CHECKPOINT_URL = "https://huggingface.co/mikaelaangel/partfield-ckpt"


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
    try:
        import partuv  # noqa: F401

        return True
    except ImportError:
        return False


def run(pairs, checkpoint, config, threshold, segmentation="ai"):
    """Unwrap (input, output) pairs, returning the first failing exit code."""
    system = platform.system()
    if system == "Windows":
        # UVGAMI_PARTUV_WSL=1 forces the bridge even when the native build exists
        if os.environ.get("UVGAMI_PARTUV_WSL") or not _native_available():
            from . import wsl

            return wsl.run(pairs, checkpoint, config, threshold, segmentation)
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
        import partuv
        from partuv.output import save_results
    except ImportError as error:
        raise UnwrapError(
            EXIT_MISSING_RUNTIME,
            f"partuv is not installed ({error}), run: uv sync --extra partuv",
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
                "torch is not installed, run: uv sync --extra partuv"
                " (or use --segmentation geometric)",
            ) from error
        if not torch.cuda.is_available():
            raise UnwrapError(EXIT_MISSING_RUNTIME, "CUDA is not available")
        # import the function explicitly: the submodule import shadows the
        # package-level lazy `preprocess` attribute with the module itself
        from partuv.preprocess import PFInferenceModel
        from partuv.preprocess import preprocess as pf_preprocess

        # loaded once, shared by every mesh in the batch
        log("loading PartField model")
        model = PFInferenceModel(checkpoint_path=str(checkpoint), device="cuda")

    def unwrap_one(input_path, output_path):
        with tempfile.TemporaryDirectory(prefix="uvgami-") as tmp:
            work = Path(tmp)
            log(f"preprocessing {input_path.name}")
            if segmentation == "geometric":
                from partuv.geometric import preprocess_geometric

                mesh, tree_dict = preprocess_geometric(str(input_path))
            else:
                mesh, _, tree_dict, _ = pf_preprocess(
                    str(input_path),
                    pf_model=model,
                    output_path=str(work / "pre" / input_path.name),
                )

            log("running PartUV pipeline")
            final_part, individual_parts = partuv.pipeline_numpy(
                np.asarray(mesh.vertices, dtype=np.float64),
                np.asarray(mesh.faces, dtype=np.int32),
                tree_dict,
                str(config),
                threshold,
            )
            save_results(work, final_part, individual_parts)

            deliver(work / "final_components.obj", output_path)

    return unwrap_all(pairs, unwrap_one)
