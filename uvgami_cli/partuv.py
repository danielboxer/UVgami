import platform
import tempfile
from pathlib import Path

from .common import EXIT_MISSING_RUNTIME, REPO_ROOT, UnwrapError, deliver, log

DEFAULT_CONFIG = REPO_ROOT / "engine" / "partuv" / "config" / "config.yaml"


def run(input_path, output_path, checkpoint, config, threshold):
    if platform.system() != "Linux":
        raise UnwrapError(
            EXIT_MISSING_RUNTIME,
            "PartUV requires Linux with CUDA (use WSL on Windows)",
        )
    config = Path(config) if config is not None else DEFAULT_CONFIG
    if not config.is_file():
        raise UnwrapError(EXIT_MISSING_RUNTIME, f"PartUV config not found: {config}")
    if not checkpoint.is_file():
        raise UnwrapError(EXIT_MISSING_RUNTIME, f"checkpoint not found: {checkpoint}")

    try:
        import torch
    except ImportError as error:
        raise UnwrapError(
            EXIT_MISSING_RUNTIME,
            "torch is not installed, run: uv sync --extra partuv",
        ) from error
    if not torch.cuda.is_available():
        raise UnwrapError(EXIT_MISSING_RUNTIME, "CUDA is not available")

    try:
        import numpy as np
        import partuv
        from partuv.preprocess import PFInferenceModel, save_results
    except ImportError as error:
        raise UnwrapError(
            EXIT_MISSING_RUNTIME,
            f"partuv is not installed ({error}), run: uv sync --extra partuv",
        ) from error

    log("loading PartField model")
    model = PFInferenceModel(checkpoint_path=str(checkpoint), device="cuda")

    with tempfile.TemporaryDirectory(prefix="uvgami-") as tmp:
        work = Path(tmp)
        log("preprocessing mesh")
        mesh, _, tree_dict, _ = partuv.preprocess(
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
