"""Launch LMDeploy's regular CLI with a Qwen3-VL ViT interception hook."""

import os
from pathlib import Path

from scripts.vit_token_cache import install_into


def main():
    from lmdeploy.cli import run
    from lmdeploy.pytorch.models.qwen3_vl import Qwen3VLVisionModel

    cache_dir = Path(os.environ.get("AVA_VIT_TOKEN_CACHE_DIR", "AVA_cache/vit_tokens"))
    namespace = os.environ.get("AVA_VIT_TOKEN_CACHE_NAMESPACE", "Qwen3-VL-8B-Instruct")
    metrics_path = Path(os.environ.get("AVA_VIT_TOKEN_CACHE_METRICS", str(cache_dir / "metrics.json")))
    sidecar = os.environ.get("AVA_VIT_CODEC_SIDECAR")
    install_into(
        Qwen3VLVisionModel, cache_dir, namespace, metrics_path=metrics_path,
        codec_sidecar=Path(sidecar) if sidecar else None,
        mv_threshold=float(os.environ.get("AVA_VIT_MV_THRESHOLD", "0")),
        residual_threshold=float(os.environ.get("AVA_VIT_RESIDUAL_THRESHOLD", "0")),
        token_error_tolerance=float(os.environ.get("AVA_VIT_TOKEN_ERROR_TOLERANCE", "0")),
    )
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
