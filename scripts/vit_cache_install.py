"""Install the cache on the visual class used by the live LMDeploy model."""

import os
from pathlib import Path

from scripts.vit_token_cache import install_into


def install_cache_hook():
    from lmdeploy.pytorch.models.qwen3_vl import Qwen3VLForConditionalGeneration

    if getattr(Qwen3VLForConditionalGeneration, "_ava_vit_cache_installed", False):
        return
    original_forward = Qwen3VLForConditionalGeneration.forward
    cache_dir = Path(os.environ.get("AVA_VIT_TOKEN_CACHE_DIR", "AVA_cache/vit_tokens"))
    namespace = os.environ.get("AVA_VIT_TOKEN_CACHE_NAMESPACE", "Qwen3-VL-8B-Instruct")
    metrics_path = Path(os.environ.get("AVA_VIT_TOKEN_CACHE_METRICS", str(cache_dir / "metrics.json")))
    sidecar = os.environ.get("AVA_VIT_CODEC_SIDECAR")

    def forward(self, *args, **kwargs):
        vision_class = type(self.visual)
        if not getattr(vision_class, "_ava_vit_cache_installed", False):
            storage_mode = os.environ.get("AVA_VIT_STORAGE", "disk")
            gpu_budget_bytes = int(float(os.environ.get("AVA_VIT_GPU_CACHE_GIB", "20")) * 2**30)
            host_budget_gib = os.environ.get("AVA_VIT_HOST_CACHE_GIB")
            host_budget_bytes = (int(float(host_budget_gib) * 2**30)
                                 if host_budget_gib is not None else None)
            host_reserve_bytes = int(float(os.environ.get("AVA_VIT_HOST_RESERVE_GIB", "32")) * 2**30)
            install_into(
                vision_class, cache_dir, namespace, metrics_path=metrics_path,
                codec_sidecar=Path(sidecar) if sidecar else None,
                mv_threshold=float(os.environ.get("AVA_VIT_MV_THRESHOLD", "0")),
                residual_threshold=float(os.environ.get("AVA_VIT_RESIDUAL_THRESHOLD", "0")),
                token_error_tolerance=float(os.environ.get("AVA_VIT_TOKEN_ERROR_TOLERANCE", "0")),
                storage_mode=storage_mode, gpu_budget_bytes=gpu_budget_bytes,
                host_budget_bytes=host_budget_bytes,
                host_reserve_bytes=host_reserve_bytes,
            )
        return original_forward(self, *args, **kwargs)

    Qwen3VLForConditionalGeneration.forward = forward
    Qwen3VLForConditionalGeneration._ava_vit_cache_installed = True
