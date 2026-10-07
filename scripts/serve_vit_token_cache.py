"""Launch LMDeploy's regular CLI with a Qwen3-VL ViT interception hook."""

import os
from pathlib import Path

from scripts.vit_token_cache import install_into


def main():
    from lmdeploy.cli import run
    from lmdeploy.pytorch.models.qwen3_vl import Qwen3VLVisionModel

    cache_dir = Path(os.environ.get("AVA_VIT_TOKEN_CACHE_DIR", "AVA_cache/vit_tokens"))
    namespace = os.environ.get("AVA_VIT_TOKEN_CACHE_NAMESPACE", "Qwen3-VL-8B-Instruct")
    install_into(Qwen3VLVisionModel, cache_dir, namespace)
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
