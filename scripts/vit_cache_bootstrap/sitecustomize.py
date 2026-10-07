"""Install the AVA ViT cache in LMDeploy's spawned engine interpreter."""

import os

if os.environ.get("AVA_VIT_TOKEN_CACHE_DIR"):
    from scripts.vit_cache_install import install_cache_hook

    install_cache_hook()
