"""Launch LMDeploy's API server with persistent post-ViT reuse enabled."""

import os
from pathlib import Path

from scripts.vit_cache_install import install_cache_hook


def main():
    from lmdeploy.cli import run

    # The API server spawns a separate PyTorch engine interpreter. sitecustomize
    # installs the model hook there before LMDeploy constructs the model.
    repo_root = Path(__file__).resolve().parents[1]
    bootstrap = Path(__file__).resolve().parent / "vit_cache_bootstrap"
    paths = [str(bootstrap), str(repo_root)]
    old_path = os.environ.get("PYTHONPATH")
    if old_path:
        paths.append(old_path)
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)
    install_cache_hook()
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
