"""Focused checks for the Qwen3-VL visual-output interception."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from scripts.vit_token_cache import install_into


def check_spine():
    class ToyVision:
        calls = 0
        spatial_merge_size = 1

        def forward(self, hidden_states, cu_seqlens, rotary_pos_emb, pos_embeds):
            self.calls += 1
            return hidden_states + pos_embeds, [hidden_states * 2]

    with TemporaryDirectory() as directory:
        store = install_into(ToyVision, Path(directory), "toy")
        model = ToyVision()
        data = torch.ones(2, 2)
        output = model.forward(data, torch.tensor([2]), (data, data), data)
        assert model.calls == 1 and store.vit_calls == 1
        assert torch.equal(output[0], data * 2)
    out = Path("implement-stage/f0_smoke.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"intercepted": True, "upstream_calls": model.calls}))


def check_cache():
    class ToyVision:
        spatial_merge_size = 2
        calls = 0

        def forward(self, hidden_states, cu_seqlens, rotary_pos_emb, pos_embeds):
            self.calls += 1
            starts = [0] + cu_seqlens.tolist()[:-1]
            main = []
            deep = []
            for start, end in zip(starts, cu_seqlens.tolist()):
                x = hidden_states[start:end] + pos_embeds[start:end]
                main.append(x.reshape(-1, 4, x.shape[-1]).mean(1))
                deep.append((x * 2).reshape(-1, 4, x.shape[-1]).mean(1))
            return torch.cat(main), [torch.cat(deep)]

    with TemporaryDirectory() as directory:
        original_forward = ToyVision.forward
        store = install_into(ToyVision, Path(directory), "toy-v1")
        model = ToyVision()
        x = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        pos = torch.ones_like(x)
        rotation = (torch.zeros_like(x), torch.ones_like(x))
        ends = torch.tensor([4, 8])
        first = model.forward(x, ends, rotation, pos)
        assert model.calls == 1 and store.misses == 2
        second = model.forward(x, ends, rotation, pos)
        assert model.calls == 1 and store.hits == 2
        assert torch.equal(first[0], second[0])
        assert torch.equal(first[1][0], second[1][0])
        x2 = x.clone()
        x2[4, 0] += 1
        model.forward(x2, ends, rotation, pos)
        assert model.calls == 2 and store.hits == 3 and store.misses == 3
        # A fresh store instance must read previously persisted records.
        from scripts.vit_token_cache import VisionOutputStore
        restarted = VisionOutputStore(Path(directory), "toy-v1")
        output = restarted.forward(model, original_forward, x, ends, rotation, pos)
        assert restarted.hits == 2 and restarted.vit_calls == 0
        assert torch.equal(first[0], output[0])


def check_codec():
    from scripts.vit_token_cache import VisionOutputStore

    class ToyVision:
        spatial_merge_size = 2
        calls = 0

        def forward(self, hidden_states, cu_seqlens, rotary_pos_emb, pos_embeds):
            self.calls += 1
            main = hidden_states[:, :1].reshape(-1, 4, 1).mean(1)
            return main, [main * 2]

    anchor = torch.ones(4, 2)
    delta = anchor.clone()
    delta[:, 1] += 1  # Different processed patches; same toy post-ViT tokens.
    pos = torch.zeros_like(anchor)
    rotation = (pos, pos)
    boundary = torch.tensor([4])
    with TemporaryDirectory() as directory:
        root = Path(directory)
        key_anchor = VisionOutputStore.patch_key(anchor)
        key_delta = VisionOutputStore.patch_key(delta)
        sidecar = root / "codec.json"
        sidecar.write_text(json.dumps({"frames": {
            key_delta: {"anchor_patch_key": key_anchor, "motion": 0.05, "residual": 0.1},
        }}))
        model = ToyVision()
        store = VisionOutputStore(root / "accepted", "toy", codec_sidecar=sidecar,
                                  mv_threshold=0.05, residual_threshold=0.1)
        store.forward(model, ToyVision.forward, anchor, boundary, rotation, pos)
        store.forward(model, ToyVision.forward, delta, boundary, rotation, pos)
        assert store.aliases == 1
        assert store._relative_error(store._load(store._anchor_path(key_delta)),
                                     store._load(store._anchor_path(key_anchor))) == 0
        before = model.calls
        store.forward(model, ToyVision.forward, delta, boundary, rotation, pos)
        assert model.calls == before and store.hits == 1

        rejected = VisionOutputStore(root / "rejected", "toy", codec_sidecar=sidecar,
                                     mv_threshold=0.04, residual_threshold=0.1)
        rejected.forward(model, ToyVision.forward, anchor, boundary, rotation, pos)
        rejected.forward(model, ToyVision.forward, delta, boundary, rotation, pos)
        assert rejected.aliases == 0

        changed = anchor.clone()
        changed[:, 0] += 1
        changed_key = VisionOutputStore.patch_key(changed)
        guarded_sidecar = root / "guarded.json"
        guarded_sidecar.write_text(json.dumps({"frames": {
            changed_key: {"anchor_patch_key": key_anchor, "motion": 0, "residual": 0},
        }}))
        guarded = VisionOutputStore(root / "guarded", "toy", codec_sidecar=guarded_sidecar)
        guarded.forward(model, ToyVision.forward, anchor, boundary, rotation, pos)
        guarded.forward(model, ToyVision.forward, changed, boundary, rotation, pos)
        assert guarded.aliases == 0

        approximate = VisionOutputStore(root / "approximate", "toy",
                                        codec_sidecar=guarded_sidecar,
                                        token_error_tolerance=0.6)
        approximate.forward(model, ToyVision.forward, anchor, boundary, rotation, pos)
        fresh_main, _ = approximate.forward(model, ToyVision.forward, changed,
                                            boundary, rotation, pos)
        reused_main, _ = approximate.forward(model, ToyVision.forward, changed,
                                             boundary, rotation, pos)
        assert approximate.aliases == 1
        assert not torch.equal(fresh_main, reused_main)
        assert torch.equal(reused_main, torch.ones_like(reused_main))


def check_integration():
    import inspect
    import subprocess
    import sys

    from lmdeploy.pytorch.models.qwen3_vl import Qwen3VLVisionModel

    params = list(inspect.signature(Qwen3VLVisionModel.forward).parameters)
    assert params == ["self", "hidden_states", "cu_seqlens", "rotary_pos_emb", "pos_embeds"]
    result = subprocess.run([sys.executable, "-m", "scripts.serve_vit_token_cache", "--help"],
                            capture_output=True, text=True, check=True)
    assert "lmdeploy" in result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spine", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--integration", action="store_true")
    parser.add_argument("--codec", action="store_true")
    args = parser.parse_args()
    if args.spine:
        check_spine()
    elif args.cache:
        check_cache()
    elif args.codec:
        check_codec()
    elif args.integration:
        check_integration()
    else:
        raise SystemExit("Only --spine is implemented at F0")


if __name__ == "__main__":
    main()
