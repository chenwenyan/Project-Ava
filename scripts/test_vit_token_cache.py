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
    else:
        raise SystemExit("Only --spine is implemented at F0")


if __name__ == "__main__":
    main()
