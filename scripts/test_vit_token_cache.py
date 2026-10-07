"""Focused checks for the Qwen3-VL visual-output interception."""

import argparse
import json
from pathlib import Path

import torch

from scripts.vit_token_cache import install_into


def check_spine():
    class ToyVision:
        calls = 0

        def forward(self, hidden_states, cu_seqlens, rotary_pos_emb, pos_embeds):
            self.calls += 1
            return hidden_states + pos_embeds, [hidden_states * 2]

    install_into(ToyVision, Path("implement-stage/.f0_cache"), "toy")
    model = ToyVision()
    data = torch.ones(2, 2)
    output = model.forward(data, torch.tensor([2]), (data, data), data)
    assert model.calls == 1
    assert torch.equal(output[0], data * 2)
    out = Path("implement-stage/f0_smoke.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"intercepted": True, "upstream_calls": model.calls}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spine", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--integration", action="store_true")
    parser.add_argument("--codec", action="store_true")
    args = parser.parse_args()
    if args.spine:
        check_spine()
    else:
        raise SystemExit("Only --spine is implemented at F0")


if __name__ == "__main__":
    main()
