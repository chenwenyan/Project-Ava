"""Join externally extracted codec scores to Qwen3-VL processed-image keys.

The input JSON must contain per-frame `key_index`, `motion`, and `residual`.
`residual` is intentionally not inferred from a decoded RGB difference: the
current PyAV path does not export H.264 transform coefficients.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

from scripts.vit_token_cache import VisionOutputStore


def make_patch_key(processor, path, dtype):
    with Image.open(path) as image:
        values = processor.image_processor(images=[image.convert("RGB")],
                                           return_tensors="pt")["pixel_values"]
    return VisionOutputStore.patch_key(values.to(dtype))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True,
                        help="JSON with frames[index]={key_index,motion,residual}")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    args = parser.parse_args()

    scores = json.loads(args.scores.read_text())
    records = scores["frames"]
    processor = AutoProcessor.from_pretrained(args.model)
    dtype = getattr(torch, args.dtype)
    keys = {}
    needed = {int(index) for index in records}
    needed.update(int(data["key_index"]) for data in records.values())
    for index in sorted(needed):
        path = args.frames_dir / f"{index}.jpg"
        if path.is_file():
            keys[index] = make_patch_key(processor, path, dtype)

    frames = {}
    for index_text, data in records.items():
        index = int(index_text)
        anchor = int(data["key_index"])
        if index == anchor or index not in keys or anchor not in keys:
            continue
        if "motion" not in data or "residual" not in data:
            raise ValueError(f"Frame {index} lacks motion or true codec residual score")
        frames[keys[index]] = {
            "anchor_patch_key": keys[anchor],
            "motion": float(data["motion"]),
            "residual": float(data["residual"]),
        }
    output = {"model": args.model, "dtype": args.dtype,
              "score_units": scores.get("score_units"), "frames": frames}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"Wrote {len(frames)} codec candidates to {args.output}")


if __name__ == "__main__":
    main()
