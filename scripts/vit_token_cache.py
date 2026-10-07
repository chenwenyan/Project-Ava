"""Persistent post-ViT output reuse for LMDeploy Qwen3-VL's PyTorch engine.

The key is the exact *processed* input to one independent vision-attention
segment. This deliberately does not infer token equality from codec motion flags.
"""

import hashlib
import os
import tempfile
from pathlib import Path

import torch


def _update_tensor_hash(digest, tensor):
    value = tensor.detach().contiguous().cpu()
    digest.update(str(tuple(value.shape)).encode())
    digest.update(str(value.dtype).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())


class VisionOutputStore:
    """Disk-backed store for the full output tuple of independent ViT segments."""

    def __init__(self, cache_dir: Path, namespace: str):
        if not namespace:
            raise ValueError("A checkpoint/preprocessor namespace is required")
        self.root = Path(cache_dir) / hashlib.sha256(namespace.encode()).hexdigest()[:24]
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.vit_calls = 0

    def _path(self, patch, rotary, pos, merge_size):
        digest = hashlib.sha256()
        digest.update(f"qwen3vl-post-vit-v1;merge={merge_size}".encode())
        for value in (patch, rotary[0], rotary[1], pos):
            _update_tensor_hash(digest, value)
        return self.root / f"{digest.hexdigest()}.pt"

    @staticmethod
    def _write(path, record):
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
            temp_path = Path(tmp.name)
        try:
            torch.save(record, temp_path)
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

    def forward(self, model, original_forward, hidden_states, cu_seqlens, rotary_pos_emb, pos_embeds):
        if not isinstance(rotary_pos_emb, tuple) or len(rotary_pos_emb) != 2:
            raise ValueError("Qwen3-VL rotary positions must be a (cos, sin) tuple")
        ends = cu_seqlens.detach().cpu().tolist()
        starts = [0] + ends[:-1]
        if not ends or starts[0] != 0 or ends[-1] != len(hidden_states):
            raise ValueError("Invalid Qwen3-VL vision segment boundaries")
        merge_area = int(model.spatial_merge_size) ** 2

        entries = []
        missing = []
        for index, (start, end) in enumerate(zip(starts, ends)):
            if end <= start or (end - start) % merge_area:
                raise ValueError("Vision segment length is incompatible with spatial merge")
            patch = hidden_states[start:end]
            rotary = tuple(value[start:end] for value in rotary_pos_emb)
            pos = pos_embeds[start:end]
            path = self._path(patch, rotary, pos, model.spatial_merge_size)
            entry = {"index": index, "start": start, "end": end, "path": path}
            if path.exists():
                try:
                    record = torch.load(path, map_location="cpu", weights_only=True)
                    if not isinstance(record.get("main"), torch.Tensor):
                        raise ValueError("Incomplete cached vision record")
                    entry["record"] = record
                    self.hits += 1
                except (OSError, RuntimeError, ValueError, KeyError):
                    missing.append(entry)
                    self.misses += 1
            else:
                missing.append(entry)
                self.misses += 1
            entries.append(entry)

        if missing:
            patches = torch.cat([hidden_states[x["start"]:x["end"]] for x in missing])
            positions = torch.cat([pos_embeds[x["start"]:x["end"]] for x in missing])
            rotations = tuple(
                torch.cat([part[x["start"]:x["end"]] for x in missing])
                for part in rotary_pos_emb
            )
            lengths = [(x["end"] - x["start"]) for x in missing]
            boundaries = torch.tensor(lengths, device=cu_seqlens.device, dtype=cu_seqlens.dtype).cumsum(0)
            main, deepstack = original_forward(model, patches, boundaries, rotations, positions)
            self.vit_calls += 1
            sizes = [length // merge_area for length in lengths]
            if len(main) != sum(sizes) or any(len(level) != sum(sizes) for level in deepstack):
                raise ValueError("Unexpected Qwen3-VL post-ViT output shape")
            main_parts = torch.split(main, sizes)
            level_parts = [torch.split(level, sizes) for level in deepstack]
            for j, entry in enumerate(missing):
                record = {
                    "main": main_parts[j].detach().cpu(),
                    "deepstack": [level[j].detach().cpu() for level in level_parts],
                }
                self._write(entry["path"], record)
                entry["record"] = record

        device = hidden_states.device
        level_count = len(entries[0]["record"]["deepstack"])
        if any(len(entry["record"]["deepstack"]) != level_count for entry in entries):
            raise ValueError("Inconsistent cached DeepStack layer count")
        combined_main = torch.cat([entry["record"]["main"].to(device) for entry in entries])
        combined_levels = [
            torch.cat([entry["record"]["deepstack"][level].to(device) for entry in entries])
            for level in range(level_count)
        ]
        return combined_main, combined_levels


def install_into(vision_class, cache_dir: Path, namespace: str):
    """Wrap the LMDeploy vision class before the model is constructed."""
    if getattr(vision_class, "_ava_vit_cache_installed", False):
        raise RuntimeError("ViT output cache is already installed")
    store = VisionOutputStore(cache_dir, namespace)
    original_forward = vision_class.forward

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb, pos_embeds):
        return store.forward(self, original_forward, hidden_states, cu_seqlens, rotary_pos_emb, pos_embeds)

    vision_class.forward = forward
    vision_class._ava_vit_cache_installed = True
    return store
