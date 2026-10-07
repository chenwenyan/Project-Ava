"""Persistent post-ViT output reuse for LMDeploy Qwen3-VL's PyTorch engine.

The key is the exact *processed* input to one independent vision-attention
segment. This deliberately does not infer token equality from codec motion flags.
"""

import hashlib
import json
import math
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

    def __init__(self, cache_dir: Path, namespace: str, metrics_path: Path = None,
                 codec_sidecar: Path = None, mv_threshold: float = 0.0,
                 residual_threshold: float = 0.0, token_error_tolerance: float = 0.0):
        if not namespace:
            raise ValueError("A checkpoint/preprocessor namespace is required")
        self.mv_threshold = float(mv_threshold)
        self.residual_threshold = float(residual_threshold)
        self.token_error_tolerance = float(token_error_tolerance)
        if not all(math.isfinite(value) and value >= 0 for value in
                   (self.mv_threshold, self.residual_threshold, self.token_error_tolerance)):
            raise ValueError("Codec and token-error thresholds must be nonnegative")
        self.codec_signals = {}
        policy_identity = namespace
        if codec_sidecar:
            sidecar_bytes = Path(codec_sidecar).read_bytes()
            data = json.loads(sidecar_bytes)
            self.codec_signals = data["frames"]
            policy_identity += ";sidecar=" + hashlib.sha256(sidecar_bytes).hexdigest()
            policy_identity += f";mv={self.mv_threshold};res={self.residual_threshold}"
            policy_identity += f";error={self.token_error_tolerance}"
        self.root = Path(cache_dir) / hashlib.sha256(policy_identity.encode()).hexdigest()[:24]
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.vit_calls = 0
        self.aliases = 0
        self.metrics_path = Path(metrics_path) if metrics_path else None
        (self.root / "patch_index").mkdir(exist_ok=True)

    def _record_metrics(self):
        if self.metrics_path is None:
            return
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"hits": self.hits, "misses": self.misses,
                   "vit_calls": self.vit_calls, "aliases": self.aliases}
        with tempfile.NamedTemporaryFile(mode="w", dir=self.metrics_path.parent,
                                         suffix=".tmp", delete=False) as tmp:
            json.dump(payload, tmp)
            temp_path = Path(tmp.name)
        os.replace(temp_path, self.metrics_path)

    def _path(self, patch, rotary, pos, merge_size):
        digest = hashlib.sha256()
        digest.update(f"qwen3vl-post-vit-v1;merge={merge_size}".encode())
        for value in (patch, rotary[0], rotary[1], pos):
            _update_tensor_hash(digest, value)
        return self.root / f"{digest.hexdigest()}.pt"

    @staticmethod
    def patch_key(patch):
        digest = hashlib.sha256()
        _update_tensor_hash(digest, patch)
        return digest.hexdigest()

    def _index_patch(self, patch_key, path):
        index_path = self.root / "patch_index" / patch_key
        index_path.write_text(path.name)

    def _anchor_path(self, patch_key):
        index_path = self.root / "patch_index" / patch_key
        if not index_path.exists():
            return None
        name = index_path.read_text().strip()
        if not name.endswith(".pt") or "/" in name or ".." in name:
            return None
        path = self.root / name
        return path if path.exists() else None

    def _load(self, path):
        for _ in range(8):
            record = torch.load(path, map_location="cpu", weights_only=True)
            if "alias" not in record:
                break
            name = record["alias"]
            if not isinstance(name, str) or "/" in name or ".." in name:
                raise ValueError("Invalid vision alias target")
            path = self.root / name
        else:
            raise ValueError("Vision alias chain exceeds eight records")
        if not isinstance(record.get("main"), torch.Tensor):
            raise ValueError("Incomplete cached vision record")
        return record

    @staticmethod
    def _relative_error(actual, anchor):
        if actual["main"].shape != anchor["main"].shape:
            return float("inf")
        if len(actual["deepstack"]) != len(anchor["deepstack"]):
            return float("inf")
        pairs = [(actual["main"], anchor["main"])] + list(zip(actual["deepstack"], anchor["deepstack"]))
        if any(a.shape != b.shape for a, b in pairs):
            return float("inf")
        numerator = sum(float(torch.sum((a.float() - b.float()) ** 2)) for a, b in pairs)
        denominator = sum(float(torch.sum(a.float() ** 2)) for a, _ in pairs)
        return (numerator / max(denominator, 1e-12)) ** 0.5

    def _candidate_anchor(self, patch_key):
        signal = self.codec_signals.get(patch_key)
        if not isinstance(signal, dict):
            return None
        try:
            mv = float(signal["motion"])
            residual = float(signal["residual"])
            anchor_key = str(signal["anchor_patch_key"])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(mv) or not math.isfinite(residual):
            return None
        if mv > self.mv_threshold or residual > self.residual_threshold:
            return None
        return self._anchor_path(anchor_key)

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
            patch_key = self.patch_key(patch)
            entry = {"index": index, "start": start, "end": end,
                     "path": path, "patch_key": patch_key}
            if path.exists():
                try:
                    entry["record"] = self._load(path)
                    self.hits += 1
                except (OSError, RuntimeError, ValueError, KeyError):
                    missing.append(entry)
                    self.misses += 1
            else:
                missing.append(entry)
                self.misses += 1
            self._index_patch(patch_key, path)
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
                anchor_path = self._candidate_anchor(entry["patch_key"])
                if anchor_path is not None and anchor_path != entry["path"]:
                    anchor = self._load(anchor_path)
                    error = self._relative_error(record, anchor)
                    if error <= self.token_error_tolerance:
                        self._write(entry["path"], {"alias": anchor_path.name, "relative_error": error})
                        self.aliases += 1
                    else:
                        self._write(entry["path"], record)
                else:
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
        self._record_metrics()
        return combined_main, combined_levels


def install_into(vision_class, cache_dir: Path, namespace: str, metrics_path: Path = None,
                 codec_sidecar: Path = None, mv_threshold: float = 0.0,
                 residual_threshold: float = 0.0, token_error_tolerance: float = 0.0):
    """Wrap the LMDeploy vision class before the model is constructed."""
    if getattr(vision_class, "_ava_vit_cache_installed", False):
        raise RuntimeError("ViT output cache is already installed")
    store = VisionOutputStore(cache_dir, namespace, metrics_path=metrics_path,
                              codec_sidecar=codec_sidecar, mv_threshold=mv_threshold,
                              residual_threshold=residual_threshold,
                              token_error_tolerance=token_error_tolerance)
    original_forward = vision_class.forward

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb, pos_embeds):
        return store.forward(self, original_forward, hidden_states, cu_seqlens, rotary_pos_emb, pos_embeds)

    vision_class.forward = forward
    vision_class._ava_vit_cache_installed = True
    return store
