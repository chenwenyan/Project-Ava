"""Persistent post-ViT output reuse for LMDeploy Qwen3-VL's PyTorch engine.

The key is the exact *processed* input to one independent vision-attention
segment. This deliberately does not infer token equality from codec motion flags.
"""

import hashlib
import json
import math
import os
import tempfile
from collections import OrderedDict
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
        payload.update(self._extra_metrics())
        with tempfile.NamedTemporaryFile(mode="w", dir=self.metrics_path.parent,
                                         suffix=".tmp", delete=False) as tmp:
            json.dump(payload, tmp)
            temp_path = Path(tmp.name)
        os.replace(temp_path, self.metrics_path)

    def _extra_metrics(self):
        return {}

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

    def _has(self, path):
        return path.exists()

    def _load(self, path, device=None, count_hit=True):
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

    def _store_computed(self, path, device_record):
        record = {"main": device_record["main"].cpu(),
                  "deepstack": [level.cpu() for level in device_record["deepstack"]]}
        self._write(path, record)

    def _write_alias(self, path, anchor_path, error):
        self._write(path, {"alias": anchor_path.name, "relative_error": error})

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
            if self._has(path):
                try:
                    entry["record"] = self._load(path, device=hidden_states.device)
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
                device_record = {
                    "main": main_parts[j].detach(),
                    "deepstack": [level[j].detach() for level in level_parts],
                }
                anchor_path = self._candidate_anchor(entry["patch_key"])
                if anchor_path is not None and anchor_path != entry["path"]:
                    anchor = self._load(anchor_path, device=hidden_states.device, count_hit=False)
                    error = self._relative_error(device_record, anchor)
                    if error <= self.token_error_tolerance:
                        self._write_alias(entry["path"], anchor_path, error)
                        self.aliases += 1
                    else:
                        self._store_computed(entry["path"], device_record)
                else:
                    self._store_computed(entry["path"], device_record)
                entry["record"] = device_record

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


class TieredVisionOutputStore(VisionOutputStore):
    """Keep hot records on GPU, colder records in bounded RAM, then disk."""

    @staticmethod
    def _mem_available_bytes():
        with open("/proc/meminfo") as info:
            for line in info:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        raise RuntimeError("MemAvailable is unavailable")

    def __init__(self, *args, gpu_budget_bytes: int,
                 host_budget_bytes: int = None, host_reserve_bytes: int = 32 * 2**30,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if gpu_budget_bytes <= 0:
            raise ValueError("GPU cache budget must be positive")
        self.gpu_budget_bytes = int(gpu_budget_bytes)
        self.host_reserve_bytes = int(host_reserve_bytes)
        available = self._mem_available_bytes()
        self.host_budget_bytes = (min(24 * 2**30,
                                      max(0, (available - self.host_reserve_bytes) // 2))
                                  if host_budget_bytes is None else int(host_budget_bytes))
        if self.host_budget_bytes < 0 or self.host_reserve_bytes < 0:
            raise ValueError("Host cache budget and reserve must be nonnegative")
        self.host_records = OrderedDict()
        self.alias_records = {}
        self.patch_indices = {}
        self.gpu_records = OrderedDict()
        self.host_bytes = 0
        self.gpu_bytes = 0
        self.peak_gpu_bytes = 0
        self.gpu_hits = 0
        self.host_hits = 0
        self.gpu_evictions = 0
        self.disk_hits = 0
        self.disk_writes = 0
        self.disk_bytes = 0

    @staticmethod
    def _record_bytes(record):
        return sum(tensor.numel() * tensor.element_size()
                   for tensor in [record["main"], *record["deepstack"]])

    def _extra_metrics(self):
        return {"storage": "tiered", "gpu_budget_bytes": self.gpu_budget_bytes,
                "gpu_bytes": self.gpu_bytes, "peak_gpu_bytes": self.peak_gpu_bytes,
                "host_budget_bytes": self.host_budget_bytes,
                "host_reserve_bytes": self.host_reserve_bytes,
                "host_bytes": self.host_bytes, "gpu_hits": self.gpu_hits,
                "host_hits": self.host_hits, "gpu_evictions": self.gpu_evictions,
                "disk_hits": self.disk_hits, "disk_writes": self.disk_writes,
                "disk_bytes": self.disk_bytes}

    def _has(self, path):
        name = path.name
        return (name in self.gpu_records or name in self.host_records
                or name in self.alias_records or path.exists())

    def _index_patch(self, patch_key, path):
        self.patch_indices[patch_key] = path.name

    def _anchor_path(self, patch_key):
        name = self.patch_indices.get(patch_key)
        return self.root / name if name and self._has(self.root / name) else None

    def _store_computed(self, path, device_record):
        size = self._record_bytes(device_record)
        if size > self.gpu_budget_bytes:
            self._admit_host(path.name, device_record, size)
        else:
            self._remember_gpu(path, device_record, clone=True)

    def _write_alias(self, path, anchor_path, error):
        self.alias_records[path.name] = {"alias": anchor_path.name,
                                         "relative_error": error}

    def _load(self, path, device=None, count_hit=True):
        name = path.name
        for _ in range(8):
            alias = self.alias_records.get(name)
            if alias is None:
                break
            name = alias["alias"]
        else:
            raise ValueError("Vision alias chain exceeds eight records")
        if device is None:
            if name in self.host_records:
                return self.host_records[name]
            if name in self.gpu_records:
                gpu_record = self.gpu_records[name][0]
                return {"main": gpu_record["main"].cpu(),
                        "deepstack": [level.cpu() for level in gpu_record["deepstack"]]}
            return super()._load(self.root / name)
        if name in self.gpu_records:
            self.gpu_records.move_to_end(name)
            if count_hit:
                self.gpu_hits += 1
            return self.gpu_records[name][0]
        if name in self.host_records:
            record = self.host_records.pop(name)
            self.host_bytes -= self._record_bytes(record)
            if count_hit:
                self.host_hits += 1
        else:
            record = super()._load(self.root / name)
            if count_hit:
                self.disk_hits += 1
        transferred = {"main": record["main"].to(device),
                       "deepstack": [level.to(device) for level in record["deepstack"]]}
        size = self._record_bytes(record)
        if size <= self.gpu_budget_bytes:
            self._remember_gpu(self.root / name, transferred, clone=False)
        elif name not in self.host_records and not (self.root / name).exists():
            self._admit_host(name, transferred, size)
        return transferred

    def _spill(self, name, record):
        path = self.root / name
        if path.exists():
            return
        cpu_record = {"main": record["main"].cpu(),
                      "deepstack": [level.cpu() for level in record["deepstack"]]}
        super()._write(path, cpu_record)
        self.disk_writes += 1
        self.disk_bytes += path.stat().st_size

    def _admit_host(self, name, record, size):
        while self.host_records and (self.host_bytes + size > self.host_budget_bytes
                                     or self._mem_available_bytes() < self.host_reserve_bytes):
            old_name, old_record = self.host_records.popitem(last=False)
            self.host_bytes -= self._record_bytes(old_record)
            self._spill(old_name, old_record)
        if (size > self.host_budget_bytes
                or self._mem_available_bytes() < self.host_reserve_bytes):
            self._spill(name, record)
            return
        self.host_records[name] = {"main": record["main"].cpu(),
                                   "deepstack": [level.cpu() for level in record["deepstack"]]}
        self.host_bytes += size

    def _remember_gpu(self, path, record, clone=True):
        name = path.name
        size = self._record_bytes(record)
        if size > self.gpu_budget_bytes:
            return
        if name in self.gpu_records:
            self.gpu_records.move_to_end(name)
            return
        while self.gpu_bytes + size > self.gpu_budget_bytes:
            evicted_name, (evicted, removed_size) = self.gpu_records.popitem(last=False)
            self._admit_host(evicted_name, evicted, removed_size)
            self.gpu_bytes -= removed_size
            self.gpu_evictions += 1
        if clone:
            record = {"main": record["main"].clone(),
                      "deepstack": [level.clone() for level in record["deepstack"]]}
        self.gpu_records[name] = (record, size)
        self.gpu_bytes += size
        self.peak_gpu_bytes = max(self.peak_gpu_bytes, self.gpu_bytes)


def install_into(vision_class, cache_dir: Path, namespace: str, metrics_path: Path = None,
                 codec_sidecar: Path = None, mv_threshold: float = 0.0,
                 residual_threshold: float = 0.0, token_error_tolerance: float = 0.0,
                 storage_mode: str = "disk", gpu_budget_bytes: int = 0,
                 host_budget_bytes: int = None, host_reserve_bytes: int = 32 * 2**30):
    """Wrap the LMDeploy vision class before the model is constructed."""
    if getattr(vision_class, "_ava_vit_cache_installed", False):
        raise RuntimeError("ViT output cache is already installed")
    if storage_mode not in {"disk", "tiered"}:
        raise ValueError(f"Unsupported vision-cache storage mode: {storage_mode}")
    store_class = TieredVisionOutputStore if storage_mode == "tiered" else VisionOutputStore
    options = ({"gpu_budget_bytes": gpu_budget_bytes,
                "host_budget_bytes": host_budget_bytes,
                "host_reserve_bytes": host_reserve_bytes}
               if storage_mode == "tiered" else {})
    store = store_class(cache_dir, namespace, metrics_path=metrics_path,
                        codec_sidecar=codec_sidecar, mv_threshold=mv_threshold,
                        residual_threshold=residual_threshold,
                        token_error_tolerance=token_error_tolerance, **options)
    original_forward = vision_class.forward

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb, pos_embeds):
        return store.forward(self, original_forward, hidden_states, cu_seqlens, rotary_pos_emb, pos_embeds)

    vision_class.forward = forward
    vision_class._ava_vit_cache_installed = True
    return store
