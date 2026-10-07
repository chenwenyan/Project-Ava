# Assumption Ledger — ViT Output Reuse
<!-- ASK mode: never -->

| ID | Under-determined by the request | Chosen | Class | Source |
|---|---|---|---|---|
| A-001 | Which visual state to persist | Cache Qwen3-VL's complete `visual.forward` tuple: final embeddings and every DeepStack output. | interface | default |
| A-002 | What proves two images may share a post-ViT state | Equality of the preprocessed patch tensor **and** positional inputs under one model namespace; codec zero flags alone are not sufficient. | semantic | default |
| A-003 | Cache fill timing | Write on first ViT access and reuse on later accesses across query processes; prior graph/CA accesses may prefill it. | interface | default |
| A-004 | Model version identity | Use a cache namespace naming the checkpoint/preprocessor build; the launcher defaults to the current Qwen3-VL model and permits an explicit override. | semantic | default |
| A-005 | What counts as a cross-query hit | The server's persistent feature-store hit for identical processed visual input; current GOP-level trace is insufficient to label exact-frame hits. | semantic | default |
| A-006 | Codec alias rule | Low motion/residual scores nominate an anchor; the first access still runs ViT and compares the complete output tuple to the anchor. A default zero token-error tolerance accepts only equal outputs. | semantic | default |
| A-007 | Engine compatibility | Target installed qwen35 LMDeploy PyTorch Qwen3-VL implementation; runtime structural mismatches raise errors. There is no general version compatibility gate. | interface | default |
| A-008 | How codec signals reach a JPEG-based ViT server | A JSON sidecar maps hashes of preprocessed patch inputs to an anchor patch hash and numeric motion/residual scores. Missing sidecar values disable codec aliasing. | interface | default |
| A-009 | What the two adjustable thresholds mean | Both `motion <= mv_threshold` and `residual <= residual_threshold` must hold before considering anchor reuse. Values and aggregation are declared by the sidecar producer. | semantic | default |
| A-010 | How approximate anchor reuse is controlled | A first-pass ViT comparison must also meet a configurable relative post-ViT token-error bound; default zero permits only exact equality. Subsequent queries reuse the accepted anchor. | semantic | default |
| A-011 | Token-error aggregation | Compare the whole main-plus-DeepStack tuple using relative L2 error, with computed output energy in the denominator and a `1e-12` floor. | semantic | sweep |
| A-012 | Motion and residual threshold defaults | Both thresholds default to zero; a sidecar with positive scores needs explicit threshold settings. | semantic | sweep |
| A-013 | When anchor aliasing is considered | Only on a candidate cache miss when its anchor is already stored. An exact record written before its anchor is available is not revisited for aliasing. | semantic | sweep |
| A-014 | Validating request output | The first access returns its freshly computed post-ViT tuple. If accepted, only subsequent accesses read the stored anchor alias. | semantic | sweep |
| A-015 | Spawned engine activation | The launcher places a startup bootstrap on spawned engines' Python path. Cache installation does not require an explicit cache-directory environment variable; absent one it uses `AVA_cache/vit_tokens`. | interface | sweep |
| A-016 | Counter units and lifetime | Hits/misses count visual segments; `vit_calls` counts original batched ViT invocations. Counters reset per server engine instance, while cache records persist. | interface | sweep |
| A-017 | Meaning of the 20 GB GPU allocation | A hard 20 GiB budget for cached post-ViT tensor payloads, excluding model weights, temporary activations, allocator metadata and unrelated KV cache. The GPU tier uses LRU eviction. | semantic | user/default |
| A-018 | Meaning of the remaining host tier | Each complete post-ViT record has one hot resident tier: GPU up to 20 GiB, then host RAM under a pressure-aware cap. Host overflow spills to disk; disk copies may remain after promotion. | semantic | user/default |
| A-019 | Latency comparison for the tiered implementation | Repeat VideoMME-Long 601–620's 60 CA queries with the existing `kf3sig_a07_t07` graphs and SA outputs. Run no-reuse and tiered arms sequentially on the same GPU to avoid simultaneous host-memory pressure. Speedup is median no-reuse CA seconds divided by median tiered CA seconds. | semantic | default |
| A-020 | Host-memory capacity policy | At server startup cap cached host tensor payload at the smaller of 24 GiB and half of `MemAvailable` above a 32 GiB reserve; during execution spill host LRU entries whenever the cap would be exceeded or current `MemAvailable` falls below the reserve. | semantic | default |

## Notes

A-002/A-006: H.264 prediction is relative to reference pictures, B frames can have two references, and in-loop filtering or preprocessing can change pixels even with zero motion/residual. Qwen3-VL's visual attention and DeepStack also make local block equality insufficient for final token equality.

A-008/A-009: The available PyAV decoder exports motion vectors but not H.264 transform residual coefficients. The implementation accepts measured residual scores from a sidecar and refuses to infer zero from a missing value. A decoded-image difference would be a proxy, not a codec residual, and is not silently substituted.
