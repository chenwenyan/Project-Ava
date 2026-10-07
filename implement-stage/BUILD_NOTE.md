# Build Note — ViT Output Reuse

| Rung | Feature | Acceptance check | Tier | Status |
|---|---|---|---|---|
| F0 | Minimal server launcher and testable interception spine | `/data/wenyan/conda/envs/qwen35/bin/python -m scripts.test_vit_token_cache --spine` | MUST | ✅ |
| F1 | Persist/reload all post-ViT outputs per visual segment | `/data/wenyan/conda/envs/qwen35/bin/python -m scripts.test_vit_token_cache --cache` | MUST | ✅ |
| F2 | Configurable codec thresholds and guarded keyframe-token alias | `/data/wenyan/conda/envs/qwen35/bin/python -m scripts.test_vit_token_cache --codec` | MUST | ✅ |
| F3 | Qwen3-VL server integration and warm-request bypass | `bash scripts/smoke_vit_token_cache.sh` | MUST | ✅ |

## Run record

- F0: `python -m scripts.test_vit_token_cache --spine` exit 0; wrote `implement-stage/f0_smoke.json`. The hook currently calls the original ViT and is labelled PLACEHOLDER(F1).
- F1: `python -m scripts.test_vit_token_cache --cache` exit 0; a two-segment run wrote complete output tuples, repeated input skipped the ViT, one changed segment recomputed, and a fresh store read the persisted record. F0 rechecked, exit 0. PLACEHOLDER(F1) retired.
- F2: synthetic codec sidecar with motion/residual thresholds passed; accepted zero-error alias and both threshold and token-error rejection verified. Earlier checks passed. Real H.264 transform residual scores require external decoder instrumentation; the available PyAV side data exposes motion vectors only.
- F3: `bash scripts/smoke_vit_token_cache.sh` exit 0 under CPJ job `job-muxjbwfu-96a5a8ab`. First request: one miss and one ViT call; repeated request: one cache hit and no additional ViT call. Both answers matched; see `results/vit_token_cache_smoke.json`. LMDeploy's API server spawned a separate engine process, so the hook is installed there through a startup bootstrap and bound to the visual class used by the live model. Earlier focused checks passed after the integration fix.
- Assumption sweep: `implement-stage/SILENT_ASSUMPTION_SWEEP_ROUND1.json` records the first review verbatim. Its four semantic gaps, two interface gaps, and one stale row were addressed in the ledger or implementation. `implement-stage/SILENT_ASSUMPTION_SWEEP.json` records the second review verbatim: zero remaining findings. `review_independence: same-family`; `acceptance_status: provisional`. The corrected approximate-alias first-access behavior and default-directory bootstrap passed focused checks; the complete GPU smoke result above predates these narrow fixes.

## Deferred

Approximate reuse for genuinely changed regions requires a separate distortion/QA experiment and is excluded from the exact-reuse implementation.

## Blockers
