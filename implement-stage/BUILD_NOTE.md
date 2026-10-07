# Build Note — ViT Output Reuse

| Rung | Feature | Acceptance check | Tier | Status |
|---|---|---|---|---|
| F0 | Minimal server launcher and testable interception spine | `/data/wenyan/conda/envs/qwen35/bin/python -m scripts.test_vit_token_cache --spine` | MUST | ✅ |
| F1 | Persist/reload all post-ViT outputs per visual segment | `/data/wenyan/conda/envs/qwen35/bin/python -m scripts.test_vit_token_cache --cache` | MUST | ✅ |
| F2 | Qwen3-VL server integration and warm-request bypass | `/data/wenyan/conda/envs/qwen35/bin/python -m scripts.test_vit_token_cache --integration` | MUST | ⬜ |
| F3 | Codec zero-signal alias audit with decoded-input equality | `/data/wenyan/conda/envs/qwen35/bin/python -m scripts.test_vit_token_cache --codec` | SHOULD | ⬜ |

## Run record

- F0: `python -m scripts.test_vit_token_cache --spine` exit 0; wrote `implement-stage/f0_smoke.json`. The hook currently calls the original ViT and is labelled PLACEHOLDER(F1).
- F1: `python -m scripts.test_vit_token_cache --cache` exit 0; a two-segment run wrote complete output tuples, repeated input skipped the ViT, one changed segment recomputed, and a fresh store read the persisted record. F0 rechecked, exit 0. PLACEHOLDER(F1) retired.

## Deferred

Approximate reuse for genuinely changed regions requires a separate distortion/QA experiment and is excluded from the exact-reuse implementation.

## Blockers
