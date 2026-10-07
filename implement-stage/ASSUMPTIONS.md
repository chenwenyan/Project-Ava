# Assumption Ledger — ViT Output Reuse
<!-- ASK mode: never -->

| ID | Under-determined by the request | Chosen | Class | Source |
|---|---|---|---|---|
| A-001 | Which visual state to persist | Cache Qwen3-VL's complete `visual.forward` tuple: final embeddings and every DeepStack output. | interface | default |
| A-002 | What proves two images may share a post-ViT state | Equality of the preprocessed patch tensor **and** positional inputs under one model namespace; codec zero flags alone are not sufficient. | semantic | default |
| A-003 | Cache fill timing | Write on first ViT access and reuse on later accesses across query processes; prior graph/CA accesses may prefill it. | interface | default |
| A-004 | Model version identity | Require an explicit cache namespace naming the exact checkpoint/preprocessor build; the launcher has a fixed default for the current Qwen3-VL model. | semantic | default |
| A-005 | What counts as a cross-query hit | The server's persistent feature-store hit for identical processed visual input; current GOP-level trace is insufficient to label exact-frame hits. | semantic | default |
| A-006 | Codec alias rule | A zero-motion/zero-residual declaration may nominate an anchor for comparison, but reuse occurs only if processed visual inputs match exactly. | semantic | default |
| A-007 | Engine compatibility | Target installed qwen35 LMDeploy PyTorch Qwen3-VL implementation; unsupported versions fail closed. | interface | default |

## Notes

A-002/A-006: H.264 prediction is relative to reference pictures, B frames can have two references, and in-loop filtering or preprocessing can change pixels even with zero motion/residual. Qwen3-VL's visual attention and DeepStack also make local block equality insufficient for final token equality.
