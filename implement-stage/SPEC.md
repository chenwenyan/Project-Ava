# ViT Output Reuse for Qwen3-VL CA

Base commit: `42dadf5258a0f62b0607327a734f40a2f36b0c65`.

Target: persist the complete post-ViT output consumed by LMDeploy Qwen3-VL (main image embeddings plus DeepStack outputs), and bypass ViT on later requests for the same preprocessed visual input. Use current AVA CA image requests and the installed Qwen3-VL PyTorch engine.

Inputs: current JPEG image requests, Qwen3-VL visual model, optional codec side information. Outputs: persistent per-visual-segment tensor records under a configured cache directory and server counters for hits, misses and ViT calls. The ordinary CA answer format stays unchanged.

Success command: a local cache test must show identical reconstructed tensors and zero additional ViT calls on a repeat; then a Qwen3-VL server smoke run must show a warm-request hit and unchanged answer path.

Scope: exact repeated-input reuse first. Low codec motion/residual scores nominate a keyframe anchor, never prove equivalent post-ViT tensors. A first-pass complete-output comparison is mandatory before aliasing; an optional nonzero token-error tolerance permits measured approximate aliases. New CA sampling and a real codec-residual extractor are excluded.
