"""Optional interception point for Qwen3-VL post-ViT feature reuse."""

from pathlib import Path


def install_into(vision_class, cache_dir: Path, namespace: str):
    """Wrap an LMDeploy vision class before the model is constructed."""
    original_forward = vision_class.forward

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb, pos_embeds):
        # PLACEHOLDER(F1): F0 verifies the server interception path only.
        return original_forward(self, hidden_states, cu_seqlens, rotary_pos_emb, pos_embeds)

    vision_class.forward = forward
    return original_forward
