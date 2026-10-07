"""Learned edit localizer: a linear probe on CLIP ViT-L/14 patch tokens predicts, for each 14x14 patch,
whether it belongs to an edited region. Gives a 16x16 tamper map and an image-level edit score.

Same frozen-encoder idea as the AI detector (UnivFD), applied per patch; trained on synthetic edits
(src/image/splice.py). Cheap: one forward pass shared with the AI detector.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def grid_info(encoder) -> tuple[int, int, int]:
    """(input size, patch size, grid) from the encoder config; 224 / 14 / 16 for CLIP ViT-L/14."""
    cfg = encoder.model.config
    return cfg.image_size, cfg.patch_size, cfg.image_size // cfg.patch_size


def patch_tokens(encoder, images) -> torch.Tensor:
    """(B, grid*grid, hidden) patch tokens from the shared ClipEncoder (CLS token dropped)."""
    with torch.no_grad():
        out = encoder.model.vision_model(pixel_values=encoder.pixels(images))
    return out.last_hidden_state[:, 1:].float()


def mask_to_patch_labels(mask: np.ndarray, size: int = 224, patch: int = 14, frac: float = 0.3) -> np.ndarray:
    """Downsample an HxW 0/1 mask to the patch grid (label 1 if >= frac of the patch is edited).
    Uses the same geometry as the CLIP processor: short side resized to `size`, then center crop."""
    h, w = mask.shape
    s = size / min(h, w)
    nh, nw = max(size, round(h * s)), max(size, round(w * s))
    m = np.asarray(Image.fromarray(mask * 255).resize((nw, nh), Image.BILINEAR), np.float32) / 255
    top, left = (nh - size) // 2, (nw - size) // 2
    m = m[top: top + size, left: left + size]
    g = size // patch
    return (m[: g * patch, : g * patch].reshape(g, patch, g, patch).mean((1, 3)) >= frac).astype(np.float32).ravel()


class TamperHead(torch.nn.Module):
    def __init__(self, dim: int = 1024):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fc = torch.nn.Linear(dim, 1)

    def forward(self, tokens):            # (B, P, D) -> (B, P) logits
        return self.fc(self.norm(tokens)).squeeze(-1)


def top_k_score(p: np.ndarray, k: int = 8) -> float:
    return float(np.sort(p)[-k:].mean())


class TamperLocalizer:
    def __init__(self, path: str, encoder):
        self.encoder = encoder
        self.grid = grid_info(encoder)[2]
        self.head = TamperHead(encoder.model.config.hidden_size).to(encoder.device)
        self.head.load_state_dict(torch.load(path, map_location=encoder.device))
        self.head.eval()

    @torch.no_grad()
    def predict(self, image) -> tuple[np.ndarray, float]:
        """Returns (grid x grid probability map, image-level edit score = mean of the top 8 patches)."""
        p = torch.sigmoid(self.head(patch_tokens(self.encoder, [image])))[0].cpu().numpy()
        return p.reshape(self.grid, self.grid), top_k_score(p)
