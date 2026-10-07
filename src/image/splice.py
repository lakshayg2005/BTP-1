"""Synthetic "edited" images with exact ground-truth masks.

No small public benchmark with real/edited/AI labels fits our budget, so, like Face X-ray (CVPR 2020) and
Self-Blended Images (CVPR 2022), we create edits ourselves from real photos:
  - splice:    paste a region from another real photo
  - ai_patch:  paste a region from an AI-generated image (mimics generative inpainting / object insertion)
  - copy_move: duplicate a region inside the same photo
Edges are feathered so the boundary is not trivially visible, and a random daily-life degradation is applied
afterwards (WhatsApp / screenshot / JPEG), since that is how edited photos usually circulate.
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.image.augment import random_daily_life


def _random_mask(w: int, h: int) -> Image.Image:
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    rw, rh = int(w * random.uniform(0.15, 0.4)), int(h * random.uniform(0.15, 0.4))
    x, y = random.randint(0, w - rw), random.randint(0, h - rh)
    if random.random() < 0.5:
        d.ellipse([x, y, x + rw, y + rh], fill=255)
    else:
        pts = [(x + random.randint(0, rw), y + random.randint(0, rh)) for _ in range(random.randint(5, 9))]
        d.polygon(pts, fill=255)
        if np.asarray(m).mean() < 255 * 0.01:          # degenerate polygon -> fall back to a box
            d.rectangle([x, y, x + rw, y + rh], fill=255)
    return m


def make_edit(base: Image.Image, donor: Image.Image | None, kind: str, degrade: bool = True):
    """Returns (edited RGB image, binary mask as uint8 array HxW, kind)."""
    base = base.convert("RGB")
    w, h = base.size
    mask = _random_mask(w, h)
    if kind == "copy_move":
        dx, dy = random.randint(-w // 3, w // 3), random.randint(-h // 3, h // 3)
        src = base.transform(base.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))
    else:
        src = donor.convert("RGB").resize((w, h), Image.BICUBIC)
    soft = mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 4)))
    out = Image.composite(src, base, soft)
    if degrade:
        out = random_daily_life(out)
        mask = mask.resize(out.size, Image.NEAREST)
    return out, (np.asarray(mask) > 127).astype(np.uint8), kind


def random_edit(base: Image.Image, real_pool: list[str], ai_pool: list[str], degrade: bool = True):
    kind = random.choice(["splice", "ai_patch", "copy_move"])
    donor = None
    if kind == "splice":
        donor = Image.open(random.choice(real_pool))
    elif kind == "ai_patch":
        donor = Image.open(random.choice(ai_pool))
    return make_edit(base, donor, kind, degrade)
