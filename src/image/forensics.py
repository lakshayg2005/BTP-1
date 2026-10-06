"""Classical, training-free evidence for edited images and provenance.

- Error Level Analysis (ELA): regions pasted/edited after the last JPEG save re-compress differently.
- Noise-residual inconsistency: spliced regions often carry a different sensor-noise level.
- Metadata: camera EXIF, editing software tags, generator tags (e.g. Stable Diffusion "parameters"),
  and presence of a C2PA content-credentials manifest (presence only; we do not verify signatures).

These are weak alone; a learned tamper localizer (TruFor-style) is added in Phase 2.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import ExifTags, Image, ImageFilter

GENERATOR_HINTS = ("stable diffusion", "midjourney", "dall-e", "dall·e", "firefly", "comfyui", "automatic1111",
                   "novelai", "flux", "imagen", "gpt-image", "openai", "gemini", "ideogram", "leonardo")
EDITOR_HINTS = ("photoshop", "lightroom", "gimp", "snapseed", "picsart", "canva", "facetune", "affinity",
                "pixlr", "remini", "meitu", "capcut")


def ela_map(img: Image.Image, quality: int = 90) -> np.ndarray:
    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    diff = np.abs(np.asarray(rgb, np.float32) - np.asarray(Image.open(buf).convert("RGB"), np.float32)).mean(-1)
    return diff / (diff.max() + 1e-8)


def noise_map(img: Image.Image, block: int = 32) -> np.ndarray:
    g = img.convert("L")
    resid = np.asarray(g, np.float32) - np.asarray(g.filter(ImageFilter.MedianFilter(3)), np.float32)
    h, w = resid.shape[0] // block, resid.shape[1] // block
    if h == 0 or w == 0:
        return np.zeros((1, 1), np.float32)
    blocks = resid[: h * block, : w * block].reshape(h, block, w, block).std(axis=(1, 3))
    return blocks


def inconsistency_score(m: np.ndarray) -> float:
    """How much a few regions stand out from the rest (robust z-score of the top 2% blocks), squashed to 0..1."""
    v = m.ravel()
    if v.size < 16:
        return 0.0
    med, mad = np.median(v), np.median(np.abs(v - np.median(v))) + 1e-6
    z = (np.percentile(v, 98) - med) / (1.4826 * mad)
    return float(1 / (1 + np.exp(-(z - 6) / 1.5)))


def read_metadata(path: str) -> dict:
    out = {"camera": None, "software": None, "generator_tag": None, "editor_tag": None, "c2pa_manifest": False}
    with open(path, "rb") as f:
        raw = f.read()
    out["c2pa_manifest"] = b"c2pa" in raw
    img = Image.open(path)
    texts = []
    exif = img.getexif()
    if exif:
        tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        if tags.get("Make") or tags.get("Model"):
            out["camera"] = f"{tags.get('Make', '')} {tags.get('Model', '')}".strip()
        if tags.get("Software"):
            out["software"] = str(tags["Software"])
            texts.append(out["software"])
    texts += [f"{k}: {v}" for k, v in (img.info or {}).items() if isinstance(v, str)]
    blob = " ".join(texts).lower()
    out["generator_tag"] = next((h for h in GENERATOR_HINTS if h in blob), None)
    out["editor_tag"] = next((h for h in EDITOR_HINTS if h in blob), None)
    return out
