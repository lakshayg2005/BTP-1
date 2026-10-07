"""AI-generated image detector: frozen CLIP ViT-L/14 features + a small linear head.

Based on UnivFD (Ojha et al., CVPR 2023): features of a large frozen vision-language encoder generalize to
unseen generators far better than a CNN trained from scratch. Only the linear head is trained, so it is
cheap and fits our 8 GB GPU / CPU budget.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from src.common.device import get_device

DEFAULT_ENCODER = os.environ.get("BTP_IMAGE_ENCODER", "openai/clip-vit-large-patch14")


class ClipEncoder:
    def __init__(self, name: str = DEFAULT_ENCODER, device: str = "auto"):
        self.device = get_device(device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.processor = CLIPImageProcessor.from_pretrained(name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(name, torch_dtype=dtype).to(self.device).eval()
        self.dtype = dtype

    def pixels(self, images: list[Image.Image]) -> torch.Tensor:
        px = self.processor(images=[im.convert("RGB") for im in images], return_tensors="pt")["pixel_values"]
        return px.to(self.device, self.dtype)

    def embed_pixels(self, px: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model(pixel_values=px).image_embeds.float(), dim=-1)

    @torch.no_grad()
    def embed(self, images: list[Image.Image]) -> np.ndarray:
        return self.embed_pixels(self.pixels(images)).cpu().numpy()


class AIImageDetector:
    """Wraps encoder + trained head (sklearn LogisticRegression saved with joblib)."""

    def __init__(self, head_path: str, encoder: ClipEncoder | None = None):
        bundle = joblib.load(head_path)
        self.head = bundle["head"]
        self.encoder = encoder or ClipEncoder(bundle.get("encoder", DEFAULT_ENCODER))

    def prob_ai(self, image: Image.Image) -> float:
        feat = self.encoder.embed([image])
        return float(self.head.predict_proba(feat)[0, 1])

    def saliency(self, image: Image.Image) -> np.ndarray:
        """Input x gradient of the AI logit, averaged over channels, normalized to 0..1 (224x224)."""
        w = torch.tensor(self.head.coef_[0], dtype=torch.float32, device=self.encoder.device)
        b = float(self.head.intercept_[0])
        px = self.encoder.pixels([image]).float().requires_grad_(True)
        model = self.encoder.model.float()
        logit = F.normalize(model(pixel_values=px).image_embeds, dim=-1)[0] @ w + b
        logit.backward()
        if self.encoder.dtype == torch.float16:
            model.half()
        sal = (px.grad * px).abs().sum(1)[0].detach().cpu().numpy()
        sal = np.clip(sal / (np.percentile(sal, 99) + 1e-8), 0, 1)
        return sal
