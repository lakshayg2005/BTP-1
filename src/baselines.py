"""Official released models, evaluated on our test sets as baselines (not part of our system).

- UnivFD (Ojha et al., CVPR 2023): the authors' released linear head `fc_weights.pth` on CLIP ViT-L/14,
  trained on ProGAN images (Wang et al. 2020 data). Same backbone as ours, so the comparison isolates
  the effect of our modern, degradation-augmented training data.
  Preprocessing follows their evaluation code: center crop 224 without resizing, CLIP normalisation,
  unnormalised `encode_image` features -> fc.
- Text: two publicly released fine-tuned detectors
  openai-community/roberta-base-openai-detector (OpenAI, GPT-2 output detector, 2019) and
  Hello-SimpleAI/chatgpt-detector-roberta (HC3 dataset, 2023).
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np
import torch
from PIL import Image

UNIVFD_URL = "https://github.com/WisconsinAIVision/UniversalFakeDetect/raw/main/pretrained_weights/fc_weights.pth"
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


class UnivFDBaseline:
    def __init__(self, encoder, weights_path: str = "models/baselines/univfd_fc_weights.pth"):
        if not os.path.exists(weights_path):
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            urllib.request.urlretrieve(UNIVFD_URL, weights_path)
        sd = torch.load(weights_path, map_location="cpu")
        sd = {k.split(".")[-1]: v for k, v in sd.items()}           # keys may be 'weight'/'bias' or 'fc.weight'
        self.w, self.b = sd["weight"].float().view(-1), sd["bias"].float().view(-1)
        self.encoder = encoder                                      # our shared ClipEncoder (same ViT-L/14)

    def _pixels(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB")
        if min(img.size) < 224:
            s = 224 / min(img.size)
            img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
        left, top = (img.width - 224) // 2, (img.height - 224) // 2
        x = torch.from_numpy(np.asarray(img.crop((left, top, left + 224, top + 224)), np.float32) / 255).permute(2, 0, 1)
        return ((x - CLIP_MEAN) / CLIP_STD)[None]

    @torch.no_grad()
    def prob_ai(self, img: Image.Image) -> float:
        px = self._pixels(img).to(self.encoder.device, self.encoder.dtype)
        feat = self.encoder.model(pixel_values=px).image_embeds.float()[0].cpu()
        return float(torch.sigmoid(feat @ self.w + self.b))


TEXT_BASELINES = {
    "roberta_openai_gpt2_detector": ("openai-community/roberta-base-openai-detector", ("fake", "label_0")),
    "roberta_chatgpt_detector_hc3": ("Hello-SimpleAI/chatgpt-detector-roberta", ("chatgpt", "label_1")),
}


class TextBaseline:
    def __init__(self, model_id: str, ai_labels: tuple[str, ...]):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model=model_id, top_k=None, truncation=True, max_length=512,
                             device=0 if torch.cuda.is_available() else -1)
        self.ai_labels = ai_labels

    def prob_ai(self, texts: list[str]) -> np.ndarray:
        out = []
        for scores in self.pipe(texts, batch_size=16):
            d = {s["label"].lower(): s["score"] for s in scores}
            out.append(next((d[l] for l in self.ai_labels if l in d), np.nan))
        return np.array(out, np.float32)
