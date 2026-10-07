"""AI-generated text detector: zero-shot likelihood statistics from two small LLMs + a tiny calibrated head.

Features (all computed per text and per sentence):
- Binoculars score (Hans et al., ICML 2024): log-perplexity under a "performer" model divided by the
  cross-perplexity between "observer" and "performer". Low = machine-like. Original uses Falcon-7B pair;
  we use a small same-tokenizer pair (Qwen2.5-0.5B base / instruct; 1.5B is a drop-in upgrade if download
  time allows) to fit 8 GB and our slow network.
- Fast-DetectGPT analytic criterion (Bao et al., ICLR 2024): how much more likely the observed tokens are
  than tokens sampled from the model's own distribution, normalised. High = machine-like.
- Mean token log-prob and mean entropy (GLTR-style, Gehrmann et al., ACL 2019).
Per-token surprisal is kept for highlighting which words look "too predictable".
"""
from __future__ import annotations

import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.device import get_device

_SIZE = os.environ.get("BTP_TEXT_LM_SIZE", "0.5B")   # set to 1.5B on Kaggle / faster networks
OBSERVER = os.environ.get("BTP_TEXT_OBSERVER", f"Qwen/Qwen2.5-{_SIZE}")
PERFORMER = os.environ.get("BTP_TEXT_PERFORMER", f"Qwen/Qwen2.5-{_SIZE}-Instruct")
FEATURE_NAMES = ["binoculars", "fast_detectgpt", "mean_logprob", "mean_entropy"]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip().split()) >= 4]


class LikelihoodScorer:
    def __init__(self, observer: str = OBSERVER, performer: str = PERFORMER, device: str = "auto", max_tokens: int = 512):
        self.device = get_device(device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(observer)
        self.obs = AutoModelForCausalLM.from_pretrained(observer, torch_dtype=dtype).to(self.device).eval()
        self.perf = AutoModelForCausalLM.from_pretrained(performer, torch_dtype=dtype).to(self.device).eval()
        self.max_tokens = max_tokens

    @torch.no_grad()
    def score(self, text: str) -> dict:
        ids = self.tok(text, return_tensors="pt", truncation=True, max_length=self.max_tokens).input_ids.to(self.device)
        if ids.shape[1] < 8:
            return {"features": None, "tokens": []}
        lo = self.obs(ids).logits[0, :-1].float()
        lp = self.perf(ids).logits[0, :-1].float()
        tgt = ids[0, 1:]

        logp_p = F.log_softmax(lp, -1)
        tok_ll = logp_p.gather(-1, tgt[:, None])[:, 0]                       # log p_perf(x_t)
        ppl = -tok_ll.mean()
        x_ppl = -(F.softmax(lo, -1) * logp_p).sum(-1).mean()                 # cross-perplexity observer->performer
        binoculars = (ppl / x_ppl).item()

        probs = logp_p.exp()
        mu = (probs * logp_p).sum(-1)
        var = (probs * logp_p ** 2).sum(-1) - mu ** 2
        fast = ((tok_ll.sum() - mu.sum()) / var.sum().sqrt()).item()

        entropy = -(probs * logp_p).sum(-1)
        tokens = [(self.tok.decode([t]), float(-l), float(e)) for t, l, e in zip(tgt.tolist(), tok_ll.tolist(), entropy.tolist())]
        return {"features": [binoculars, fast, tok_ll.mean().item(), entropy.mean().item()], "tokens": tokens}
