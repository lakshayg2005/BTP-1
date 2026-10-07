"""AI-generated (spoofed / cloned) speech detector: frozen self-supervised speech encoder + linear head.

Self-supervised speech features (wav2vec2 / XLS-R / WavLM) with a light back-end are among the most
robust audio deepfake detectors on in-the-wild data (Tak et al., Odyssey 2022; Muller et al., Interspeech 2022).
We mean-pool a middle-to-late hidden layer, which carries more artifact information than the last layer.
Sliding-window scoring gives time localization ("which seconds sound synthetic").
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModel

from src.common.device import get_device

DEFAULT_AUDIO_ENCODER = os.environ.get("BTP_AUDIO_ENCODER", "facebook/wav2vec2-xls-r-300m")
SR = 16000


def load_audio(path: str, max_seconds: float | None = None) -> np.ndarray:
    import torchaudio
    wav, sr = torchaudio.load(path)
    wav = wav.mean(0)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if max_seconds:
        wav = wav[: int(max_seconds * SR)]
    return wav.numpy().astype(np.float32)


class SpeechEncoder:
    def __init__(self, name: str = DEFAULT_AUDIO_ENCODER, layer: int = 12, device: str = "auto"):
        self.device = get_device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.fe = AutoFeatureExtractor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, torch_dtype=self.dtype).to(self.device).eval()
        self.layer = min(layer, self.model.config.num_hidden_layers)

    @torch.no_grad()
    def embed(self, wavs: list[np.ndarray]) -> np.ndarray:
        inp = self.fe(wavs, sampling_rate=SR, return_tensors="pt", padding=True)
        x = inp["input_values"].to(self.device, self.dtype)
        hs = self.model(x, output_hidden_states=True).hidden_states[self.layer].float()
        return F.normalize(hs.mean(1), dim=-1).cpu().numpy()


class AIAudioDetector:
    def __init__(self, head_path: str, encoder: SpeechEncoder | None = None):
        bundle = joblib.load(head_path)
        self.head = bundle["head"]
        self.encoder = encoder or SpeechEncoder(bundle.get("encoder", DEFAULT_AUDIO_ENCODER), bundle.get("layer", 12))

    def prob_ai(self, wav: np.ndarray) -> float:
        return float(self.head.predict_proba(self.encoder.embed([wav[: 30 * SR]]))[0, 1])

    def segment_scores(self, wav: np.ndarray, win: float = 2.0, hop: float = 1.0) -> list[tuple[float, float, float]]:
        """(start_s, end_s, p_ai) for sliding windows; used to show which parts sound synthetic."""
        w, h = int(win * SR), int(hop * SR)
        if len(wav) <= w:
            return [(0.0, len(wav) / SR, self.prob_ai(wav))]
        starts = list(range(0, len(wav) - w + 1, h))
        out = []
        for i in range(0, len(starts), 16):
            chunk = [wav[s: s + w] for s in starts[i: i + 16]]
            p = self.head.predict_proba(self.encoder.embed(chunk))[:, 1]
            out += [(s / SR, (s + w) / SR, float(q)) for s, q in zip(starts[i: i + 16], p)]
        return out


def bandwidth_hz(wav: np.ndarray, sr: int = SR) -> float:
    """Highest frequency holding meaningful energy. Many TTS vocoders and phone calls are band-limited,
    so this is reported as context, never as proof."""
    spec = np.abs(np.fft.rfft(wav * np.hanning(len(wav)))) ** 2
    freqs = np.fft.rfftfreq(len(wav), 1 / sr)
    cum = np.cumsum(spec) / (spec.sum() + 1e-12)
    return float(freqs[np.searchsorted(cum, 0.995)])
