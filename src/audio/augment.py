"""Daily-life audio degradations: WhatsApp voice notes (Opus), phone calls (8 kHz band), noise, MP3."""
import random

import numpy as np
import torch
import torchaudio.functional as AF

SR = 16000


def phone_band(w: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(w)
    t = AF.resample(AF.resample(t, SR, 8000), 8000, SR)
    return AF.highpass_biquad(t, SR, 300).numpy()


def add_noise(w: np.ndarray, snr_db: float) -> np.ndarray:
    p = np.mean(w ** 2) + 1e-9
    return (w + np.random.randn(len(w)).astype(np.float32) * np.sqrt(p / 10 ** (snr_db / 10))).astype(np.float32)


def codec(w: np.ndarray, fmt: str) -> np.ndarray:
    try:
        t = torch.from_numpy(w)[None]
        if fmt == "mp3":
            out = AF.apply_codec(t, SR, "mp3", compression=random.choice([-4.5, -6.0, -9.0]))
        else:
            out = AF.apply_codec(t, SR, "ogg", encoder="opus" if fmt == "opus" else None)
        return out[0].numpy()[: len(w)].astype(np.float32)
    except Exception:  # codec support depends on the ffmpeg build; fall back to the clean signal
        return w


def random_daily_life(w: np.ndarray) -> np.ndarray:
    ops = [lambda x: x, phone_band, lambda x: add_noise(x, random.uniform(10, 30)),
           lambda x: codec(x, "opus"), lambda x: codec(x, "mp3"),
           lambda x: add_noise(phone_band(x), random.uniform(15, 30))]
    w = random.choice(ops)(w)
    return (w * random.uniform(0.3, 1.0)).astype(np.float32)
