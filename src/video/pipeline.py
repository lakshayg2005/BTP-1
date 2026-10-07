"""Video expert: composes the image and audio experts over time (no video-specific training).

- Samples N frames, scores each with the AI-image detector and the edit localizer -> per-frame timeline.
- Temporal evidence: share of AI-looking frames and how much the score jumps between frames
  (a short burst of high scores suggests a partly manipulated clip).
- Audio track (if present and ffmpeg is available) goes through the audio expert. A real-looking video with
  a synthetic voice is the common "dubbed / voice-cloned" daily-life fake, reported as EDITED.
Public AI-video sets (GenVideo, GenVidBench) are 15-45 GB, beyond our budget, so video is evaluated qualitatively.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image

from src.common.evidence import Evidence, Verdict

DEFAULT_THRESHOLDS = {"ai_high": 0.75, "ai_low": 0.30, "frame_ai": 0.80, "burst_frac": 0.2}


def sample_frames(path: str, n: int = 16) -> tuple[list[Image.Image], list[float]]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames, times = [], []
    for idx in np.linspace(0, total - 1, min(n, total)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
            times.append(idx / fps)
    cap.release()
    return frames, times


def extract_audio(path: str) -> str | None:
    if not shutil.which("ffmpeg"):
        return None
    out = os.path.join(tempfile.mkdtemp(), "audio.wav")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", out],
                       capture_output=True)
    return out if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 16000 else None


class VideoExpert:
    def __init__(self, image_expert, audio_expert=None, thresholds: dict | None = None, n_frames: int = 16):
        self.img = image_expert
        self.aud = audio_expert
        self.t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.n = n_frames

    def analyze(self, path: str) -> Verdict:
        t = self.t
        frames, times = sample_frames(path, self.n)
        if not frames:
            return Verdict("uncertain", 0.0, {}, [Evidence("video.unreadable", "decoder", "score", 0, "neutral", 0.0,
                                                           "The video could not be decoded.")])
        det = self.img.detector
        p = np.array([det.prob_ai(f) for f in frames])
        edit = np.array([self.img.tamper.predict(f)[1] for f in frames]) if self.img.tamper else np.zeros(len(frames))
        ai_frac = float((p >= t["frame_ai"]).mean())
        jump = float(np.abs(np.diff(p)).max()) if len(p) > 1 else 0.0
        worst = int(np.argmax(p))
        ev = [
            Evidence("video.ai_mean", "frame_detector", "score", round(float(p.mean()), 3),
                     "ai_generated" if p.mean() >= 0.5 else "real", abs(float(p.mean()) - 0.5) * 2,
                     f"Across {len(frames)} sampled frames the average AI-generation probability is {p.mean():.0%}."),
            Evidence("video.timeline", "frame_detector", "segments", [(round(a, 1), round(float(b), 2)) for a, b in zip(times, p)],
                     "edited" if 0 < ai_frac < 0.6 else "neutral", ai_frac,
                     f"{ai_frac:.0%} of frames look AI-generated; the most suspicious frame is at {times[worst]:.1f}s ({p[worst]:.0%})."),
            Evidence("video.temporal_jump", "frame_detector", "score", round(jump, 3), "edited" if jump >= 0.5 else "neutral",
                     min(1.0, jump), f"The largest frame-to-frame change in AI score is {jump:.2f}"
                     + (" (a sudden switch, as when part of a clip was replaced)." if jump >= 0.5 else ".")),
        ]
        if self.img.tamper:
            ev.append(Evidence("video.frame_edits", "patch_localizer", "score", round(float(edit.max()), 3),
                               "edited" if edit.max() >= 0.7 else "neutral", float(edit.max()),
                               f"The strongest edited-region score in any frame is {edit.max():.2f}."))

        audio_v = None
        if self.aud is not None and (wav := extract_audio(path)):
            audio_v = self.aud.analyze(wav)
            pa = audio_v.probabilities.get("ai_generated", 0.0)
            ev.append(Evidence("video.audio_ai", "speech_detector", "score", round(pa, 3),
                               "ai_generated" if pa >= 0.5 else "real", abs(pa - 0.5) * 2,
                               f"The soundtrack's voice has a {pa:.0%} probability of being AI-generated or cloned."))

        label, conf, probs = self._fuse(float(p.mean()), ai_frac, float(edit.max()), audio_v)
        return Verdict(label, conf, probs, ev, artifacts={"frame_times": str([round(x, 1) for x in times])})

    def _fuse(self, p_mean, ai_frac, edit_max, audio_v):
        t = self.t
        pa = audio_v.probabilities.get("ai_generated", 0.0) if audio_v else 0.0
        p_edit = (1 - p_mean) * max(edit_max if edit_max >= 0.7 else 0.0, pa if pa >= 0.8 else 0.0,
                                    1.0 if 0 < ai_frac < 0.6 and ai_frac >= t["burst_frac"] else 0.0)
        p_real = max(0.0, 1 - p_mean - p_edit)
        probs = {"real": round(p_real, 3), "edited": round(p_edit, 3), "ai_generated": round(p_mean, 3)}
        if p_mean >= t["ai_high"]:
            return "ai_generated", p_mean, probs
        if p_mean < t["ai_high"] and p_edit >= 0.5:
            return "edited", p_edit, probs
        if p_mean <= t["ai_low"] and p_edit < 0.2:
            return "real", p_real, probs
        return "uncertain", max(probs.values()), probs
