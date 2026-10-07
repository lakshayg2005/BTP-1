"""Audio expert: AI-speech probability, suspicious time segments, bandwidth context -> verdict.

Audio "edited" = real speech where only some segments are synthetic or spliced (partial spoof, as in
PartialSpoof). We flag it when a whole-clip score is low-to-mid but some windows score high.
"""
from __future__ import annotations

from src.audio.detector import AIAudioDetector, bandwidth_hz, load_audio, SR
from src.common.evidence import Evidence, Verdict

DEFAULT_THRESHOLDS = {"ai_high": 0.80, "ai_low": 0.30, "seg_high": 0.85, "seg_frac_edit": 0.10}


class AudioExpert:
    def __init__(self, head_path: str, thresholds: dict | None = None):
        self.detector = AIAudioDetector(head_path)
        self.t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    def analyze(self, path: str) -> Verdict:
        wav = load_audio(path, max_seconds=120)
        p_ai = self.detector.prob_ai(wav)
        segs = self.detector.segment_scores(wav)
        hot = [s for s in segs if s[2] >= self.t["seg_high"]]
        frac = len(hot) / max(1, len(segs))
        bw = bandwidth_hz(wav)
        ev = [
            Evidence("audio.ai_score", "speech_detector", "score", round(p_ai, 3),
                     "ai_generated" if p_ai >= 0.5 else "real", abs(p_ai - 0.5) * 2,
                     f"The speech detector gives a {p_ai:.0%} probability that this voice is AI-generated or cloned."),
            Evidence("audio.segments", "speech_detector", "segments", [(round(a, 1), round(b, 1), round(p, 2)) for a, b, p in hot],
                     "edited" if 0 < frac < 0.6 else ("ai_generated" if frac >= 0.6 else "real"), frac,
                     (f"{len(hot)} of {len(segs)} two-second windows sound synthetic"
                      + (f", e.g. {hot[0][0]:.0f}-{hot[0][1]:.0f}s." if hot else "."))),
            Evidence("audio.bandwidth", "spectrum", "score", round(bw), "neutral", 0.0,
                     f"Most audio energy lies below {bw/1000:.1f} kHz"
                     + (" (narrow band, typical of phone calls, voice notes or some TTS)." if bw < 4500 else ".")),
        ]
        label, conf, probs = self._fuse(p_ai, frac)
        return Verdict(label, conf, probs, ev, artifacts={"segments": str([(round(a, 1), round(b, 1), round(p, 2)) for a, b, p in segs])})

    def _fuse(self, p_ai: float, frac: float):
        t = self.t
        p_edit = (1 - p_ai) * min(1.0, frac / 0.3)
        p_real = max(0.0, 1 - p_ai - p_edit)
        probs = {"real": round(p_real, 3), "edited": round(p_edit, 3), "ai_generated": round(p_ai, 3)}
        if p_ai >= t["ai_high"]:
            return "ai_generated", p_ai, probs
        if p_ai < t["ai_high"] and frac >= t["seg_frac_edit"] and frac < 0.6:
            return "edited", max(p_edit, 0.5), probs
        if p_ai <= t["ai_low"] and frac == 0:
            return "real", p_real, probs
        return "uncertain", max(probs.values()), probs
