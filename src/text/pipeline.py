"""Text expert: whole-text and per-sentence AI probability -> human / mixed (AI-edited) / AI / uncertain.

For text, "edited" means mixed authorship: some sentences look machine-written, others human-written.
Short texts (< ~50 words) are always reported with an explicit low-reliability warning, since every
known detector is unreliable there (Sadasivan et al. 2023; Liang et al. 2023 bias against non-native writers).
"""
from __future__ import annotations

import joblib
import numpy as np

from src.common.evidence import Evidence, Verdict
from src.text.detector import LikelihoodScorer, split_sentences

DEFAULT_THRESHOLDS = {"ai_high": 0.85, "ai_low": 0.25, "sent_ai": 0.80, "mixed_low": 0.2, "mixed_high": 0.7, "min_words": 50}


class TextExpert:
    def __init__(self, head_path: str, thresholds: dict | None = None, scorer: LikelihoodScorer | None = None):
        bundle = joblib.load(head_path)
        self.head = bundle["head"]
        # the head only works with the language-model pair it was trained on, which is stored in the bundle
        obs = bundle.get("encoder")
        self.scorer = scorer or (LikelihoodScorer(obs, bundle.get("performer", f"{obs}-Instruct")) if obs else LikelihoodScorer())
        self.t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    def _p(self, feats) -> float:
        return float(self.head.predict_proba(np.array([feats], np.float32))[0, 1])

    def analyze(self, text: str) -> Verdict:
        t = self.t
        whole = self.scorer.score(text)
        if whole["features"] is None:
            return Verdict("uncertain", 0.0, {}, [Evidence("text.too_short", "length", "score", len(text.split()), "neutral", 0.0,
                                                           "The text is too short to analyse.")])
        p_ai = self._p(whole["features"])
        sents = split_sentences(text)
        sent_scores = []
        for s in sents[:40]:
            r = self.scorer.score(s)
            if r["features"] is not None:
                sent_scores.append((s, self._p(r["features"])))
        ai_frac = np.mean([p >= t["sent_ai"] for _, p in sent_scores]) if sent_scores else 0.0
        n_words = len(text.split())

        # the most "too predictable" tokens: low surprisal where the model was confident
        low = sorted(whole["tokens"], key=lambda x: x[1])[: 15]
        ev = [
            Evidence("text.ai_score", "likelihood_head", "score", round(p_ai, 3), "ai_generated" if p_ai >= 0.5 else "real",
                     abs(p_ai - 0.5) * 2, f"Word-choice statistics give a {p_ai:.0%} probability that the text is AI-generated."),
            Evidence("text.binoculars", "binoculars", "score", round(whole["features"][0], 3), "neutral", 0.0,
                     f"Binoculars score {whole['features'][0]:.2f} (lower means more machine-like word choices)."),
            Evidence("text.sentences", "likelihood_head", "tokens",
                     [(s[:80], round(p, 2)) for s, p in sorted(sent_scores, key=lambda x: -x[1])[:5]],
                     "edited" if t["mixed_low"] <= ai_frac <= t["mixed_high"] else ("ai_generated" if ai_frac > t["mixed_high"] else "real"),
                     float(ai_frac), f"{ai_frac:.0%} of {len(sent_scores)} sentences look machine-written on their own."),
        ]
        if n_words < t["min_words"]:
            ev.append(Evidence("text.short_warning", "length", "score", n_words, "neutral", 0.0,
                               f"Only {n_words} words: AI-text detection is unreliable on short texts."))

        p_mixed = (1 - p_ai) * ai_frac if t["mixed_low"] <= ai_frac <= t["mixed_high"] else 0.0
        probs = {"real": round(max(0.0, 1 - p_ai - p_mixed), 3), "edited": round(p_mixed, 3), "ai_generated": round(p_ai, 3)}
        if n_words < t["min_words"]:
            label, conf = "uncertain", max(probs.values())
        elif p_ai >= t["ai_high"]:
            label, conf = "ai_generated", p_ai
        elif t["mixed_low"] <= ai_frac <= t["mixed_high"] and len(sent_scores) >= 4:
            label, conf = "edited", max(p_mixed, 0.5)
        elif p_ai <= t["ai_low"] and ai_frac < t["mixed_low"]:
            label, conf = "real", probs["real"]
        else:
            label, conf = "uncertain", max(probs.values())
        return Verdict(label, conf, probs, ev, artifacts={"predictable_tokens": str([w for w, _, _ in low])})
