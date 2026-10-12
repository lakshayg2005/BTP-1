"""Image expert: runs the AI detector, edit localizer, forensics and metadata checks, then fuses them into a
3-way verdict (real / edited / AI-generated) or "uncertain".

Fusion is a small learned 3-class classifier over the detector scores (models/fusion_head.joblib, trained by
scripts/train_fusion.py on a held-out calibration half of the test sets). It replaced hand-set thresholds,
which labelled most real photos as edited. "uncertain" = the fusion's top probability is below a threshold.
Without a fusion head, a conservative rule-based fallback is used.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
from PIL import Image

from src.common.evidence import Evidence, Verdict
from src.image import forensics
from src.image.ai_detector import AIImageDetector

CLASSES = ["real", "edited", "ai_generated"]
FEATURES = ["p_ai", "tamper_top8", "tamper_mean", "tamper_frac50", "ela", "noise", "meta_generator", "meta_editor", "meta_camera"]
FALLBACK = {"ai_high": 0.80, "ai_low": 0.30, "edit_high": 0.85, "tau": 0.55, "tamper_flag": 0.70}


def _save_map(m: np.ndarray, size: tuple[int, int], path: str) -> str:
    im = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    im.save(path)
    return path


class ImageExpert:
    def __init__(self, head_path: str, thresholds: dict | None = None, out_dir: str = "outputs",
                 tamper_path: str | None = None, fusion_path: str | None = "auto"):
        self.detector = AIImageDetector(head_path)
        self.tamper = None
        if tamper_path and os.path.exists(tamper_path):
            from src.image.tamper import TamperLocalizer
            self.tamper = TamperLocalizer(tamper_path, self.detector.encoder)
        if fusion_path == "auto":
            fusion_path = os.path.join(os.path.dirname(head_path) or ".", "fusion_head.joblib")
        self.fusion = joblib.load(fusion_path) if fusion_path and os.path.exists(fusion_path) else None
        self.t = {**FALLBACK, **(thresholds or {})}
        if self.fusion:
            self.t["tau"] = self.fusion["tau"]
            self.t["tamper_flag"] = self.fusion["tamper_flag"]
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def features(self, img: Image.Image, path: str) -> tuple[dict, dict]:
        """Numeric scores used by the fusion, plus intermediate maps for evidence and artifacts."""
        p_ai = self.detector.prob_ai(img)
        ela = forensics.ela_map(img)
        s_ela, s_noise = forensics.inconsistency_score(ela), forensics.inconsistency_score(forensics.noise_map(img))
        tmap, s_learn = (self.tamper.predict(img) if self.tamper is not None else (np.zeros((16, 16)), 0.0))
        meta = forensics.read_metadata(path)
        f = {"p_ai": p_ai, "tamper_top8": s_learn, "tamper_mean": float(tmap.mean()), "tamper_frac50": float((tmap > 0.5).mean()),
             "ela": s_ela, "noise": s_noise, "meta_generator": float(bool(meta["generator_tag"])),
             "meta_editor": float(bool(meta["editor_tag"])), "meta_camera": float(bool(meta["camera"]))}
        return f, {"ela": ela, "tmap": tmap, "meta": meta}

    def analyze(self, path: str, with_maps: bool = True) -> Verdict:
        img = Image.open(path).convert("RGB")
        stem = os.path.splitext(os.path.basename(path))[0]
        f, inter = self.features(img, path)
        tmap, meta = inter["tmap"], inter["meta"]
        p_ai, s_learn = f["p_ai"], f["tamper_top8"]
        flag = self.t["tamper_flag"]
        ev: list[Evidence] = []
        artifacts: dict[str, str] = {}

        ev.append(Evidence("image.ai_score", "clip_detector", "score", round(p_ai, 3),
                           "ai_generated" if p_ai >= 0.5 else "real", abs(p_ai - 0.5) * 2,
                           f"The learned detector gives a {p_ai:.0%} probability that the whole image is AI-generated."))
        if self.tamper is not None:
            ys, xs = np.unravel_index(np.argmax(tmap), tmap.shape)
            g = tmap.shape[0]
            where = f"{'top' if ys < g * 0.375 else 'bottom' if ys >= g * 0.625 else 'middle'}-{'left' if xs < g * 0.375 else 'right' if xs >= g * 0.625 else 'centre'}"
            ev.append(Evidence("image.tamper_localizer", "patch_localizer", "map", round(s_learn, 3),
                               "edited" if s_learn >= flag else "real", min(1.0, abs(s_learn - flag) / max(flag, 1 - flag)),
                               (f"The edit localizer finds a region that looks pasted or altered, strongest in the {where} of the image (score {s_learn:.2f})."
                                if s_learn >= flag else f"The edit localizer finds no clearly altered region (score {s_learn:.2f}).")))
            if with_maps:
                artifacts["tamper_map"] = _save_map(tmap, img.size, f"{self.out_dir}/{stem}_tamper.png")
        # classical forensics fire on many untouched photos, so they are reported as context only
        ev.append(Evidence("image.ela_inconsistency", "ela", "score", round(f["ela"], 3), "neutral", 0.0,
                           f"Error-level analysis inconsistency score is {f['ela']:.2f} (context only; often high on untouched photos too)."))
        ev.append(Evidence("image.noise_inconsistency", "noise_residual", "score", round(f["noise"], 3), "neutral", 0.0,
                           f"Sensor-noise inconsistency score is {f['noise']:.2f} (context only; often high on untouched photos too)."))
        if meta["generator_tag"]:
            ev.append(Evidence("image.meta_generator", "metadata", "metadata", meta["generator_tag"], "ai_generated", 0.9,
                               f"File metadata mentions an AI generator ('{meta['generator_tag']}')."))
        if meta["editor_tag"]:
            ev.append(Evidence("image.meta_editor", "metadata", "metadata", meta["editor_tag"], "edited", 0.5,
                               f"File metadata shows it was saved by editing software ('{meta['editor_tag']}')."))
        if meta["camera"]:
            ev.append(Evidence("image.meta_camera", "metadata", "metadata", meta["camera"], "real", 0.3,
                               f"Camera EXIF data is present ({meta['camera']}); this can be copied or faked, so it is weak evidence."))
        if meta["c2pa_manifest"]:
            ev.append(Evidence("image.meta_c2pa", "metadata", "metadata", True, "neutral", 0.0,
                               "A C2PA content-credentials manifest is present (not cryptographically verified here)."))
        if with_maps:
            artifacts["ai_saliency"] = _save_map(self.detector.saliency(img), img.size, f"{self.out_dir}/{stem}_ai_saliency.png")
            artifacts["ela_map"] = _save_map(inter["ela"], img.size, f"{self.out_dir}/{stem}_ela.png")

        label, conf, probs = self._fuse(f)
        v = Verdict(label, conf, probs, ev, artifacts=artifacts)
        v.features = f
        return v

    def _fuse(self, f: dict) -> tuple[str, float, dict]:
        if self.fusion:
            x = np.array([[f[k] for k in self.fusion["features"]]], np.float32)
            p = self.fusion["head"].predict_proba(x)[0]
            probs = {c: round(float(p[list(self.fusion["classes"]).index(c)]), 3) for c in CLASSES}
            if f["meta_generator"]:                       # explicit generator tag in metadata is strong evidence
                probs = {"real": 0.02, "edited": 0.03, "ai_generated": 0.95}
            label = max(probs, key=probs.get)
            conf = probs[label]
            return (label if conf >= self.t["tau"] else "uncertain"), conf, probs
        # fallback rules (no fusion head trained yet): conservative, prefers "uncertain"
        t, p_ai, edit = self.t, f["p_ai"], f["tamper_top8"]
        if f["meta_generator"]:
            p_ai = max(p_ai, 0.95)
        p_edit = (1 - p_ai) * edit
        p_real = (1 - p_ai) * (1 - edit)
        probs = {"real": round(p_real, 3), "edited": round(p_edit, 3), "ai_generated": round(p_ai, 3)}
        if p_ai >= t["ai_high"]:
            return "ai_generated", p_ai, probs
        if p_ai <= t["ai_low"] and edit >= t["edit_high"]:
            return "edited", p_edit / (p_edit + p_real), probs
        if p_ai <= t["ai_low"] and edit < t["tamper_flag"]:
            return "real", p_real / (p_edit + p_real), probs
        return "uncertain", max(probs.values()), probs
