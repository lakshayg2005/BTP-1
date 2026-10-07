"""Image expert: runs the AI detector, forensics and metadata checks, fuses them into a 3-way verdict.

Verdict rules are deliberately simple and explicit (thresholds in configs/image.yaml), so every decision can
be traced to evidence. "uncertain" is returned when evidence is weak or conflicting.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from src.common.evidence import Evidence, Verdict
from src.image import forensics
from src.image.ai_detector import AIImageDetector

DEFAULT_THRESHOLDS = {"ai_high": 0.80, "ai_low": 0.30, "edit_high": 0.70, "edit_low": 0.40}


def _save_map(m: np.ndarray, size: tuple[int, int], path: str) -> str:
    im = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    im.save(path)
    return path


class ImageExpert:
    def __init__(self, head_path: str, thresholds: dict | None = None, out_dir: str = "outputs",
                 tamper_path: str | None = None):
        self.detector = AIImageDetector(head_path)
        self.tamper = None
        if tamper_path and os.path.exists(tamper_path):
            from src.image.tamper import TamperLocalizer
            self.tamper = TamperLocalizer(tamper_path, self.detector.encoder)
        self.t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def analyze(self, path: str, with_maps: bool = True) -> Verdict:
        img = Image.open(path).convert("RGB")
        stem = os.path.splitext(os.path.basename(path))[0]
        ev: list[Evidence] = []
        artifacts: dict[str, str] = {}

        # 1. learned AI-generation score
        p_ai = self.detector.prob_ai(img)
        ev.append(Evidence("image.ai_score", "clip_detector", "score", round(p_ai, 3),
                           "ai_generated" if p_ai >= 0.5 else "real", abs(p_ai - 0.5) * 2,
                           f"The learned detector gives a {p_ai:.0%} probability that the whole image is AI-generated."))

        # 2. local edit / splice signals
        ela = forensics.ela_map(img)
        noise = forensics.noise_map(img)
        s_ela, s_noise = forensics.inconsistency_score(ela), forensics.inconsistency_score(noise)
        edit_score = max(s_ela, s_noise)
        if self.tamper is not None:
            tmap, s_learn = self.tamper.predict(img)
            # learned localizer is the primary edit signal; classical maps only add weak support
            edit_score = max(s_learn, 0.6 * edit_score)
            ys, xs = np.unravel_index(np.argmax(tmap), tmap.shape)
            where = f"{'top' if ys < 6 else 'bottom' if ys > 9 else 'middle'}-{'left' if xs < 6 else 'right' if xs > 9 else 'centre'}"
            ev.append(Evidence("image.tamper_localizer", "patch_localizer", "map", round(s_learn, 3),
                               "edited" if s_learn >= self.t["edit_low"] else "neutral", s_learn,
                               (f"The edit localizer finds a region that looks pasted or altered, strongest in the {where} of the image (score {s_learn:.2f})."
                                if s_learn >= self.t["edit_low"] else f"The edit localizer finds no altered region (score {s_learn:.2f}).")))
            if with_maps:
                artifacts["tamper_map"] = _save_map(tmap, img.size, f"{self.out_dir}/{stem}_tamper.png")
        ev.append(Evidence("image.ela_inconsistency", "ela", "score", round(s_ela, 3),
                           "edited" if s_ela >= self.t["edit_low"] else "neutral", s_ela,
                           f"Error-level analysis shows {'a region that re-compresses differently from the rest' if s_ela >= self.t['edit_low'] else 'uniform compression across the image'} (score {s_ela:.2f})."))
        ev.append(Evidence("image.noise_inconsistency", "noise_residual", "score", round(s_noise, 3),
                           "edited" if s_noise >= self.t["edit_low"] else "neutral", s_noise,
                           f"Sensor-noise level is {'inconsistent between regions' if s_noise >= self.t['edit_low'] else 'consistent across the image'} (score {s_noise:.2f})."))

        # 3. metadata / provenance
        meta = forensics.read_metadata(path)
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
            artifacts["ela_map"] = _save_map(ela, img.size, f"{self.out_dir}/{stem}_ela.png")

        label, conf, probs = self._fuse(p_ai, edit_score, meta)
        return Verdict(label, conf, probs, ev, artifacts=artifacts)

    def _fuse(self, p_ai: float, edit: float, meta: dict) -> tuple[str, float, dict]:
        t = self.t
        if meta["generator_tag"]:
            p_ai = max(p_ai, 0.9)
        p_edit = (1 - p_ai) * edit
        p_real = (1 - p_ai) * (1 - edit)
        probs = {"real": round(p_real, 3), "edited": round(p_edit, 3), "ai_generated": round(p_ai, 3)}
        if p_ai >= t["ai_high"]:
            return "ai_generated", p_ai, probs
        if p_ai <= t["ai_low"] and edit >= t["edit_high"]:
            return "edited", p_edit / (p_edit + p_real), probs
        if p_ai <= t["ai_low"] and edit <= t["edit_low"]:
            return "real", p_real / (p_edit + p_real), probs
        return "uncertain", max(probs.values()), probs
