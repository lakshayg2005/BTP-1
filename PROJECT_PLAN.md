# BTP-1 Project Plan: Explainable Detection of AI-Generated, Edited and Real Content

Working title: **"Evidence-grounded, explainable detection of AI-generated and edited media in daily-life content (image, video, audio, text)"**

Hardware: friend's RTX 4070 Laptop (8 GB VRAM) over SSH for heavy jobs; my laptop (CPU, 8 GB RAM) for coding and the demo.
Paper references: see RESEARCH_NOTES.md.

---

## 1. Goal and what makes it "daily-life"

Input: any image, video, audio clip or text that a person meets on WhatsApp, Instagram, YouTube, news sites, etc.
Output, for every input:
1. **Verdict**: Real / Edited (real content, partly changed) / AI-generated / Uncertain.
2. **Confidence**: a calibrated probability, not a raw score.
3. **Where**: a heatmap, mask, time segment or highlighted words.
4. **Why**: a short plain-English explanation that only cites evidence the system actually measured.

Daily-life content is hard because it is compressed, resized, screenshotted, re-encoded and made by the newest generators. DailyBench (2026) shows detectors drop from 91-96% to 52-79% on such content. So the plan centres on:
- training and testing on modern generators and social-media degradations;
- a held-out daily-life test set we collect ourselves;
- an **"Uncertain" option**, so the system abstains instead of being confidently wrong.

---

## 2. Architecture: evidence-grounded multi-expert system

```
input -> modality router
          |-- Image expert:  AI-gen detector  + edit/tamper localizer + metadata/provenance check
          |-- Video expert:  frame-level image expert + temporal consistency + face/lip-sync (talking heads)
          |-- Audio expert:  self-supervised speech features + spoof head + segment localization
          |-- Text expert:   zero-shot likelihood detectors + fine-tuned classifier + token highlights
          v
   Evidence JSON (scores, maps, segments, tokens, metadata)
          v
   Fusion + calibration -> 3-way verdict or "Uncertain"
          v
   Explainer: vision-language model / LLM writes explanation from the evidence only
          v
   Faithfulness check: every claim must map to an evidence item, else rewrite or drop
          v
   Web demo (upload -> verdict, confidence, heatmap, explanation)
```

Why this design instead of one big model like Omni-Fake-R1:
- Omni-Fake-R1, BusterX++ and Ivy-Fake train large multimodal LLMs with RL on data-center GPUs. That does not fit in 8 GB.
- Specialist experts + an evidence-grounded explainer fit our hardware, are easier to debug, and give explanations tied to real measurements (idea from arXiv 2606.16137 and the ACM MM 2026 challenge winner, arXiv 2608.20913).
- If more GPU becomes available, the explainer can be upgraded to an RL-trained VLM without changing the rest.

---

## 3. Per-modality design

| Modality | Detector | Localization / evidence | Key papers |
|---|---|---|---|
| Image: AI-gen | Frozen strong vision encoder (CLIP / SigLIP / DINOv2 family) + small trained head, plus a low-level artifact branch (frequency / up-sampling traces). Train with JPEG, resize, screenshot and social-media augmentations. | Saliency map, frequency map | UnivFD, NPR, AIDE, DailyBench |
| Image: edited | Pretrained tamper localizer (TruFor-style), inference only | Tamper mask | TruFor, DailyBench ManipulationBench, FakeShield, SIDA |
| Image: provenance | EXIF / C2PA content credentials when present | Metadata facts | C2PA |
| Video | Image expert on sampled frames + temporal consistency features; face-swap expert for face videos; audio-visual sync for talking heads | Per-frame score timeline, face regions | Omni-Fake, EDVD-LLaMA, DeepfakeBench, AltFreezing |
| Audio | Frozen self-supervised speech encoder (wav2vec2 / WavLM / XLS-R) + AASIST-style head; train with codec, noise and phone-call augmentation | Suspicious time segments, spectrogram saliency | AASIST, wav2vec2-AASIST, In-the-Wild, XAI-grounded speech explanations |
| Text | Zero-shot detectors (Fast-DetectGPT, Binoculars with small 4-bit models) + fine-tuned small classifier; ensemble | Per-token highlights | Fast-DetectGPT, Binoculars, TELL, RAID |

Text has no clean "edited" class. Instead we report: human / AI-generated / mixed (AI-assisted or paraphrased), with highlighted spans.

---

## 4. Data

- Training (subsets that fit our disk): Omni-Fake-Set, Ivy-Fake, DailyBench-style real/synthetic/edited images, ASVspoof 5 + In-the-Wild audio, RAID text, FaceForensics++ / Celeb-DF video.
- Testing: Omni-Fake-OOD, DailyBench, RAID, In-the-Wild.
- **Our contribution: a small "Daily-Life Test Set"**, a few hundred items we collect: real phone photos and voice notes, the same items sent through WhatsApp/Instagram, edits done with phone editors, and outputs from current popular generators. This tests exactly the target use case.
- Raw datasets live on my laptop (E:/F:). Only compact cached features go to the GPU machine (disk there is 93% full).

---

## 5. Evaluation

- Detection: accuracy, AUROC, F1 per class, and accuracy on unseen generators (OOD).
- Robustness: performance after JPEG, resize, screenshot and re-encoding.
- Calibration: expected calibration error; accuracy vs. coverage when "Uncertain" is allowed.
- Localization: IoU / F1 for masks, segment overlap for audio.
- Explanation: faithfulness (claims supported by evidence), plus a small human study rating usefulness.
- Ablations: each expert removed, with vs. without augmentation, with vs. without grounding.

---

## 6. Phases and timeline (part-time, about 18-20 weeks)

| Phase | Weeks | Output |
|---|---|---|
| 0. Setup: GPU env, read Omni-Fake fully, download data subsets | 1-2 | Working pipeline skeleton |
| 1. Image AI-gen expert with robustness augmentations | 3 | Model + DailyBench / Omni-Fake numbers |
| 2. Image edit localizer + provenance + 3-way fusion | 2 | Real / edited / AI verdict with masks |
| 3. Explainer v1 (evidence JSON -> LLM) + faithfulness checker | 2 | Image explanations |
| 4. Audio expert | 2 | Model + segment localization |
| 5. Text expert | 1-2 | Ensemble + token highlights |
| 6. Video expert (reuses image + audio) | 2 | Frame timeline, talking-head check |
| 7. Daily-life test set collection + full evaluation | 2 | Results tables, ablations |
| 8. Web demo, report, slides | 2 | Final deliverables |
| Stretch (if more GPU) | - | LoRA / GRPO fine-tune of a small VLM explainer on Ivy-Fake / Omni-Fake explanations |

Image comes first because video reuses it and it has the best data.

---

## 7. Honest limits

- No detector is always correct; newer generators keep arriving. We address this with calibrated confidence, an "Uncertain" output and OOD tests, not by claiming perfect accuracy.
- Text detection is the least reliable modality, especially for short or heavily edited text and for non-native writers. The demo must say so.
- 8 GB VRAM rules out reproducing Omni-Fake-R1 training; our contribution is the grounded multi-expert design, the 3-way label, the text modality and the daily-life test set.
