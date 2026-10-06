# BTP-1 Research Notes: Explainable AI-Content Detection

Project: detect and explain real vs. edited vs. AI-generated content across image, video, audio and text.
Assumption for now: **no GPU** (CPU only, plus free Kaggle/Colab tiers if needed).

Verification status key:
- **[Abstract read]**: I fetched the arXiv abstract this session.
- **[Snippet only]**: seen in a search result, not opened.
- **[Memory]**: recalled from training knowledge, not re-checked online. Verify before citing.

---

## A. Anchor paper (in repo: Omni-Fake.pdf)

| Paper | Date | Status | Role in our project |
|---|---|---|---|
| Omni-Fake: Benchmarking Unified Multimodal Social Media Deepfake Detection (Li, Huang, et al.) arXiv 2605.01638 | 2 May 2026 | Abstract read (PDF not yet read) | Main base paper. Image/audio/video/talking-head, 1M+ samples, OOD set 200k+, Omni-Fake-R1 (RL detector with localization and explanation). Gaps: no text, no explicit "edited" class. |

## B. Recent related papers (2025-2026)

| Paper | Date | Status | Role |
|---|---|---|---|
| BusterX++ (arXiv 2507.14632) | v1 Jul 2025, rev Jun 2026 | Abstract read | Unified image+video MLLM, pure RL beats SFT+RL. Shows the RL-explainer pattern. |
| Ivy-Fake (arXiv 2506.00979), ICMR 2026 | v1 Jun 2025, v6 May 2026 | Abstract read | 106K annotated samples, GRPO reasoning chains. Annotation format idea. |
| EDVD-LLaMA (arXiv 2510.16442) | Oct 2025 | Abstract read | Explainable deepfake video, chain-of-thought with face constraints. |
| XAI-Grounded Explanation Generation for Speech Deepfake Detection (arXiv 2606.16137) | 15 Jun 2026 | Abstract read | KEY for our no-GPU design: feed gradient/XAI evidence to an LLM without training, to ground explanations. Uses PartialSpoof. |
| TELL: Show, Don't TELL: Explainable AI-Generated Text Detection (arXiv 2605.27921) | 27 May 2026 | Abstract read | Text explanation via highlighted features. AUROC 0.927. |
| Explainable Deepfake Detection with Feature-robust Augmentation and Evidence-grounded Explanation Optimization (arXiv 2608.20913) | 21 Aug 2026 | Abstract read | ACM MM 2026 explainable deepfake challenge winner. Preference optimization on explanations. Idea: evidence-grounded explanation scoring. |
| DailyBench (arXiv 2607.24016) | Jul 2026, rev Sep 2026 | Abstract read | FakeBench + ManipulationBench. Real vs. synthetic vs. edited images. Detectors drop from 91-96% on GenImage to 52-79%. |
| UniAIDet (arXiv 2510.23023) | Oct 2025 | Snippet only | Image AIGC detection and localization benchmark. |
| DetectZoo (arXiv 2606.04205) | Jun 2026 | Snippet only | Toolkit for text/audio/image detectors. Possible code reuse. |
| Memory-Anchored Multimodal Reasoning for Explainable Video Forensics (arXiv 2508.14581) | Aug 2025 | Snippet only | Explainable video forensics. |
| Integrated explainable framework for image, audio, video (ScienceDirect S2090447926002649) | 2026 | Snippet only | Non-LLM baseline: cross-attention fusion + SHAP/LIME/LRP/Grad-CAM. |
| Investigating MLLMs for Audio Deepfake Detection (arXiv 2601.00777) | Jan 2026 | Snippet only | Audio LLM viability. |
| AIGVDBench, MintVid, Video as Natural Augmentation (arXiv 2605.21977) | 2026 | Snippet only | Video benchmarks and unified image+video detection. |

## C. Earlier foundations (all [Memory], verify before citing)

Image: Wang et al. CNN-generated images easy to spot (CVPR 2020); Frank et al. frequency analysis (ICML 2020); UnivFD, Ojha et al. (CVPR 2023); Corvi et al. diffusion detection (ICASSP 2023); DIRE (ICCV 2023); NPR (CVPR 2024); AIDE / Chameleon (ICLR 2025); GenImage (NeurIPS 2023 D&B).
Edited-image forensics: RGB-N (CVPR 2018); ManTra-Net (CVPR 2019); TruFor (CVPR 2023).
Video: FaceForensics++ (ICCV 2019); Celeb-DF (CVPR 2020); DFDC (2020); Face X-ray (CVPR 2020); LipForensics (CVPR 2021); RealForensics (CVPR 2022); SBI (CVPR 2022); AltFreezing (CVPR 2023); DeepfakeBench (NeurIPS 2023 D&B); DeMamba/GenVideo (2024).
Audio: ASVspoof 2019 / ASVspoof 5; RawNet2 (ICASSP 2021); AASIST (ICASSP 2022); wav2vec2+augmentation, Tak et al. (Odyssey 2022); WaveFake (NeurIPS D&B 2021); In-the-Wild, Muller et al. (Interspeech 2022); SHAP for spoofing, Ge et al. (Odyssey 2022).
Text: GLTR (ACL 2019 demo); DetectGPT (ICML 2023); Fast-DetectGPT (ICLR 2024); Binoculars (ICML 2024); Ghostbuster (NAACL 2024); Kirchenbauer watermark (ICML 2023); SynthID-Text (Nature 2024); RAID (ACL 2024); Sadasivan et al. 2023; Liang et al. bias against non-native writers (Patterns 2023).
Multimodal / unified: LOKI (ICLR 2025); FakeShield (ICLR 2025); SIDA (CVPR 2025); DD-VQA / Common Sense Reasoning for Deepfake Detection (ECCV 2024); DGM4 (CVPR 2023).
XAI: LIME (KDD 2016); Grad-CAM (ICCV 2017); SHAP (NeurIPS 2017). Provenance: C2PA.

---

## D0. Compute

Training/feature extraction runs on a remote RTX 4070 Laptop GPU (8 GB VRAM, ~4 GB free RAM, limited disk).
Access details and the cleanup checklist are kept locally in LOCAL_ACCESS_NOTES.md (git-ignored).

## D. Design decisions log (update as we go)

- 2026-10-07: Assume no GPU. Use frozen pretrained encoders + light classifier heads trained on CPU; ground LLM explanations in XAI evidence (idea from arXiv 2606.16137) instead of training an RL MLLM like Omni-Fake-R1.
- Novelty targets: (1) text modality added to an Omni-Fake-style system; (2) explicit 3-way real / edited / AI-generated label (DailyBench motivation); (3) evidence-grounded explanations with faithfulness check.
