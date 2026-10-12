# Viva reading list: papers, datasets and tools actually used

Read in this order. **Tier 1** is what the project is built on; expect questions on these.
IDs marked ✓ were checked online on 2026-10-08/11; IDs marked (check) are from memory, so verify them before citing.

---

## Tier 1: must read (the project stands on these)

### 1. Omni-Fake: our base paper ✓
- *Omni-Fake: Benchmarking Unified Multimodal Social Media Deepfake Detection*, Li, Huang et al., **CVPR 2026**, arXiv **2605.01638** (May 2026). PDF is in this repo.
- **What it does:** a 1M+ sample benchmark covering image, audio, video and talking-head content, and Omni-Fake-R1, an RL-trained multimodal LLM that detects, localizes and explains.
- **What we took:** the *problem definition* (detect + localize + explain, multimodal, social-media / daily-life content) and the out-of-distribution evaluation mindset.
- **What we did differently and why:** their model needs data-center GPUs (RL on a large MLLM), and the code is not released yet. We built a modular expert system that fits a free T4 GPU, and we **added text** and an **explicit "edited" class**, which Omni-Fake does not separate.
- *Likely Q:* "Why not reproduce Omni-Fake-R1?" No released code, and RL fine-tuning of a multimodal LLM needs far more than our 16 GB GPU. Our design is the feasible, interpretable alternative.

### 2. UnivFD: core of our image detector ✓
- *Towards Universal Fake Image Detectors that Generalize Across Generative Models*, Ojha, Li, Lee, **CVPR 2023**, arXiv **2302.10174**. Code: github.com/WisconsinAIVision/UniversalFakeDetect
- **Idea:** do not train a CNN from scratch. Use a **frozen CLIP ViT-L/14** and train only a **linear layer** on its features. This generalizes to unseen generators much better.
- **In our code:** `src/image/ai_detector.py` (CLIP + logistic-regression head), `src/baselines.py` (their released head as a baseline).
- *Likely Q:* "What is a linear probe / transfer learning?" The big model is frozen and used as a feature extractor; we train only a small classifier on its 768-number output.
- *Likely Q:* "Your training set is smaller than theirs. Why is yours better?" Theirs was trained on **ProGAN only (2019)**. Ours covers **74 modern generators** (OpenFake) and SID-Set, plus daily-life degradations. Coverage of generators matters more than raw size.

### 3. CLIP: the backbone (check)
- *Learning Transferable Visual Models From Natural Language Supervision*, Radford et al., **ICML 2021**, arXiv 2103.00020 (check).
- Know: trained on 400M image-text pairs; ViT-L/14 means a Vision Transformer, Large, with 14×14 pixel patches; a 224×224 input gives a 16×16 = 256 patch grid. **Our edit localizer classifies each of these 256 patches.**

### 4. SID-Set / SIDA: our 3-way benchmark ✓
- *SIDA: Social Media Image Deepfake Detection, Localization and Explanation with Large Multimodal Model*, Huang et al., **CVPR 2025**, arXiv **2412.04292**. Dataset: huggingface.co/datasets/saberzl/SID_Set (CC-BY-4.0).
- Same research group as Omni-Fake. Labels: **0 real, 1 fully synthetic, 2 tampered**, with **masks** of the tampered region. It is exactly our 3-class problem.
- **In our code:** `scripts/download_sidset.py`, `scripts/train_tamper.py` (masks supervise the edit localizer), `scripts/evaluate.py` (main 3-way test = their public **validation** split, because their test split is restricted).

### 5. OpenFake: our main AI-image training data ✓
- *OpenFake*, ComplexDataLab, arXiv **2509.09495** (2025). huggingface.co/datasets/ComplexDataLab/OpenFake (CC-BY-NC-4.0, non-commercial).
- 2.45M images, 75+ generators (SD family, Flux, Midjourney, DALL-E, Imagen, …); real photos from Pexels, LAION and Reddit, which is close to daily-life content.
- We sampled 3,000 real + 3,000 AI for training (from **74 generators**) and 500 + 500 for testing.

### 6. Binoculars and Fast-DetectGPT: our text detector ✓
- *Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text*, Hans et al., **ICML 2024**, arXiv **2401.12070**.
- *Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature*, Bao et al., **ICLR 2024**, arXiv **2310.05130**.
- **Binoculars idea:** perplexity of the text under one LLM, divided by the "cross-perplexity" between two LLMs. Machine text is unsurprising to LLMs, so the ratio is lower.
- **Fast-DetectGPT idea:** compare the log-likelihood of the actual tokens with what the model would expect from its own samples. Machine text sits near the model's peak.
- **In our code:** `src/text/detector.py` computes both from a **Qwen2.5-1.5B base + instruct pair** (the originals use 7B models, too big for us), plus a small calibrated classifier on the 4 scores.
- *Likely Q:* "Zero-shot?" They need no training on AI text; we only calibrate a 4-feature head on RAID.

### 7. RAID: our text benchmark ✓
- *RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors*, Dugan et al., **ACL 2024**, arXiv **2405.07940**. MIT license.
- 11 LLMs, 11 domains, **adversarial attacks** (paraphrase, synonym swap, homoglyphs…). It explains our result: clean-text AUROC 0.975 falls to **0.757 on attacked text**, a known weakness of all detectors.

### 8. Self-supervised speech features for spoof detection: our audio detector ✓
- *Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation*, Tak et al., **Odyssey 2022**, arXiv **2202.12233**. Code: github.com/TakHemlata/SSL_Anti-spoofing
- *Does Audio Deepfake Detection Generalize?*, Müller et al., **Interspeech 2022**, arXiv **2203.16263**. It shows detectors fail on real-world ("in-the-wild") audio, which is why we test on an unseen generator.
- **In our code:** `src/audio/detector.py` uses frozen **XLS-R 300M** (multilingual wav2vec 2.0) features from a middle layer, mean-pooled, plus a linear head, with 2-second windows for localization. We simplified their AASIST back-end to a linear head.

### 9. Grounded explanations: our explainer design ✓
- *XAI-Grounded Explanation Generation for Speech Deepfake Detection with Training-Free Multimodal LLMs*, Li et al., **Interspeech 2026**, arXiv **2606.16137**. Key idea: feed **measured XAI evidence** to an LLM instead of letting it guess from raw media.
- *Explainable Deepfake Detection with Feature-robust Augmentation and Evidence-grounded Explanation Optimization*, arXiv **2608.20913** (ACM MM 2026 challenge winner). It defines the two failure modes we check for: **omitted evidence** and **invented, irrelevant evidence**.
- **In our code:** `src/explain/llm.py`. The LLM sees only the evidence list. Every sentence must cite an evidence id and may only use numbers that were measured, or it falls back to `src/explain/template.py`.
- *Likely Q:* "How do you know the explanation is true?" The faithfulness check is automatic, and we report the pass rate in the results.

---

## Tier 2: should read (motivation, methods we borrowed, baselines)

| Paper | ID | Why it matters to us |
|---|---|---|
| **DailyBench**: real vs. AI vs. manipulated images from modern generators (2026) | 2607.24016 ✓ | Motivation: detectors drop from 91–96% to 52–79% on modern, real-world images, hence our daily-life augmentations and modern data |
| **Face X-ray** (Li et al., CVPR 2020) | 1912.13458 (check) | Idea of **making your own manipulations with known masks**, used for our synthetic edits (`src/image/splice.py`) |
| **Self-Blended Images (SBI)** (Shiohara & Yamasaki, CVPR 2022) | 2204.08376 (check) | Same self-blending idea |
| **Wang et al., "CNN-generated images are surprisingly easy to spot… for now"** (CVPR 2020) | 1912.11035 ✓ | The ProGAN training data behind the UnivFD baseline |
| **XLS-R** (Babu et al., 2021) | 2111.09296 (check) | The speech encoder we use |
| **Qwen2.5 technical report** (2024) | 2412.15115 (check) | LLMs used in the text detector and explainer |
| **GLTR** (Gehrmann et al., ACL 2019 demo) | 1906.04043 (check) | Per-token "predictability" highlighting, used in our text evidence |
| **DetectGPT** (Mitchell et al., ICML 2023) | 2301.11305 (check) | The original probability-curvature idea that Fast-DetectGPT speeds up |
| **On Calibration of Modern Neural Networks** (Guo et al., ICML 2017) | 1706.04599 (check) | ECE, the calibration metric in our tables |
| **GPT detectors are biased against non-native English writers** (Liang et al., Patterns 2023) | 2304.02819 ✓ | Why we warn on short texts and report text as least reliable |
| **Can AI-generated text be reliably detected?** (Sadasivan et al., 2023) | 2303.11156 (check) | Paraphrase attacks, which explains our attacked-text drop |
| **GPT-2 output detector** (Solaiman et al., "Release strategies…", 2019) | 1908.09203 (check) | Text baseline `roberta-base-openai-detector` |
| **HC3 / ChatGPT detector** (Guo et al., "How close is ChatGPT to human experts?", 2023) | 2301.07597 (check) | Text baseline `chatgpt-detector-roberta` |

## Tier 3: related work (know they exist, one line each)

- **Ivy-Fake** (2506.00979 ✓, ICMR 2026): explainable image + video benchmark, RL (GRPO) reasoning.
- **BusterX++** (2507.14632 ✓): one MLLM for image + video detection and explanation with pure RL.
- **EDVD-LLaMA** (2510.16442 ✓): explainable deepfake video detection with chain-of-thought.
- **TELL** (2605.27921 ✓): explainable text detection. Code is released, but it runs on a 120B model via a paid API, so we could not use it.
- **FakeShield** (ICLR 2025): explainable image forgery detection with an MLLM.
- **GenVidBench** (AAAI 2026) / **GenVideo** (2024): AI-video datasets, 15–45 GB, too big for our budget, which is why video is composition-only.
- **C2PA** (c2pa.org): content-credential standard. We only detect whether a manifest is present; we do not verify it.
- **Error Level Analysis**: N. Krawetz, "A Picture's Worth…", Black Hat 2007 (a talk, not a paper). Used in `src/image/forensics.py`.

---

## Datasets used

| Dataset | Modality | License | Our use |
|---|---|---|---|
| OpenFake (core) | image | CC-BY-NC-4.0 | 3k+3k train, 500+500 test |
| SID-Set | image (3-way + masks) | CC-BY-4.0 | 1.5k per class train, 500 per class test (validation split) |
| Our synthetic edits | image | ours | extra edit-localizer training, secondary test |
| garystafford/deepfake-audio-detection v4 | audio | CC-BY-4.0 | 1,866 clips; split by source recording; **ElevenLabs held out** as unseen |
| RAID | text | MIT | 2k+2k train, 400+400 test, ~30% attacked |

## Models and tools

| Tool | Used for |
|---|---|
| **CLIP ViT-L/14** (`openai/clip-vit-large-patch14`) | Frozen image features, patch tokens, saliency |
| **XLS-R 300M** (`facebook/wav2vec2-xls-r-300m`) | Frozen speech features |
| **Qwen2.5-1.5B** + **Qwen2.5-1.5B-Instruct** | Binoculars and Fast-DetectGPT scores; the instruct model also writes explanations |
| **PyTorch**, **Hugging Face transformers / datasets / huggingface_hub** | Models, data loading |
| **pyarrow** | Fast label-first sampling from huge parquet datasets (`src/common/hf_sample.py`) |
| **scikit-learn** | Logistic-regression heads, AUROC, F1 |
| **Pillow, OpenCV, torchaudio, soundfile** | Image, video and audio I/O and augmentations |
| **Gradio** | Web demo (`app/demo.py`) |
| **Kaggle** (free T4 GPU) | All training and evaluation |
| **Git / GitHub** | Version control (github.com/lakshayg2005/BTP-1) |

## Key terms to be able to explain

- **AUROC:** the probability that a random fake gets a higher "fake" score than a random real item. 0.5 is a coin flip, 1.0 is perfect. It does not depend on a threshold.
- **ECE (expected calibration error):** the gap between stated confidence and actual accuracy. Lower is better.
- **Linear probe / transfer learning:** a frozen pretrained model plus a small trained classifier.
- **Zero-shot detector:** needs no training on the target class (Binoculars, Fast-DetectGPT).
- **OOD / unseen generator:** a test on a generator never seen in training (ElevenLabs in audio). It is the honest generalization number.
- **IoU:** overlap between the predicted edit region and the true mask.
- **Abstention ("uncertain"):** refusing to decide on weak evidence. Trades coverage for accuracy.

## Results so far (Kaggle, 2026-10-08/09)

| Model | Test | Result |
|---|---|---|
| Image AI detector | OpenFake test (22 generators) | AUROC 0.980, acc 91.0%; degraded copies AUROC 0.980 |
| Image AI detector | SID-Set real vs. AI | AUROC 0.992, acc 96.1% |
| Edit localizer | SID-Set tampered vs. clean | image AUROC 0.872, patch AUROC 0.851, IoU 0.28 |
| Edit localizer | OpenFake + our synthetic edits | image AUROC 0.895, patch AUROC 0.875, IoU 0.23 |
| Audio | seen TTS / degraded | AUROC 0.9997 / 0.9995 |
| Audio | **unseen ElevenLabs** | AUROC 0.980, acc 87.2% |
| Text | clean / **attacked** RAID | AUROC 0.975 / 0.757 |

### Baselines (official released models, same test images), evaluation run of 2026-10-11

| Task | Ours | Official baseline |
|---|---|---|
| AI-image detection, SID-Set | **AUROC 0.992** | UnivFD released head (ProGAN-trained): 0.452 |
| AI-image detection, OpenFake | **AUROC 0.980** | UnivFD released head: 0.494 |
| AI-text detection, RAID (clean + attacked) | **AUROC 0.883** | OpenAI RoBERTa detector 0.814; ChatGPT-detector RoBERTa (HC3) 0.634 |

Per-generator AI-image AUROC (OpenFake test): GPT-image-1.5 0.995, Midjourney-7 0.965, Sora-2 0.989, Veo-3 0.982,
Flux.2-klein 0.969, Seedream-5 0.999, Recraft-v3 0.907 (lowest).

### Lessons from the first full evaluation (be ready to discuss)
- **Hand-set 3-way thresholds failed:** real-photo recall was 0.2%, and most real photos were called "edited", because the
  edit signals (especially classical noise/ELA) are high on untouched photos too. Fix: a **learned 3-class fusion**
  trained on a held-out *calibration half* of the test sets, with results reported on the other half; ELA and noise were
  demoted to context-only evidence. (Re-run pending.)
- **LLM explanations:** only 12.5% of Qwen2.5-1.5B drafts passed the faithfulness check; the checker caught the rest
  and the system fell back to the evidence-only explanation. Conclusion: small LLMs hallucinate even when given only
  evidence, so an automatic faithfulness check is necessary. (Prompt now uses short [E1] tags; re-run pending.)
