# Explainable Detection of AI-Generated, Edited and Real Content

BTP project. Detects whether an **image, video, audio clip or text** is **real**, **edited** or **AI-generated**,
shows **where** (heatmaps, edit masks, time segments, sentences) and explains **why**, using only measured evidence.
It answers **uncertain** instead of guessing when evidence is weak.

Base paper: Omni-Fake (arXiv 2605.01638, 2026). References: [RESEARCH_NOTES.md](RESEARCH_NOTES.md). Plan: [PROJECT_PLAN.md](PROJECT_PLAN.md).

## How it works

```
input -> modality expert(s) -> Evidence list -> fusion -> verdict (+ "uncertain") -> grounded explainer -> faithfulness check
```

| Expert | Method | Trained on |
|---|---|---|
| Image: AI-generated | Frozen CLIP ViT-L/14 + linear head (UnivFD idea), WhatsApp/screenshot/JPEG augmentations, saliency map | OpenFake (75+ generators) + SID-Set real/synthetic |
| Image: edited | Patch-level linear probe on CLIP tokens -> edit map; plus ELA, noise residuals, EXIF/C2PA/generator tags | SID-Set tampered images with masks (CVPR 2025) + our synthetic splices / copy-moves / AI patches |
| Audio | Frozen XLS-R 300M + linear head; 2-s window localization; phone/Opus/MP3/noise augmentations | 6 commercial/open TTS (ElevenLabs held out as unseen) |
| Text | Binoculars + Fast-DetectGPT + GLTR statistics from a Qwen2.5 pair, calibrated head; per-sentence mixed-authorship check | RAID (incl. adversarial attacks) |
| Video | Frame-level image expert + temporal jumps + soundtrack through the audio expert (catches voice-cloned dubs) | (composition, no video training) |
| Explainer | Qwen2.5-1.5B-Instruct writes from evidence only; every sentence must cite evidence and use only measured numbers, else falls back to a template | - |

## Baselines (official released models, same test sets)

- Images: UnivFD authors' released head (`fc_weights.pth`, ProGAN-trained) on the same CLIP ViT-L/14.
- Text: `openai-community/roberta-base-openai-detector`, `Hello-SimpleAI/chatgpt-detector-roberta`.

Main 3-way image test: SID-Set validation split (the official test split is restricted).

## Run (Kaggle, free GPU)

Import `kaggle/btp_kaggle.ipynb`, set Accelerator = GPU and Internet = On, *Save & Run All*. It trains every phase,
runs `scripts/evaluate.py` and shows `results/summary.md`. Any phase alone: `PY=python bash scripts/run_phase.sh <deps|image|tamper|audio|text|evaluate>`.

## Use

```bash
python -m scripts.analyze photo.jpg --explainer llm
python -m scripts.analyze voice_note.ogg
python -m scripts.analyze clip.mp4
python -m scripts.analyze --text "paste text here"
python -m app.demo --share          # web demo
python -m tests.smoke_test          # CPU end-to-end check with tiny models
```

## Layout

```
src/common/   evidence + verdict structures      src/image/  AI detector, edit localizer, forensics, splices, fusion
src/audio/    speech detector, augment, fusion   src/text/   likelihood detector, fusion
src/video/    frame + audio composition          src/explain/ template + grounded LLM explainer
scripts/      data download, features, training, evaluation, analyze CLI
app/demo.py   gradio web demo                    kaggle/     notebook generator + notebook
```

Data licenses: OpenFake CC-BY-NC-4.0, SID-Set CC-BY-4.0, RAID MIT, deepfake-audio-detection (garystafford) CC-BY-4.0. Academic, non-commercial use.
