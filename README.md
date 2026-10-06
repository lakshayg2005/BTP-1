# Explainable Detection of AI-Generated, Edited and Real Content

BTP project. Detects whether an image, video, audio clip or text is **real**, **edited** or **AI-generated**,
says **where** (heatmaps, masks, segments, highlighted words) and **why** (an explanation grounded only in measured evidence).
Returns **uncertain** instead of guessing when evidence is weak.

Base paper: Omni-Fake (arXiv 2605.01638, 2026). Full references: [RESEARCH_NOTES.md](RESEARCH_NOTES.md). Plan: [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Status

| Phase | Modality | Status |
|---|---|---|
| 1 | Image: AI-generated detector (CLIP ViT-L/14 + linear head, daily-life augmentations), forensics, metadata, 3-way verdict, template explainer | In progress |
| 2 | Image: learned edit localizer, LLM explainer | Planned |
| 3 | Audio | Planned |
| 4 | Text | Planned |
| 5 | Video | Planned |
| 6 | Demo + evaluation | Planned |

## Layout

```
src/common/    evidence + verdict data structures, device helper
src/image/     ai_detector (CLIP + head), augment (daily-life degradations), forensics (ELA, noise, metadata), pipeline (fusion)
src/explain/   template explainer (evidence-only)
scripts/       download_openfake, extract_image_features, train_image_head, analyze
```

## Image quickstart

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
export HF_HOME=$PWD/.hf_home            # keep model/dataset caches inside the project
python -m scripts.download_openfake --split train --per-label 3000
python -m scripts.download_openfake --split test  --per-label 500
python -m scripts.extract_image_features --root data/images/train --out features/train.npz --aug 2
python -m scripts.extract_image_features --root data/images/test  --out features/test.npz  --aug 1
python -m scripts.train_image_head --train features/train.npz --test features/test.npz --out models/image_head.joblib
python -m scripts.analyze some_image.jpg --head models/image_head.joblib
```

Training data: OpenFake (ComplexDataLab/OpenFake, CC-BY-NC-4.0, arXiv 2509.09495). Non-commercial academic use.
