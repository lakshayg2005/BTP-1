"""Web demo: upload an image, video, audio clip or paste text -> verdict, confidence, heatmaps, explanation.

  python -m app.demo                 (local)
  python -m app.demo --share         (public link, e.g. from Kaggle)
  python -m app.demo --explainer template   (no LLM; runs on CPU-only laptops)
"""
import argparse
import os
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze import modality  # noqa: E402
from src.explain.template import LABEL_TEXT, explain as template_explain  # noqa: E402

MODELS = os.environ.get("BTP_MODELS", "models")
ARGS = None


@lru_cache(None)
def image_expert():
    from src.image.pipeline import ImageExpert
    return ImageExpert(f"{MODELS}/image_head.joblib", out_dir="outputs", tamper_path=f"{MODELS}/tamper_head.pt")


@lru_cache(None)
def audio_expert():
    from src.audio.pipeline import AudioExpert
    p = f"{MODELS}/audio_head.joblib"
    return AudioExpert(p) if Path(p).exists() else None


@lru_cache(None)
def text_expert():
    from src.text.pipeline import TextExpert
    return TextExpert(f"{MODELS}/text_head.joblib")


@lru_cache(None)
def llm():
    from src.explain.llm import LLMExplainer
    return LLMExplainer()


def overlay(img_path: str, map_path: str) -> Image.Image:
    import matplotlib.cm as cm
    base = Image.open(img_path).convert("RGB")
    m = np.asarray(Image.open(map_path).convert("L").resize(base.size), np.float32) / 255
    heat = Image.fromarray((cm.jet(m)[..., :3] * 255).astype(np.uint8))
    return Image.blend(base, heat, 0.45)


def run(file, text):
    if file is None and not (text and text.strip()):
        return "Upload a file or paste text.", "", None, []
    kind = "text" if file is None else modality(file)
    if kind == "image":
        v = image_expert().analyze(file)
    elif kind == "video":
        from src.video.pipeline import VideoExpert
        v = VideoExpert(image_expert(), audio_expert()).analyze(file)
    elif kind == "audio":
        if audio_expert() is None:
            return "Audio model not trained yet.", "", None, []
        v = audio_expert().analyze(file)
    elif kind == "text":
        v = text_expert().analyze(text if file is None else Path(file).read_text(encoding="utf-8"))
    else:
        return f"Unsupported file type: {Path(file).suffix}", "", None, []

    if ARGS.explainer == "llm":
        v.explanation, rep = llm().explain(v)
        if rep["used"] != "llm":
            v.explanation += "\n\n(LLM draft failed the faithfulness check; evidence-only explanation shown.)"
    else:
        v.explanation = template_explain(v)

    gallery = []
    for name in ("ai_saliency", "tamper_map", "ela_map"):
        if name in v.artifacts and kind == "image":
            gallery.append((overlay(file, v.artifacts[name]), name.replace("_", " ")))
    probs = {k.replace("_", " "): float(p) for k, p in v.probabilities.items()}
    head = f"## {LABEL_TEXT[v.label]}\nConfidence: **{v.confidence:.0%}** - modality: {kind}"
    evidence = "\n".join(f"- `{e.id}` ({e.supports}, strength {e.strength:.2f}): {e.description}" for e in v.evidence)
    return head, v.explanation + "\n\n### Evidence\n" + evidence, probs, gallery


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--explainer", choices=["llm", "template"], default="llm")
    ARGS = ap.parse_args()

    with gr.Blocks(title="Real, Edited or AI?") as ui:
        gr.Markdown("# Real, Edited or AI-generated?\nUpload an image, video or audio clip, or paste text. "
                    "The system shows its verdict, where it looked, and why. It answers *uncertain* rather than guess.")
        with gr.Row():
            with gr.Column():
                f = gr.File(label="Image / video / audio / .txt", type="filepath")
                t = gr.Textbox(label="...or paste text", lines=6)
                b = gr.Button("Analyze", variant="primary")
            with gr.Column():
                verdict = gr.Markdown()
                probs = gr.Label(label="Probabilities")
        expl = gr.Markdown()
        gal = gr.Gallery(label="Where the system looked", columns=3, height=320)
        b.click(run, [f, t], [verdict, expl, probs, gal])
    ui.launch(share=ARGS.share)


if __name__ == "__main__":
    main()
