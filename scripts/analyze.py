"""Analyze one input end to end: verdict, confidence, evidence, maps and explanation.

  python -m scripts.analyze photo.jpg
  python -m scripts.analyze voice_note.ogg
  python -m scripts.analyze essay.txt            (or --text "some text")
  add --explainer llm for the grounded LLM explanation, --json for the full record
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.explain.template import explain as template_explain  # noqa: E402

IMAGE = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic"}
AUDIO = {".wav", ".mp3", ".ogg", ".opus", ".m4a", ".flac", ".aac"}
VIDEO = {".mp4", ".mov", ".webm", ".mkv", ".avi"}


def modality(path: str | None) -> str:
    if path is None:
        return "text"
    ext = Path(path).suffix.lower()
    if ext in IMAGE:
        return "image"
    if ext in AUDIO:
        return "audio"
    if ext in VIDEO:
        return "video"
    return "text" if ext in {".txt", ".md"} else "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--text")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--explainer", choices=["template", "llm"], default="template")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    kind = modality(None if args.text else args.path)
    if kind == "image":
        from src.image.pipeline import ImageExpert
        v = ImageExpert(f"{args.models}/image_head.joblib", out_dir=args.out_dir,
                        tamper_path=f"{args.models}/tamper_head.pt").analyze(args.path)
    elif kind == "video":
        from src.audio.pipeline import AudioExpert
        from src.image.pipeline import ImageExpert
        from src.video.pipeline import VideoExpert
        img = ImageExpert(f"{args.models}/image_head.joblib", out_dir=args.out_dir, tamper_path=f"{args.models}/tamper_head.pt")
        aud = AudioExpert(f"{args.models}/audio_head.joblib") if Path(f"{args.models}/audio_head.joblib").exists() else None
        v = VideoExpert(img, aud).analyze(args.path)
    elif kind == "audio":
        from src.audio.pipeline import AudioExpert
        v = AudioExpert(f"{args.models}/audio_head.joblib").analyze(args.path)
    elif kind == "text":
        from src.text.pipeline import TextExpert
        text = args.text or Path(args.path).read_text(encoding="utf-8")
        v = TextExpert(f"{args.models}/text_head.joblib").analyze(text)
    else:
        sys.exit(f"Unsupported file type: {args.path}")

    report = {"used": "template"}
    if args.explainer == "llm":
        from src.explain.llm import LLMExplainer
        v.explanation, report = LLMExplainer().explain(v)
    else:
        v.explanation = template_explain(v)

    if args.json:
        print(json.dumps({**v.to_dict(), "explainer": report}, indent=2, default=str))
    else:
        print(v.explanation)
        if v.artifacts:
            print("\nArtifacts:", ", ".join(f"{k}={p}" for k, p in v.artifacts.items()))
        if report.get("used") == "template_fallback":
            print("\n(LLM draft failed the faithfulness check; showing the evidence-only explanation.)")


if __name__ == "__main__":
    main()
