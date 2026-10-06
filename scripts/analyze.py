"""Analyze one file end to end and print verdict + explanation (+ JSON with --json).

  python -m scripts.analyze path/to/image.jpg --head models/image_head.joblib
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.explain.template import explain  # noqa: E402
from src.image.pipeline import ImageExpert  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--head", default="models/image_head.joblib")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    v = ImageExpert(args.head, out_dir=args.out_dir).analyze(args.path)
    v.explanation = explain(v)
    print(json.dumps(v.to_dict(), indent=2, default=str) if args.json else v.explanation)
    if v.artifacts:
        print("\nMaps:", ", ".join(f"{k}={p}" for k, p in v.artifacts.items()))


if __name__ == "__main__":
    main()
