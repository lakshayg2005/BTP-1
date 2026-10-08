"""Download a small, balanced slice of OpenFake (ComplexDataLab/OpenFake, arXiv 2509.09495), config 'core'.

Uses src/common/hf_sample.py: reads only the label column first, then the images of the rows it needs,
drawing at most a few dozen images per row group so the sample spans many files and generators.
Writes <out>/<split>/{real,ai}/<id>.png plus manifest.csv (generator name per image). All images are re-saved
as PNG so file format cannot leak the label.

  python -m scripts.download_openfake --split train --per-label 3000 --out data/images
"""
import argparse
import csv
import hashlib
import io
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.hf_sample import sample_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--per-label", type=int, default=3000)
    ap.add_argument("--out", default="data/images")
    ap.add_argument("--max-side", type=int, default=1024, help="downscale huge images to save disk")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="core", help="OpenFake subset: 'core' (curated) or 'reddit' (in-the-wild)")
    args = ap.parse_args()

    out = Path(args.out, args.split)
    for d in ("real", "ai"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rows = []
    label_of = lambda x: "real" if str(x).lower() == "real" else "ai"  # noqa: E731  ('real' / 'fake')
    for lab, r in sample_rows("ComplexDataLab/OpenFake", f"{args.config}/{args.split}-*.parquet", "label", label_of,
                              {"real": args.per_label, "ai": args.per_label}, ["image", "label", "model"], seed=args.seed):
        try:
            img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        except Exception:
            continue
        if max(img.size) > args.max_side:
            s = args.max_side / max(img.size)
            img = img.resize((int(img.width * s), int(img.height * s)), Image.BICUBIC)
        name = hashlib.sha1(r["image"]["bytes"][:4096]).hexdigest()[:16]
        path = out / lab / f"{name}.png"
        img.save(path)
        rows.append({"path": str(path).replace(os.sep, "/"), "label": lab, "model": r.get("model", ""), "type": ""})
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "model", "type"])
        w.writeheader()
        w.writerows(rows)
    print({k: sum(r["label"] == k for r in rows) for k in ("real", "ai")}, "generators:", len({r["model"] for r in rows}))


if __name__ == "__main__":
    main()
