"""Download a balanced slice of SID-Set (saberzl/SID_Set; SIDA, Huang et al., CVPR 2025; CC-BY-4.0).

SID-Set is the standard benchmark for exactly our 3-way task: real / fully synthetic / tampered (locally
edited) social-media images, with binary masks of the edited regions. The official test split is restricted,
so we use its public **validation** split as our test set and the train split for training.
Sampling reads the label column first (src/common/hf_sample.py), so only needed images are downloaded.

Writes data/sid/<split>/{real,ai,edited}/<id>.png and data/sid/<split>/edited_masks/<id>.npy (packbits).
  python -m scripts.download_sidset --split train --per-class 1500
  python -m scripts.download_sidset --split validation --per-class 500 --out-split test
"""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.hf_sample import sample_rows  # noqa: E402

NAMES = {0: "real", 1: "ai", 2: "edited"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-split", default=None)
    ap.add_argument("--per-class", type=int, default=1500)
    ap.add_argument("--out", default="data/sid")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out, args.out_split or args.split)
    for d in ("real", "ai", "edited", "edited_masks"):
        (out / d).mkdir(parents=True, exist_ok=True)

    counts = {k: 0 for k in NAMES}
    for y, r in sample_rows("saberzl/SID_Set", f"data/{args.split}-*.parquet", "label", int,
                            {k: args.per_class for k in NAMES}, ["img_id", "image", "mask", "label"], seed=args.seed):
        try:
            img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        except Exception:
            continue
        mask = None
        if y == 2:
            if not (r.get("mask") and r["mask"].get("bytes")):
                continue                                         # tampered sample without a usable mask
            m = Image.open(io.BytesIO(r["mask"]["bytes"])).convert("L").resize(img.size, Image.NEAREST)
            mask = (np.asarray(m) > 127).astype(np.uint8)
            if mask.sum() == 0:
                continue
        if max(img.size) > args.max_side:
            s = args.max_side / max(img.size)
            img = img.resize((int(img.width * s), int(img.height * s)), Image.BICUBIC)
            if mask is not None:
                mask = (np.asarray(Image.fromarray(mask * 255).resize(img.size, Image.NEAREST)) > 127).astype(np.uint8)
        name = str(r.get("img_id") or f"{y}_{counts[y]}").replace("/", "_")[:40]
        img.save(out / NAMES[y] / f"{name}.png")
        if mask is not None:
            np.save(out / "edited_masks" / f"{name}.npy", np.packbits(mask, axis=-1))
        counts[y] += 1
    print({NAMES[k]: v for k, v in counts.items()})


if __name__ == "__main__":
    main()
