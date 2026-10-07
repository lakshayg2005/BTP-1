"""Stream a balanced slice of SID-Set (saberzl/SID_Set; SIDA, Huang et al., CVPR 2025; CC-BY-4.0).

SID-Set is the standard benchmark for exactly our 3-way task: real / fully synthetic / tampered (locally
edited) social-media images, with binary masks of the edited regions. The official test split is restricted,
so we use its public **validation** split as our test set and the train split for training.

Writes data/sid/<split>/{real,ai,edited}/<id>.png and data/sid/<split>/edited_masks/<id>.npy (packbits).
  python -m scripts.download_sidset --split train --per-class 1500
  python -m scripts.download_sidset --split validation --per-class 500 --out-split test
"""
import argparse
import os

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

NAMES = {0: "real", 1: "ai", 2: "edited"}


def to_mask(m, size) -> np.ndarray | None:
    if m is None:
        return None
    if not isinstance(m, Image.Image):
        m = Image.fromarray(np.asarray(m))
    m = m.convert("L").resize(size, Image.NEAREST)
    return (np.asarray(m) > 127).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-split", default=None)
    ap.add_argument("--per-class", type=int, default=1500)
    ap.add_argument("--out", default="data/sid")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = f"{args.out}/{args.out_split or args.split}"
    for d in ("real", "ai", "edited", "edited_masks"):
        os.makedirs(f"{out}/{d}", exist_ok=True)

    ds = load_dataset("saberzl/SID_Set", split=args.split, streaming=True).shuffle(seed=args.seed, buffer_size=1000)
    counts = {0: 0, 1: 0, 2: 0}
    bar = tqdm(total=3 * args.per_class)
    for ex in ds:
        y = int(ex["label"])
        if counts.get(y, args.per_class) >= args.per_class:
            if all(c >= args.per_class for c in counts.values()):
                break
            continue
        try:
            img = ex["image"].convert("RGB")
        except Exception:
            continue
        mask = to_mask(ex.get("mask"), img.size) if y == 2 else None
        if y == 2 and (mask is None or mask.sum() == 0):
            continue                                    # tampered sample without a usable mask
        if max(img.size) > args.max_side:
            s = args.max_side / max(img.size)
            img = img.resize((int(img.width * s), int(img.height * s)), Image.BICUBIC)
            if mask is not None:
                mask = np.asarray(Image.fromarray(mask * 255).resize(img.size, Image.NEAREST)) > 127
        name = str(ex.get("img_id") or f"{y}_{counts[y]}").replace("/", "_")[:40]
        img.save(f"{out}/{NAMES[y]}/{name}.png")
        if mask is not None:
            np.save(f"{out}/edited_masks/{name}.npy", np.packbits(np.asarray(mask, np.uint8)))
        counts[y] += 1
        bar.update(1)
    print({NAMES[k]: v for k, v in counts.items()})


if __name__ == "__main__":
    main()
