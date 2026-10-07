"""Extract CLIP features for a folder dataset laid out as <root>/<label>/**/*.{jpg,png,webp}, label in {real, ai}.

Each image is encoded once clean and `--aug` extra times with random daily-life degradations.
Output: a .npz with X (features), y (1 = ai), paths, aug flag.

Example (GPU machine):
  python -m scripts.extract_image_features --root data/images/train --out features/train.npz --aug 2
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.image.ai_detector import DEFAULT_ENCODER, ClipEncoder  # noqa: E402
from src.image.augment import random_daily_life  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LABELS = {"real": 0, "ai": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--aug", type=int, default=0, help="augmented copies per image")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="max images per label (0 = all)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    items = []
    for name, y in LABELS.items():
        files = sorted(p for p in Path(args.root, name).rglob("*") if p.suffix.lower() in EXTS)
        items += [(str(p), y) for p in (files[: args.limit] if args.limit else files)]
    print(f"{len(items)} images: " + ", ".join(f"{n}={sum(1 for _, y in items if y == v)}" for n, v in LABELS.items()))

    enc = ClipEncoder(device=args.device)
    X, Y, P, A = [], [], [], []
    batch_imgs, batch_meta = [], []

    def flush():
        if batch_imgs:
            X.append(enc.embed(batch_imgs))
            for m in batch_meta:
                Y.append(m[0]); P.append(m[1]); A.append(m[2])
            batch_imgs.clear(); batch_meta.clear()

    for path, y in tqdm(items):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:  # corrupt files happen in scraped datasets
            print(f"skip {path}: {e}")
            continue
        for k in range(args.aug + 1):
            batch_imgs.append(img if k == 0 else random_daily_life(img))
            batch_meta.append((y, path, k > 0))
        if len(batch_imgs) >= args.batch:
            flush()
    flush()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, X=np.concatenate(X).astype(np.float16), y=np.array(Y), paths=np.array(P),
                        aug=np.array(A), encoder=np.array(DEFAULT_ENCODER))
    print(f"saved {args.out}: {sum(len(x) for x in X)} vectors")


if __name__ == "__main__":
    main()
