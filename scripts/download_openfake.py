"""Stream a small, balanced slice of OpenFake (ComplexDataLab/OpenFake, arXiv 2509.09495) to disk.

Writes <out>/<split>/{real,ai}/<sha>.png plus a manifest.csv with the generator name per image.
All images are re-saved as PNG so file format cannot leak the label (real photos are often JPEG,
generated ones PNG). Set HF_HOME to a folder inside the project to keep caches out of the home dir.

  python -m scripts.download_openfake --split train --per-label 3000 --out data/images
"""
import argparse
import csv
import os

from datasets import load_dataset
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--per-label", type=int, default=3000)
    ap.add_argument("--out", default="data/images")
    ap.add_argument("--max-side", type=int, default=1024, help="downscale huge images to save disk")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="core", help="OpenFake subset: 'core' (curated) or 'reddit' (in-the-wild)")
    args = ap.parse_args()

    ds = load_dataset("ComplexDataLab/OpenFake", args.config, split=args.split, streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=1000)
    feat = (ds.features or {}).get("label") if getattr(ds, "features", None) else None
    names = getattr(feat, "names", None)
    print("label feature:", feat)
    counts = {"real": 0, "ai": 0}
    rows = []
    for d in ("real", "ai"):
        os.makedirs(f"{args.out}/{args.split}/{d}", exist_ok=True)
    bar = tqdm(total=2 * args.per_label)
    for ex in ds:
        raw = ex["label"]
        if not isinstance(raw, str):                      # ClassLabel int -> name
            raw = names[raw] if names else str(raw)
        lab = "real" if str(raw).lower() == "real" else "ai"
        if counts[lab] >= args.per_label:
            if all(c >= args.per_label for c in counts.values()):
                break
            continue
        try:
            img = ex["image"].convert("RGB")
        except Exception:
            continue
        if max(img.size) > args.max_side:
            s = args.max_side / max(img.size)
            img = img.resize((int(img.width * s), int(img.height * s)))
        name = (ex.get("sha256") or ex.get("hash") or f"{lab}_{counts[lab]}")[:16]
        path = f"{args.out}/{args.split}/{lab}/{name}.png"
        img.save(path)
        rows.append({"path": path, "label": lab, "model": ex.get("model", ""), "type": ex.get("type", "")})
        counts[lab] += 1
        bar.update(1)
    with open(f"{args.out}/{args.split}/manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "model", "type"])
        w.writeheader()
        w.writerows(rows)
    print(counts)


if __name__ == "__main__":
    main()
