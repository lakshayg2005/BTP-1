"""Stream a balanced slice of RAID (liamdugan/raid, MIT, Dugan et al. ACL 2024) and compute likelihood features.

Human texts have model == "human". We keep mostly un-attacked texts plus a share of adversarial ones
(paraphrase, synonym, ...) so the head sees realistic edits. Texts are saved alongside features.

  python -m scripts.extract_text_features --per-label 2000 --out features/txt_train.npz
  python -m scripts.extract_text_features --per-label 400 --seed 1 --skip 20000 --out features/txt_test.npz
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.text.detector import OBSERVER, LikelihoodScorer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label", type=int, default=2000)
    ap.add_argument("--attacked-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0, help="skip rows so train and test slices differ")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = load_dataset("liamdugan/raid", split="train", streaming=True).skip(args.skip).shuffle(seed=args.seed, buffer_size=10000)
    scorer = LikelihoodScorer()
    want = {0: args.per_label, 1: args.per_label}
    want_attacked = int(args.per_label * args.attacked_frac)
    got = {0: 0, 1: 0}
    got_attacked = {0: 0, 1: 0}
    X, Y, meta = [], [], []
    bar = tqdm(total=2 * args.per_label)
    for ex in ds:
        y = 0 if ex["model"] == "human" else 1
        attacked = ex.get("attack", "none") not in ("none", None)
        if got[y] >= want[y] or (attacked and got_attacked[y] >= want_attacked) or ex.get("domain") == "code":
            if all(got[k] >= want[k] for k in got):
                break
            continue
        text = ex["generation"]
        if len(text.split()) < 30:
            continue
        r = scorer.score(text)
        if r["features"] is None or not np.isfinite(r["features"]).all():
            continue
        X.append(r["features"]); Y.append(y)
        meta.append({"model": ex["model"], "domain": ex["domain"], "attack": ex.get("attack"), "text": text[:2000]})
        got[y] += 1; got_attacked[y] += attacked
        bar.update(1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, X=np.array(X, np.float32), y=np.array(Y), aug=np.array([m["attack"] not in ("none", None) for m in meta]),
                        paths=np.array([f"raid_{i}" for i in range(len(Y))]), encoder=np.array(OBSERVER))
    with open(Path(args.out).with_suffix(".jsonl"), "w", encoding="utf-8") as f:
        for m, y in zip(meta, Y):
            f.write(json.dumps({**m, "label": y}) + "\n")
    print(got, got_attacked)


if __name__ == "__main__":
    main()
