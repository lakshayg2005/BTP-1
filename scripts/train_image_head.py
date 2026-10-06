"""Train the logistic-regression head on cached CLIP features and report test metrics.

  python -m scripts.train_image_head --train features/train.npz --test features/test.npz --out models/image_head.joblib
Runs in seconds on CPU.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.image.ai_detector import DEFAULT_ENCODER  # noqa: E402


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    conf = np.where(p >= 0.5, p, 1 - p)
    correct = (p >= 0.5) == y
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def metrics(y, p):
    return {"n": int(len(y)), "acc": accuracy_score(y, p >= 0.5), "f1": f1_score(y, p >= 0.5),
            "auroc": roc_auc_score(y, p) if len(set(y)) > 1 else None, "ece": ece(y, p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    d = np.load(args.train)
    head = LogisticRegression(C=args.C, max_iter=5000, class_weight="balanced")
    head.fit(d["X"].astype(np.float32), d["y"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump({"head": head, "encoder": DEFAULT_ENCODER}, args.out)

    report = {"train": metrics(d["y"], head.predict_proba(d["X"].astype(np.float32))[:, 1])}
    for t in args.test:
        e = np.load(t)
        p = head.predict_proba(e["X"].astype(np.float32))[:, 1]
        report[Path(t).stem] = metrics(e["y"], p)
        if e["aug"].any():
            report[Path(t).stem + "_clean_only"] = metrics(e["y"][~e["aug"]], p[~e["aug"]])
            report[Path(t).stem + "_degraded_only"] = metrics(e["y"][e["aug"]], p[e["aug"]])
    print(json.dumps(report, indent=2, default=float))
    with open(Path(args.out).with_suffix(".metrics.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)


if __name__ == "__main__":
    main()
