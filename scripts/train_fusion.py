"""Train the 3-way fusion (real / edited / AI) on a held-out CALIBRATION half of the test sets.

Each test image is assigned to 'calib' or 'eval' by a hash of its file name (stable across runs). The fusion is
trained on calib only; scripts/evaluate.py reports results on the eval half only, so reported numbers are not
inflated. Scores for every image are cached in results/image_features.csv for analysis.

Also picks:
  - tau: the confidence below which the system answers "uncertain" (smallest tau keeping >= 90% accuracy on
    decided calib cases, capped at 0.75; falls back to 0.5);
  - tamper_flag: the edit-localizer score above which evidence is described as "altered region found"
    (Youden-optimal threshold for edited vs untouched on calib).

  python -m scripts.train_fusion
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.image.pipeline import FEATURES, ImageExpert  # noqa: E402

ROOTS = ["data/sid/test", "data/images/test"]
DIRS = {"real": "real", "ai": "ai_generated", "edited": "edited"}


def split_of(path: str) -> str:
    return "calib" if int(hashlib.md5(Path(path).name.encode()).hexdigest(), 16) % 2 == 0 else "eval"


def main():
    expert = ImageExpert("models/image_head.joblib", tamper_path="models/tamper_head.pt", fusion_path=None)
    rows = []
    for root in ROOTS:
        for d, truth in DIRS.items():
            files = sorted(Path(root, d).glob("*.png"))
            print(f"{root}/{d}: {len(files)} images", flush=True)
            for n, p in enumerate(files):
                if n % 250 == 0:
                    print(f"  {n}/{len(files)}", flush=True)
                f, _ = expert.features(Image.open(p).convert("RGB"), str(p))
                rows.append({"path": str(p).replace("\\", "/"), "set": root, "truth": truth, "split": split_of(str(p)), **f})
    Path("results").mkdir(exist_ok=True)
    with open("results/image_features.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cal = [r for r in rows if r["split"] == "calib"]
    X = np.array([[r[k] for k in FEATURES] for r in cal], np.float32)
    y = np.array([r["truth"] for r in cal])
    head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0))
    head.fit(X, y)
    P = head.predict_proba(X)
    classes = list(head.classes_)
    conf, pred = P.max(1), np.array(classes)[P.argmax(1)]

    tau = 0.5
    for t in np.arange(0.40, 0.76, 0.01):
        m = conf >= t
        if m.mean() > 0 and (pred[m] == y[m]).mean() >= 0.90:
            tau = float(round(t, 2))
            break
    else:
        tau = 0.5
    is_edit = np.array([r["truth"] == "edited" for r in cal])
    fpr, tpr, thr = roc_curve(is_edit, [r["tamper_top8"] for r in cal])
    tamper_flag = float(thr[np.argmax(tpr - fpr)])

    joblib.dump({"head": head, "classes": classes, "features": FEATURES, "tau": tau, "tamper_flag": tamper_flag},
                "models/fusion_head.joblib")
    m = conf >= tau
    report = {"calib_n": int(len(cal)), "eval_n": int(sum(r["split"] == "eval" for r in rows)), "tau": tau,
              "tamper_flag": round(tamper_flag, 3), "calib_accuracy_all": round(float((pred == y).mean()), 4),
              "calib_coverage_at_tau": round(float(m.mean()), 4),
              "calib_accuracy_on_decided": round(float((pred[m] == y[m]).mean()), 4) if m.any() else None,
              "coefficients": {c: dict(zip(FEATURES, np.round(head[-1].coef_[i], 3).tolist())) for i, c in enumerate(classes)}}
    Path("models/fusion_head.metrics.json").write_text(json.dumps(report, indent=2))
    with open("results/eval_split.json", "w") as fh:
        json.dump(sorted(r["path"] for r in rows if r["split"] == "eval"), fh)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
