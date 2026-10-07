"""Full evaluation -> results/summary.md and results/*.json.

1. Image, 3-way (real / edited / AI) on held-out test images, with the "uncertain" option:
   coverage, accuracy on decided cases, per-class recall, confusion matrix; AI-detection AUROC per generator.
2. Audio: AUROC / accuracy on seen generators and on the unseen generator (ElevenLabs).
3. Text: AUROC overall, per domain, clean vs adversarially attacked.
4. Explanations: faithfulness rate of the LLM explainer on a sample of images, plus example explanations.

  python -m scripts.evaluate --n-explain 24
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.explain.template import explain as template_explain  # noqa: E402

CLASSES = ["real", "edited", "ai_generated"]
OUT = Path("results")


def auroc(y, p):
    return round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else None


def eval_image(n_explain):
    from src.image.pipeline import ImageExpert
    expert = ImageExpert("models/image_head.joblib", tamper_path="models/tamper_head.pt", out_dir="results/maps")
    root = Path("data/images/test")
    model_of = {}
    if (root / "manifest.csv").exists():
        model_of = {r["path"]: r["model"] for r in csv.DictReader(open(root / "manifest.csv"))}
    items = [(str(p), "real") for p in sorted((root / "real").glob("*.png"))] + \
            [(str(p), "ai_generated") for p in sorted((root / "ai").glob("*.png"))] + \
            [(str(p), "edited") for p in sorted((root / "edited").glob("*.png"))]
    rows, verdicts = [], {}
    for path, truth in tqdm(items, desc="image eval"):
        v = expert.analyze(path, with_maps=False)
        verdicts[path] = v
        rows.append({"path": path, "truth": truth, "pred": v.label, "p_ai": v.probabilities["ai_generated"],
                     "p_edit": v.probabilities["edited"], "model": model_of.get(path, "edited" if truth == "edited" else "")})

    decided = [r for r in rows if r["pred"] != "uncertain"]
    conf = {t: Counter(r["pred"] for r in rows if r["truth"] == t) for t in CLASSES}
    res = {
        "n": len(rows),
        "coverage": round(len(decided) / len(rows), 4),
        "accuracy_on_decided": round(np.mean([r["pred"] == r["truth"] for r in decided]), 4) if decided else None,
        "accuracy_counting_uncertain_as_wrong": round(np.mean([r["pred"] == r["truth"] for r in rows]), 4),
        "recall": {t: round(conf[t][t] / max(1, sum(conf[t].values())), 4) for t in CLASSES},
        "confusion (truth -> predicted)": {t: dict(conf[t]) for t in CLASSES},
        "ai_auroc_real_vs_ai": auroc([r["truth"] == "ai_generated" for r in rows if r["truth"] != "edited"],
                                     [r["p_ai"] for r in rows if r["truth"] != "edited"]),
    }
    # AI-detection AUROC per generator (each generator's fakes vs all real test images)
    reals = [r["p_ai"] for r in rows if r["truth"] == "real"]
    per_gen = defaultdict(list)
    for r in rows:
        if r["truth"] == "ai_generated":
            per_gen[r["model"] or "unknown"].append(r["p_ai"])
    res["ai_auroc_per_generator"] = {g: {"n": len(v), "auroc": auroc([0] * len(reals) + [1] * len(v), reals + v)}
                                     for g, v in sorted(per_gen.items(), key=lambda kv: -len(kv[1])) if len(v) >= 5}
    # edit kinds
    kinds = defaultdict(list)
    for r in rows:
        if r["truth"] == "edited":
            kinds[Path(r["path"]).stem.split("_", 1)[1]].append(r["pred"] == "edited")
    res["edited_recall_by_kind"] = {k: round(float(np.mean(v)), 4) for k, v in kinds.items()}

    # explanations
    expl = {"n": 0}
    if n_explain:
        from src.explain.llm import LLMExplainer
        llm = LLMExplainer()
        random.seed(0)
        sample = random.sample(rows, min(n_explain, len(rows)))
        faithful, examples = [], []
        for r in tqdm(sample, desc="explanations"):
            v = verdicts[r["path"]]
            text, rep = llm.explain(v)
            faithful.append(rep["used"] == "llm")
            if len(examples) < 6:
                examples.append({"file": Path(r["path"]).name, "truth": r["truth"], "pred": v.label,
                                 "explanation": text, "explainer": rep["used"], "problems": rep.get("problems", [])})
        expl = {"n": len(sample), "llm_faithful_rate": round(float(np.mean(faithful)), 4), "examples": examples}
    else:
        expl["examples"] = [{"file": Path(r["path"]).name, "truth": r["truth"], "explanation": template_explain(verdicts[r["path"]])}
                            for r in rows[:: max(1, len(rows) // 6)][:6]]
    return res, expl, rows


def eval_head(npz, head_path):
    d = np.load(npz)
    p = joblib.load(head_path)["head"].predict_proba(d["X"].astype(np.float32))[:, 1]
    return d, p


def eval_audio():
    res = {}
    for split in ("test", "test_unseen"):
        f = Path(f"features/aud_{split}.npz")
        if f.exists() and Path("models/audio_head.joblib").exists():
            d, p = eval_head(f, "models/audio_head.joblib")
            clean = ~d["aug"]
            res[split] = {"n": int(clean.sum()), "auroc": auroc(d["y"][clean], p[clean]),
                          "acc": round(float(((p[clean] >= 0.5) == d["y"][clean]).mean()), 4),
                          "auroc_degraded": auroc(d["y"][~clean], p[~clean]) if (~clean).any() else None}
    return res


def eval_text():
    f = Path("features/txt_test.npz")
    if not (f.exists() and Path("models/text_head.joblib").exists()):
        return {}
    d, p = eval_head(f, "models/text_head.joblib")
    meta = [json.loads(l) for l in open(f.with_suffix(".jsonl"), encoding="utf-8")]
    y = d["y"]
    res = {"n": int(len(y)), "auroc": auroc(y, p), "acc": round(float(((p >= 0.5) == y).mean()), 4)}
    att = np.array([m["attack"] not in ("none", None) for m in meta])
    res["auroc_clean"] = auroc(y[~att], p[~att])
    res["auroc_attacked"] = auroc(y[att], p[att]) if att.any() else None
    by_dom = defaultdict(list)
    for i, m in enumerate(meta):
        by_dom[m["domain"]].append(i)
    res["auroc_per_domain"] = {k: auroc(y[v], p[v]) for k, v in by_dom.items() if len(v) >= 20}
    return res


def md_table(d: dict, title: str) -> str:
    lines = [f"### {title}", "", "| metric | value |", "|---|---|"]
    for k, v in d.items():
        if not isinstance(v, (dict, list)):
            lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-explain", type=int, default=24)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    (OUT / "maps").mkdir(exist_ok=True)

    report = {}
    image, expl, _ = eval_image(args.n_explain)
    report["image"], report["explanations"] = image, expl
    report["audio"], report["text"] = eval_audio(), eval_text()
    for k in ("image_head", "tamper_head", "audio_head", "text_head"):
        for ext in (".metrics.json",):
            f = Path(f"models/{k}{ext}")
            if f.exists():
                report[f"{k}_training_report"] = json.loads(f.read_text())
    (OUT / "results.json").write_text(json.dumps(report, indent=2, default=str))

    md = ["# Results", ""]
    md.append(md_table(image, "Image: 3-way real / edited / AI (with 'uncertain')"))
    md.append("Recall per class: " + json.dumps(image["recall"]) + "\n")
    md.append("Confusion (truth -> predicted): " + json.dumps(image["confusion (truth -> predicted)"]) + "\n")
    md.append("Edited recall by edit kind: " + json.dumps(image["edited_recall_by_kind"]) + "\n")
    md.append("### AI-image AUROC per generator\n\n| generator | n | AUROC |\n|---|---|---|")
    md += [f"| {g} | {v['n']} | {v['auroc']} |" for g, v in image["ai_auroc_per_generator"].items()]
    if "image_head_training_report" in report:
        md.append("\n### Image AI detector: clean vs daily-life degraded\n\n| set | AUROC | acc | ECE |\n|---|---|---|---|")
        md += [f"| {k} | {v.get('auroc')} | {v.get('acc')} | {v.get('ece')} |" for k, v in report["image_head_training_report"].items()]
    if "tamper_head_training_report" in report:
        md.append("\n" + md_table(report["tamper_head_training_report"], "Edit localizer (patch level)"))
    for name in ("audio", "text"):
        for k, v in report[name].items():
            if isinstance(v, dict) and "auroc" in v:
                md.append(md_table(v, f"{name.title()} - {k}"))
        if report[name] and "auroc" in report[name]:
            md.append(md_table(report[name], name.title()))
            if "auroc_per_domain" in report[name]:
                md.append("Per domain: " + json.dumps(report[name]["auroc_per_domain"]) + "\n")
    md.append(f"\n### Explanations\n\nLLM explanations passing the faithfulness check: {expl.get('llm_faithful_rate')} (n={expl['n']})\n")
    for e in expl["examples"]:
        md.append(f"**{e['file']}** (truth: {e['truth']})\n\n```\n{e['explanation']}\n```\n")
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
