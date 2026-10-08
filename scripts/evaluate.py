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
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.explain.template import explain as template_explain  # noqa: E402

CLASSES = ["real", "edited", "ai_generated"]
OUT = Path("results")


def auroc(y, p):
    return round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else None


_EXPERT = {}


def image_expert():
    if "x" not in _EXPERT:
        from src.image.pipeline import ImageExpert
        _EXPERT["x"] = ImageExpert("models/image_head.joblib", tamper_path="models/tamper_head.pt", out_dir="results/maps")
    return _EXPERT["x"]


def eval_image(root, n_explain, baseline=None):
    """3-way evaluation on <root>/{real,ai,edited}; also scores the UnivFD baseline on real vs AI."""
    expert = image_expert()
    root = Path(root)
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
                     "p_edit": v.probabilities["edited"], "model": model_of.get(path, "edited" if truth == "edited" else ""),
                     "p_ai_univfd": baseline.prob_ai(Image.open(path)) if baseline and truth != "edited" else None})

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
    if baseline:
        nr = [r for r in rows if r["truth"] != "edited"]
        res["ai_auroc_real_vs_ai_UnivFD_official_baseline"] = auroc([r["truth"] == "ai_generated" for r in nr], [r["p_ai_univfd"] for r in nr])
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
            stem = Path(r["path"]).stem
            kinds[stem.split("_", 1)[1] if stem[:5].isdigit() and "_" in stem else "sid_tampered"].append(r["pred"] == "edited")
    res["edited_recall_by_kind"] = {k: round(float(np.mean(v)), 4) for k, v in kinds.items()}

    # explanations
    expl = {"n": 0}
    try:
        if not n_explain:
            raise RuntimeError("LLM explanations disabled")
        from src.explain.llm import LLMExplainer
        llm = LLMExplainer()
        random.seed(0)
        # stratified: equal numbers of real / edited / AI cases so the examples cover every verdict type
        sample = []
        for t in CLASSES:
            pool = [r for r in rows if r["truth"] == t]
            sample += random.sample(pool, min(n_explain // len(CLASSES), len(pool)))
        faithful, examples = [], []
        for r in tqdm(sample, desc="explanations"):
            v = verdicts[r["path"]]
            text, rep = llm.explain(v)
            faithful.append(rep["used"] == "llm")
            if sum(e["truth"] == r["truth"] for e in examples) < 2:
                examples.append({"file": Path(r["path"]).name, "truth": r["truth"], "pred": v.label,
                                 "explanation": text, "explainer": rep["used"], "problems": rep.get("problems", [])})
        expl = {"n": len(sample), "llm_faithful_rate": round(float(np.mean(faithful)), 4), "examples": examples}
    except Exception as e:  # explanations must never take the numeric results down with them
        print(f"[warn] LLM explanations skipped: {type(e).__name__}: {e}")
        expl = {"n": 0, "error": f"{type(e).__name__}: {e}"[:300]}
    if not expl.get("examples"):
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


def eval_text_baselines(meta, y):
    from src.baselines import TEXT_BASELINES, TextBaseline
    texts = [m["text"] for m in meta]
    out = {}
    for name, (mid, labels) in TEXT_BASELINES.items():
        try:
            p = TextBaseline(mid, labels).prob_ai(texts)
            ok = ~np.isnan(p)
            out[name] = {"model": mid, "auroc": auroc(y[ok], p[ok]), "acc": round(float(((p[ok] >= 0.5) == y[ok]).mean()), 4)}
        except Exception as e:  # a baseline failing must not kill the evaluation
            out[name] = {"model": mid, "error": str(e)[:200]}
    return out


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
    res["official_baselines"] = eval_text_baselines(meta, y)
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
    from src.baselines import UnivFDBaseline
    try:
        base = UnivFDBaseline(image_expert().detector.encoder)
    except Exception as e:
        print("UnivFD baseline unavailable:", e)
        base = None
    main_root = "data/sid/test" if Path("data/sid/test/edited").exists() else "data/images/test"
    image, expl, _ = eval_image(main_root, args.n_explain, base)
    image["test_set"] = main_root
    report["image"], report["explanations"] = image, expl
    if main_root != "data/images/test" and Path("data/images/test/real").exists():
        report["image_openfake_synthetic_edits"], _, _ = eval_image("data/images/test", 0, base)
        report["image_openfake_synthetic_edits"]["test_set"] = "data/images/test"
    report["audio"], report["text"] = eval_audio(), eval_text()
    for k in ("image_head", "tamper_head", "audio_head", "text_head"):
        for ext in (".metrics.json",):
            f = Path(f"models/{k}{ext}")
            if f.exists():
                report[f"{k}_training_report"] = json.loads(f.read_text())
    (OUT / "results.json").write_text(json.dumps(report, indent=2, default=str))

    md = ["# Results", ""]
    md.append(md_table(image, f"Image: 3-way real / edited / AI (with 'uncertain') on {image['test_set']}"))
    md.append("Recall per class: " + json.dumps(image["recall"]) + "\n")
    md.append("Confusion (truth -> predicted): " + json.dumps(image["confusion (truth -> predicted)"]) + "\n")
    md.append("Edited recall by edit kind: " + json.dumps(image["edited_recall_by_kind"]) + "\n")
    if "image_openfake_synthetic_edits" in report:
        o = report["image_openfake_synthetic_edits"]
        md.append(md_table(o, "Image: OpenFake test + our synthetic edits"))
        md.append("Recall per class: " + json.dumps(o["recall"]) + "\n")
        image = {**image, "ai_auroc_per_generator": o["ai_auroc_per_generator"]}
    md.append("### AI-image AUROC per generator\n\n| generator | n | AUROC |\n|---|---|---|")
    md += [f"| {g} | {v['n']} | {v['auroc']} |" for g, v in image["ai_auroc_per_generator"].items()]
    if "image_head_training_report" in report:
        md.append("\n### Image AI detector: clean vs daily-life degraded\n\n| set | AUROC | acc | ECE |\n|---|---|---|---|")
        md += [f"| {k} | {v.get('auroc')} | {v.get('acc')} | {v.get('ece')} |" for k, v in report["image_head_training_report"].items()]
    for set_name, tv in report.get("tamper_head_training_report", {}).items():
        md.append("\n" + md_table(tv, f"Edit localizer (patch level) - {set_name}"))
    for name in ("audio", "text"):
        for k, v in report[name].items():
            if isinstance(v, dict) and "auroc" in v:
                md.append(md_table(v, f"{name.title()} - {k}"))
        if report[name] and "auroc" in report[name]:
            md.append(md_table(report[name], name.title()))
            if "auroc_per_domain" in report[name]:
                md.append("Per domain: " + json.dumps(report[name]["auroc_per_domain"]) + "\n")
            for bn, bv in report[name].get("official_baselines", {}).items():
                md.append(f"Baseline {bn} ({bv['model']}): AUROC {bv.get('auroc')}, acc {bv.get('acc')} {bv.get('error', '')}\n")
    if "ai_auroc_real_vs_ai_UnivFD_official_baseline" in report["image"]:
        md.append("\n### Image AI detection: ours vs official UnivFD weights (same CLIP backbone)\n\n"
                  "| model | AUROC real vs AI |\n|---|---|\n"
                  f"| UnivFD official head (ProGAN-trained) | {report['image']['ai_auroc_real_vs_ai_UnivFD_official_baseline']} |\n"
                  f"| Ours (OpenFake-trained, daily-life augmented) | {report['image']['ai_auroc_real_vs_ai']} |\n")
    md.append(f"\n### Explanations\n\nLLM explanations passing the faithfulness check: {expl.get('llm_faithful_rate')} (n={expl['n']})\n")
    for e in expl["examples"]:
        md.append(f"**{e['file']}** (truth: {e['truth']})\n\n```\n{e['explanation']}\n```\n")
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
