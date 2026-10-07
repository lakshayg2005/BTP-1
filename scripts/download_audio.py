"""Download garystafford/deepfake-audio-detection (v4, CC-BY-4.0, 1,866 clips) and split it safely.

Real clips come from 14 YouTube recordings; fake clips from 6 TTS platforms (file prefix el_, hg_, hu_, lv_, po_, sp_).
To avoid leakage and test generalization:
  - real clips are split by SOURCE recording (no recording appears in both train and test);
  - ElevenLabs (el_) is held out entirely as an UNSEEN generator in test_unseen;
  - the other platforms are split 80/20 at random.
Writes data/audio/{train,test,test_unseen}/{real,ai}/*.flac and data/audio/manifest.csv.
"""
import argparse
import csv
import io
import random
import re
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

PLATFORM = {"el": "ElevenLabs", "hg": "Kokoro", "hu": "Hume AI", "lv": "Luvvoice", "po": "Amazon Polly", "sp": "Speechify", "yt": "real"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/audio")
    ap.add_argument("--holdout", default="el")
    args = ap.parse_args()
    random.seed(0)

    ds = load_dataset("garystafford/deepfake-audio-detection", split="train").cast_column("audio", Audio(decode=False))
    rows = []
    for ex in ds:
        name = Path(ex["audio"]["path"] or "").name
        prefix = name.split("_")[0] if "_" in name else ("yt" if ex["label"] == 0 else "unk")
        rows.append((ex, name, prefix))

    if sum(bool(n) for _, n, _ in rows) < 0.9 * len(rows):
        # file names missing in this copy of the dataset: fall back to a plain random split, no holdout
        print("WARNING: no file names -> random 80/20 split, no speaker grouping or unseen-generator test")
        rows = [(ex, n or f"clip_{i}", "rand") for i, (ex, n, _) in enumerate(rows)]
        args.holdout = None

    def source(n):
        """Source recording id of a real clip (filename up to _part / _c_part / _p2_); each clip alone in fallback."""
        return re.split(r"_(?:c_)?part|_p2_", n)[0] if args.holdout else n
    real_sources = sorted({source(n) for ex, n, p in rows if ex["label"] == 0})
    random.shuffle(real_sources)
    test_sources = set(real_sources[: max(2, len(real_sources) // 5)])

    out = Path(args.out)
    manifest = []
    for i, (ex, name, prefix) in enumerate(rows):
        lab = "real" if ex["label"] == 0 else "ai"
        if lab == "real":
            split = "test" if source(name) in test_sources else "train"
        elif prefix == args.holdout:
            split = "test_unseen"
        else:
            split = "test" if random.random() < 0.2 else "train"
        d = out / split / lab
        d.mkdir(parents=True, exist_ok=True)
        path = d / (name or f"{lab}_{i}.flac")
        wav, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]))
        sf.write(path, wav, sr)
        manifest.append({"path": str(path), "split": split, "label": lab, "platform": PLATFORM.get(prefix, prefix)})
    # unseen-generator test also needs real clips: reuse the held-out real recordings
    for m in [m for m in manifest if m["split"] == "test" and m["label"] == "real"]:
        d = out / "test_unseen" / "real"
        d.mkdir(parents=True, exist_ok=True)
        dst = d / Path(m["path"]).name
        dst.write_bytes(Path(m["path"]).read_bytes())
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "split", "label", "platform"])
        w.writeheader()
        w.writerows(manifest)
    for s in ("train", "test", "test_unseen"):
        print(s, {l: len(list((out / s / l).glob("*"))) for l in ("real", "ai") if (out / s / l).exists()})
    print("real test recordings:", sorted(test_sources))


if __name__ == "__main__":
    main()
