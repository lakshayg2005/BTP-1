"""Extract speech-encoder features for <root>/{real,ai}/**/*.{wav,flac,mp3,ogg,m4a}.

Clips are cut to --seconds (default 6 s) to bound memory; each clip is encoded clean plus --aug degraded copies.

  python -m scripts.extract_audio_features --root data/audio/train --out features/aud_train.npz --aug 1
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio.augment import random_daily_life  # noqa: E402
from src.audio.detector import DEFAULT_AUDIO_ENCODER, SR, SpeechEncoder, load_audio  # noqa: E402

EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}
LABELS = {"real": 0, "ai": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--aug", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--encoder", default=DEFAULT_AUDIO_ENCODER)
    ap.add_argument("--layer", type=int, default=12)
    args = ap.parse_args()

    items = []
    for name, y in LABELS.items():
        files = sorted(p for p in Path(args.root, name).rglob("*") if p.suffix.lower() in EXTS)
        items += [(str(p), y) for p in (files[: args.limit] if args.limit else files)]
    print(f"{len(items)} clips")

    enc = SpeechEncoder(args.encoder, args.layer)
    X, Y, P, A, buf = [], [], [], [], []

    def flush():
        if buf:
            X.append(enc.embed([b[0] for b in buf]))
            for _, y, p, a in buf:
                Y.append(y); P.append(p); A.append(a)
            buf.clear()

    for n, (path, y) in enumerate(tqdm(items)):
        if n % 300 == 0:
            print(f"  {n}/{len(items)} clips", flush=True)
        try:
            wav = load_audio(path, args.seconds)
        except Exception as e:
            print(f"skip {path}: {e}")
            continue
        if len(wav) < SR // 2:
            continue
        for k in range(args.aug + 1):
            buf.append((wav if k == 0 else random_daily_life(wav), y, path, k > 0))
        if len(buf) >= args.batch:
            flush()
    flush()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, X=np.concatenate(X).astype(np.float16), y=np.array(Y), paths=np.array(P),
                        aug=np.array(A), encoder=np.array(args.encoder), layer=np.array(args.layer))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
