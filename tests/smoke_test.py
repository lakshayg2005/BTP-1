"""End-to-end smoke test on CPU with tiny random models and synthetic data (no real datasets, few MB download).

Checks that every stage runs and produces outputs of the right shape; the numbers are meaningless.
  python -m tests.smoke_test
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TINY = {
    "BTP_IMAGE_ENCODER": "hf-internal-testing/tiny-random-CLIPModel",
    "BTP_AUDIO_ENCODER": "hf-internal-testing/tiny-random-wav2vec2",
    "BTP_TEXT_OBSERVER": "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    "BTP_TEXT_PERFORMER": "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
}


def run(*args, cwd):
    print("$", " ".join(args))
    r = subprocess.run([sys.executable, *args], cwd=cwd, env={**os.environ, **TINY}, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-5000:])
        raise SystemExit(f"FAILED: {' '.join(args)}")
    return r.stdout


def make_data(work: Path):
    from PIL import Image
    import soundfile as sf
    rng = np.random.default_rng(0)
    for split, n in (("train", 12), ("test", 6)):
        for lab in ("real", "ai"):
            d = work / "data/images" / split / lab
            d.mkdir(parents=True)
            for i in range(n):
                arr = rng.integers(0, 255, (rng.integers(200, 400), rng.integers(200, 400), 3), dtype=np.uint8)
                if lab == "ai":
                    arr = (arr * 0.3 + 120).astype(np.uint8)
                Image.fromarray(arr).save(d / f"{lab}{i}.png")
        for lab in ("real", "ai"):
            d = work / "data/audio" / split / lab
            d.mkdir(parents=True)
            for i in range(n):
                t = np.linspace(0, 3, 48000)
                w = np.sin(2 * np.pi * (200 + 50 * i) * t) * 0.3 + rng.normal(0, 0.05 if lab == "real" else 0.005, t.size)
                sf.write(d / f"{lab}{i}.flac", w.astype(np.float32), 16000)


def main():
    work = Path(tempfile.mkdtemp(prefix="btp_smoke_"))
    for d in ("src", "scripts", "app"):
        shutil.copytree(ROOT / d, work / d)
    make_data(work)

    run("-m", "scripts.extract_image_features", "--root", "data/images/train", "--out", "features/img_train.npz", "--aug", "1", "--batch", "8", cwd=work)
    run("-m", "scripts.extract_image_features", "--root", "data/images/test", "--out", "features/img_test.npz", "--aug", "1", cwd=work)
    print(run("-m", "scripts.train_head", "--train", "features/img_train.npz", "--test", "features/img_test.npz", "--out", "models/image_head.joblib", cwd=work)[-400:])
    print(run("-m", "scripts.train_tamper", "--epochs", "2", cwd=work)[-600:])
    assert len(list((work / "data/images/test/edited").glob("*.png"))) == 6

    for s in ("train", "test"):
        run("-m", "scripts.extract_audio_features", "--root", f"data/audio/{s}", "--out", f"features/aud_{s}.npz", "--aug", "1", cwd=work)
    run("-m", "scripts.train_head", "--train", "features/aud_train.npz", "--test", "features/aud_test.npz", "--out", "models/audio_head.joblib", cwd=work)

    # text: tiny head on random 4-d features (RAID download skipped)
    import joblib
    from sklearn.linear_model import LogisticRegression
    X = np.random.default_rng(0).normal(size=(40, 4)); y = np.r_[np.zeros(20), np.ones(20)]
    joblib.dump({"head": LogisticRegression().fit(X, y), "encoder": TINY["BTP_TEXT_OBSERVER"],
                 "performer": TINY["BTP_TEXT_PERFORMER"]}, work / "models/text_head.joblib")

    img = next((work / "data/images/test/edited").glob("*.png"))
    print(run("-m", "scripts.analyze", str(img), cwd=work))
    print(run("-m", "scripts.analyze", str(next((work / "data/audio/test/ai").glob("*.flac"))), cwd=work))
    long_text = "The committee reviewed the proposal in detail. " * 12 + "Members agreed to revisit it next month after more data."
    print(run("-m", "scripts.analyze", "--text", long_text, cwd=work))

    # video: write a short clip from test images, with no audio
    import cv2
    vid = work / "clip.mp4"
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 5, (256, 256))
    for p in sorted((work / "data/images/test/real").glob("*.png")) * 3:
        vw.write(cv2.resize(cv2.imread(str(p)), (256, 256)))
    vw.release()
    print(run("-m", "scripts.analyze", str(vid), cwd=work))

    os.makedirs(work / "data/images/test", exist_ok=True)
    print(run("-m", "scripts.evaluate", "--n-explain", "0", cwd=work)[-2500:])
    print(f"\nSMOKE TEST PASSED (workdir {work})")


if __name__ == "__main__":
    main()
