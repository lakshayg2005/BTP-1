"""Train the patch-level edit localizer on synthetic edits and build the fixed edited TEST set.

Train: every real train image -> one random edit (+mask); plus untouched real and AI images (all-zero masks).
Test:  every real test image -> one edit, saved to data/images/test/edited/ with masks in edited_masks/,
       so the 3-way evaluation (real / edited / AI) uses the same files every time.

  python -m scripts.train_tamper --train-root data/images/train --test-root data/images/test --out models/tamper_head.pt
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.image.ai_detector import DEFAULT_ENCODER, ClipEncoder  # noqa: E402
from src.image.augment import random_daily_life  # noqa: E402
from src.image.splice import random_edit  # noqa: E402
from src.image.tamper import TamperHead, grid_info, mask_to_patch_labels, patch_tokens  # noqa: E402


def files(root, label):
    return sorted(str(p) for p in Path(root, label).glob("*.png"))


def build(real, ai, n_clean, geo, save_dir=None, desc=""):
    """Generator of (PIL image, per-patch labels, image label 0=clean/1=edited); streamed to bound RAM."""
    for i, p in enumerate(tqdm(real, desc=f"edits {desc}")):
        img, mask, kind = random_edit(Image.open(p), real, ai)
        if save_dir:
            img.save(f"{save_dir}/edited/{i:05d}_{kind}.png")
            np.save(f"{save_dir}/edited_masks/{i:05d}_{kind}.npy", np.packbits(mask))
        yield img, mask_to_patch_labels(mask, geo[0], geo[1]), 1
    for p in random.sample(real, min(n_clean, len(real))) + random.sample(ai, min(n_clean, len(ai))):
        yield random_daily_life(Image.open(p).convert("RGB")), np.zeros(geo[2] ** 2, np.float32), 0


def encode(enc, samples, bs=32):
    X, Y, I, buf = [], [], [], []

    def flush():
        X.append(patch_tokens(enc, [s[0] for s in buf]).half().cpu())
        Y.append(torch.tensor(np.stack([s[1] for s in buf])))
        I.extend(s[2] for s in buf)
        buf.clear()

    for s in samples:
        buf.append(s)
        if len(buf) == bs:
            flush()
    if buf:
        flush()
    return torch.cat(X), torch.cat(Y), np.array(I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", default="data/images/train")
    ap.add_argument("--test-root", default="data/images/test")
    ap.add_argument("--out", default="models/tamper_head.pt")
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    enc = ClipEncoder(os.environ.get("BTP_IMAGE_ENCODER", DEFAULT_ENCODER))
    geo = grid_info(enc)
    tr_real, tr_ai = files(args.train_root, "real"), files(args.train_root, "ai")
    te_real, te_ai = files(args.test_root, "real"), files(args.test_root, "ai")
    Path(args.test_root, "edited").mkdir(exist_ok=True)
    Path(args.test_root, "edited_masks").mkdir(exist_ok=True)

    Xtr, Ytr, _ = encode(enc, build(tr_real, tr_ai, len(tr_real) // 2, geo, desc="train"))
    Xte, Yte, Ite = encode(enc, build(te_real, te_ai, len(te_real) // 2, geo, save_dir=args.test_root, desc="test"))

    dev = enc.device
    head = TamperHead(enc.model.config.hidden_size).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_w = torch.tensor((Ytr.numel() - Ytr.sum()) / max(1, Ytr.sum()), device=dev)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for ep in range(args.epochs):
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(perm), 64):
            idx = perm[i: i + 64]
            loss = loss_fn(head(Xtr[idx].to(dev).float()), Ytr[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        print(f"epoch {ep + 1}: loss {tot / len(Xtr):.4f}")

    head.eval()
    with torch.no_grad():
        P = torch.cat([torch.sigmoid(head(Xte[i: i + 64].to(dev).float())).cpu() for i in range(0, len(Xte), 64)])
    img_score = P.sort(1).values[:, -min(8, P.shape[1]):].mean(1).numpy()
    pred = (P > 0.5).float()
    edited = Ite == 1
    inter = (pred[edited] * Yte[edited]).sum(1)
    union = ((pred[edited] + Yte[edited]) > 0).float().sum(1).clamp(min=1)
    report = {
        "patch_auroc": float(roc_auc_score(Yte.numpy().ravel(), P.numpy().ravel())),
        "image_auroc_edited_vs_clean": float(roc_auc_score(Ite, img_score)),
        "mean_patch_iou_on_edited": float((inter / union).mean()),
        "n_test_edited": int(edited.sum()), "n_test_clean": int((~edited).sum()),
    }
    print(json.dumps(report, indent=2))
    Path(args.out).parent.mkdir(exist_ok=True)
    torch.save(head.state_dict(), args.out)
    Path(args.out).with_suffix(".metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
