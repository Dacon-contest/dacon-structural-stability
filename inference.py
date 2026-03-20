"""
추론 v2: 멀티 백본 앙상블 + TTA + Temperature Calibration

Usage:
  python inference.py --backbones eva_giant dinov3_huge --tta
  python inference.py --backbones eva02_large --tta --validate
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets import get_val_transforms, get_tta_transforms, load_video_tensor
from models import build_model, get_backbone_config, get_train_preset, get_backbone_choices

import torch.multiprocessing as _mp
_mp.set_sharing_strategy("file_system")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "submissions")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class InferDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sid = self.df.iloc[idx]["id"]
        d = os.path.join(self.data_dir, sid)
        fr = cv2.cvtColor(cv2.imread(os.path.join(d, "front.png")), cv2.COLOR_BGR2RGB)
        tp = cv2.cvtColor(cv2.imread(os.path.join(d, "top.png")), cv2.COLOR_BGR2RGB)
        fr = self.transform(image=fr)["image"]
        tp = self.transform(image=tp)["image"]
        return fr, tp, sid


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs, all_ids = [], []
    for fr, tp, ids in tqdm(loader, desc="Predict", leave=False):
        with autocast("cuda"):
            logits = model(fr.to(device), tp.to(device))
        all_probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
        all_ids.extend(ids)
    return np.concatenate(all_probs), all_ids


@torch.no_grad()
def predict_tta(model, csv_path, data_dir, img_size, device, bs=8):
    model.eval()
    tta_tfs = get_tta_transforms(img_size)
    all_probs, all_ids = [], None
    for tf in tta_tfs:
        ds = InferDataset(csv_path, data_dir, tf)
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
        probs, ids = predict(model, loader, device)
        all_probs.append(probs)
        if all_ids is None:
            all_ids = ids
    return np.mean(all_probs, axis=0), all_ids


def ensemble_predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_csv = os.path.join(DATA_DIR, "open", "sample_submission.csv")
    test_dir = os.path.join(DATA_DIR, "open", "test")

    all_preds = []
    for bk in args.backbones:
        cfg = get_backbone_config(bk)
        img_size = cfg["img_size"]
        preset = get_train_preset(bk)
        bs = preset["batch_size"] * 2  # inference can use larger batch

        for fold in range(args.n_folds):
            path = os.path.join(SAVE_DIR, f"{bk}_fold{fold}.pth")
            if not os.path.exists(path):
                continue
            print(f"  Loading {bk} fold {fold}")
            model = build_model(bk, pretrained=False, num_classes=2, drop_rate=0.0,
                                head_type=args.head_type)
            saved = torch.load(path, map_location=device, weights_only=True)
            model_sd = model.state_dict()
            filtered = {k: v for k, v in saved.items()
                        if k in model_sd and v.shape == model_sd[k].shape}
            model.load_state_dict(filtered, strict=False)
            model = model.to(device)

            if args.tta:
                probs, ids = predict_tta(model, test_csv, test_dir, img_size, device, bs)
            else:
                tf = get_val_transforms(img_size)
                ds = InferDataset(test_csv, test_dir, tf)
                loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
                probs, ids = predict(model, loader, device)

            all_preds.append(probs)
            del model
            torch.cuda.empty_cache()

    if not all_preds:
        print("[ERROR] No predictions!")
        return

    ens = np.mean(all_preds, axis=0)

    # Temperature scaling
    if args.temperature != 1.0:
        logits = np.log(ens + 1e-10) / args.temperature
        e = np.exp(logits)
        ens = e / e.sum(axis=1, keepdims=True)

    # Build submission
    sub = pd.read_csv(test_csv)
    eps = 1e-7
    sub["unstable_prob"] = np.clip(ens[:, 1], eps, 1 - eps)
    sub["stable_prob"] = np.clip(ens[:, 0], eps, 1 - eps)
    row_sum = sub["unstable_prob"] + sub["stable_prob"]
    sub["unstable_prob"] /= row_sum
    sub["stable_prob"] /= row_sum

    name = f"submission_{'_'.join(args.backbones)}"
    if args.tta:
        name += "_tta"
    out = os.path.join(OUTPUT_DIR, f"{name}.csv")
    sub[["id", "unstable_prob", "stable_prob"]].to_csv(out, index=False)

    print(f"\n  Saved: {out}")
    print(f"  Models: {args.backbones} × {len(all_preds)} checkpoints")
    print(f"  TTA: {args.tta} | Temp: {args.temperature}")
    print(f"  unstable_prob: mean={sub['unstable_prob'].mean():.4f} std={sub['unstable_prob'].std():.4f}")


def validate_on_dev(args):
    from sklearn.metrics import log_loss as sk_logloss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_csv = os.path.join(DATA_DIR, "open", "dev.csv")
    dev_dir = os.path.join(DATA_DIR, "open", "dev")
    dev_df = pd.read_csv(dev_csv)

    all_preds = []
    for bk in args.backbones:
        cfg = get_backbone_config(bk)
        img_size = cfg["img_size"]
        for fold in range(args.n_folds):
            path = os.path.join(SAVE_DIR, f"{bk}_fold{fold}.pth")
            if not os.path.exists(path):
                continue
            model = build_model(bk, pretrained=False, num_classes=2, drop_rate=0.0,
                                head_type=args.head_type)
            saved = torch.load(path, map_location=device, weights_only=True)
            model_sd = model.state_dict()
            filtered = {k: v for k, v in saved.items()
                        if k in model_sd and v.shape == model_sd[k].shape}
            model.load_state_dict(filtered, strict=False)
            model = model.to(device)

            if args.tta:
                probs, _ = predict_tta(model, dev_csv, dev_dir, img_size, device)
            else:
                tf = get_val_transforms(img_size)
                ds = InferDataset(dev_csv, dev_dir, tf)
                loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
                probs, _ = predict(model, loader, device)
            all_preds.append(probs)
            del model
            torch.cuda.empty_cache()

    if not all_preds:
        print("[ERROR] No checkpoints!")
        return

    ens = np.mean(all_preds, axis=0)
    true = (dev_df["label"] == "unstable").astype(int).values
    ll = sk_logloss(true, ens, labels=[0, 1])
    acc = (ens.argmax(1) == true).mean()
    print(f"\n  Dev LogLoss: {ll:.6f}")
    print(f"  Dev Accuracy: {acc:.4f}")
    print(f"  Ensembled: {len(all_preds)} checkpoints")


def main():
    p = argparse.ArgumentParser(description="추론 v2")
    p.add_argument("--backbones", nargs="+", default=["dinov2_large"], choices=get_backbone_choices())
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--head_type", type=str, default="simple",
                   choices=["attn_gate", "simple"],
                   help="Head 구조 (학습 시 사용한 것과 동일해야 함)")
    p.add_argument("--validate", action="store_true", help="Dev set 검증")
    args = p.parse_args()

    if args.validate:
        validate_on_dev(args)
    else:
        ensemble_predict(args)


if __name__ == "__main__":
    main()
