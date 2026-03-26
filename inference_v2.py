"""
inference_v2.py — Dual-Crop + Fold Selection

변경사항 (vs inference.py):
  1. Front/Top 별도 CenterCrop 비율: front=0.9 (기본), top=0.7 (기본)
     → Top view는 블록이 중앙에 작게 있으므로 더 aggressive crop
  2. --folds 파라미터: 특정 fold만 선택 가능 (예: --folds 0 2)
     → 5-fold 앙상블 대신 best fold 1개만 사용 가능
  3. TTA도 front/top 별도 crop 적용

Usage:
  # Best fold 0만 사용 + dual crop
  python inference_v2.py --backbones eva_giant --tta --folds 0

  # Fold 0,2만 앙상블
  python inference_v2.py --backbones eva_giant --tta --folds 0 2

  # 전체 fold (기존과 동일)
  python inference_v2.py --backbones eva_giant --tta

  # 특정 체크포인트 파일 직접 지정 (vfa 등)
  python inference_v2.py --backbones eva_giant --tta --checkpoint eva_giant_vfa_fold1.pth

  # 여러 체크포인트 직접 지정
  python inference_v2.py --backbones eva_giant --tta --checkpoint eva_giant_vfa_fold1.pth eva_giant_fold0.pth

  # Top crop 비율 변경
  python inference_v2.py --backbones eva_giant --tta --folds 0 --top_crop 0.6
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

from datasets import load_video_tensor
from models import build_model, get_backbone_config, get_train_preset, get_backbone_choices

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.multiprocessing as _mp
_mp.set_sharing_strategy("file_system")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "submissions")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ========== Dual-Crop Transform 생성 ==========
def make_dual_transforms(img_size, front_crop_ratio=0.9, top_crop_ratio=0.7):
    """Front/Top에 각각 다른 CenterCrop 비율 적용"""
    norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    fc = int(img_size * front_crop_ratio)
    tc = int(img_size * top_crop_ratio)
    front_tf = A.Compose([
        A.CenterCrop(fc, fc), A.Resize(img_size, img_size),
        A.Normalize(**norm), ToTensorV2(),
    ])
    top_tf = A.Compose([
        A.CenterCrop(tc, tc), A.Resize(img_size, img_size),
        A.Normalize(**norm), ToTensorV2(),
    ])
    return front_tf, top_tf


def make_dual_tta_transforms(img_size, front_crop_ratio=0.9, top_crop_ratio=0.7):
    """TTA 변환 — front/top 각각 별도 crop"""
    norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    fc = int(img_size * front_crop_ratio)
    tc = int(img_size * top_crop_ratio)

    # tighter crop: front 81%, top = top_crop * 0.9
    fc_tight = int(img_size * front_crop_ratio * 0.9)
    tc_tight = int(img_size * top_crop_ratio * 0.9)

    front_ttas = [
        # 0: base
        A.Compose([A.CenterCrop(fc, fc), A.Resize(img_size, img_size),
                    A.Normalize(**norm), ToTensorV2()]),
        # 1: hflip
        A.Compose([A.CenterCrop(fc, fc), A.Resize(img_size, img_size),
                    A.HorizontalFlip(p=1.0), A.Normalize(**norm), ToTensorV2()]),
        # 2: brightness
        A.Compose([A.CenterCrop(fc, fc), A.Resize(img_size, img_size),
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
                    A.Normalize(**norm), ToTensorV2()]),
        # 3: tighter crop
        A.Compose([A.CenterCrop(fc_tight, fc_tight), A.Resize(img_size, img_size),
                    A.Normalize(**norm), ToTensorV2()]),
    ]
    top_ttas = [
        A.Compose([A.CenterCrop(tc, tc), A.Resize(img_size, img_size),
                    A.Normalize(**norm), ToTensorV2()]),
        A.Compose([A.CenterCrop(tc, tc), A.Resize(img_size, img_size),
                    A.HorizontalFlip(p=1.0), A.Normalize(**norm), ToTensorV2()]),
        A.Compose([A.CenterCrop(tc, tc), A.Resize(img_size, img_size),
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
                    A.Normalize(**norm), ToTensorV2()]),
        A.Compose([A.CenterCrop(tc_tight, tc_tight), A.Resize(img_size, img_size),
                    A.Normalize(**norm), ToTensorV2()]),
    ]
    return front_ttas, top_ttas


class DualCropDataset(Dataset):
    """Front/Top에 서로 다른 transform 적용"""
    def __init__(self, csv_path, data_dir, front_tf, top_tf):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.front_tf = front_tf
        self.top_tf = top_tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sid = self.df.iloc[idx]["id"]
        d = os.path.join(self.data_dir, sid)
        fr = cv2.cvtColor(cv2.imread(os.path.join(d, "front.png")), cv2.COLOR_BGR2RGB)
        tp = cv2.cvtColor(cv2.imread(os.path.join(d, "top.png")), cv2.COLOR_BGR2RGB)
        fr = self.front_tf(image=fr)["image"]
        tp = self.top_tf(image=tp)["image"]
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
def predict_tta(model, csv_path, data_dir, img_size, device,
                front_crop, top_crop, bs=8, nw=0):
    model.eval()
    front_ttas, top_ttas = make_dual_tta_transforms(img_size, front_crop, top_crop)
    all_probs, all_ids = [], None
    for ftf, ttf in zip(front_ttas, top_ttas):
        ds = DualCropDataset(csv_path, data_dir, ftf, ttf)
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
        probs, ids = predict(model, loader, device)
        all_probs.append(probs)
        if all_ids is None:
            all_ids = ids
    return np.mean(all_probs, axis=0), all_ids


def _find_checkpoint_paths(backbone, folds, seeds):
    """특정 fold 목록에 대해서만 checkpoint 탐색"""
    paths = []
    for seed in seeds:
        sfx = '' if seed == 42 else f'_s{seed}'
        for fold in folds:
            p = os.path.join(SAVE_DIR, f"{backbone}{sfx}_fold{fold}.pth")
            if os.path.exists(p):
                paths.append((p, backbone, fold, seed))
    return paths


def _resolve_checkpoints(args):
    """--checkpoint 직접 지정 또는 --folds 기반 자동 탐색"""
    results = []  # list of (path, backbone, label)
    if args.checkpoint:
        for ckpt_name in args.checkpoint:
            p = os.path.join(SAVE_DIR, ckpt_name)
            if os.path.exists(p):
                results.append((p, args.backbones[0], ckpt_name))
            else:
                print(f"  [WARN] checkpoint not found: {p}")
    else:
        for bk in args.backbones:
            for path, _, fold, seed in _find_checkpoint_paths(bk, args.folds, args.seeds):
                sfx = '' if seed == 42 else f' s{seed}'
                results.append((path, bk, f"{bk} fold{fold}{sfx}"))
    return results


def ensemble_predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_csv = os.path.join(DATA_DIR, "open", "sample_submission.csv")
    test_dir = os.path.join(DATA_DIR, "open", "test")

    print(f"  Front crop: {args.front_crop} | Top crop: {args.top_crop}")
    if args.checkpoint:
        print(f"  Checkpoints: {args.checkpoint}")
    else:
        print(f"  Folds: {args.folds}")

    ckpts = _resolve_checkpoints(args)
    if not ckpts:
        print("[ERROR] No checkpoints found!")
        return

    all_preds = []
    for path, bk, label in ckpts:
        cfg = get_backbone_config(bk)
        img_size = cfg["img_size"]
        preset = get_train_preset(bk)
        bs = preset["batch_size"] * 2

        print(f"  Loading {label}")
        model = build_model(bk, pretrained=False, num_classes=2, drop_rate=0.0,
                            head_type=args.head_type)
        saved = torch.load(path, map_location=device, weights_only=True)
        model_sd = model.state_dict()
        filtered = {k: v for k, v in saved.items()
                    if k in model_sd and v.shape == model_sd[k].shape}
        model.load_state_dict(filtered, strict=False)
        model = model.to(device)

        nw = getattr(args, 'num_workers', 4)
        if args.tta:
            probs, ids = predict_tta(model, test_csv, test_dir, img_size, device,
                                     args.front_crop, args.top_crop, bs, nw=nw)
        else:
            ftf, ttf = make_dual_transforms(img_size, args.front_crop, args.top_crop)
            ds = DualCropDataset(test_csv, test_dir, ftf, ttf)
            loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
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

    # 파일명에 fold/checkpoint 정보 포함
    name = f"submission_{'_'.join(args.backbones)}"
    if args.checkpoint:
        # 체크포인트 파일명에서 핵심 정보 추출
        ckpt_tag = '_'.join(os.path.splitext(c)[0] for c in args.checkpoint)
        name = f"submission_{ckpt_tag}"
    else:
        if len(args.seeds) > 1:
            name += f"_{len(args.seeds)}seeds"
        if len(args.folds) < 5:
            name += f"_f{''.join(str(f) for f in args.folds)}"
    if args.tta:
        name += "_tta"
    name += "_v2"
    out = os.path.join(OUTPUT_DIR, f"{name}.csv")
    sub[["id", "unstable_prob", "stable_prob"]].to_csv(out, index=False)

    print(f"\n  Saved: {out}")
    print(f"  Models: {args.backbones} × {len(all_preds)} checkpoints")
    print(f"  Folds: {args.folds} | TTA: {args.tta} | Temp: {args.temperature}")
    print(f"  Front crop: {args.front_crop} | Top crop: {args.top_crop}")
    print(f"  unstable_prob: mean={sub['unstable_prob'].mean():.4f} std={sub['unstable_prob'].std():.4f}")


def validate_on_dev(args):
    from sklearn.metrics import log_loss as sk_logloss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_csv = os.path.join(DATA_DIR, "open", "dev.csv")
    dev_dir = os.path.join(DATA_DIR, "open", "dev")
    dev_df = pd.read_csv(dev_csv)

    print(f"  Front crop: {args.front_crop} | Top crop: {args.top_crop}")
    print(f"  Folds: {args.folds}")

    ckpts = _resolve_checkpoints(args)
    if not ckpts:
        print("[ERROR] No checkpoints!")
        return

    all_preds = []
    for path, bk, label in ckpts:
        cfg = get_backbone_config(bk)
        img_size = cfg["img_size"]
        print(f"  Loading {label}")
        model = build_model(bk, pretrained=False, num_classes=2, drop_rate=0.0,
                            head_type=args.head_type)
        saved = torch.load(path, map_location=device, weights_only=True)
        model_sd = model.state_dict()
        filtered = {k: v for k, v in saved.items()
                    if k in model_sd and v.shape == model_sd[k].shape}
        model.load_state_dict(filtered, strict=False)
        model = model.to(device)

        nw = getattr(args, 'num_workers', 4)
        if args.tta:
            probs, _ = predict_tta(model, dev_csv, dev_dir, img_size, device,
                                   args.front_crop, args.top_crop, nw=nw)
        else:
            ftf, ttf = make_dual_transforms(img_size, args.front_crop, args.top_crop)
            ds = DualCropDataset(dev_csv, dev_dir, ftf, ttf)
            loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=nw, pin_memory=True)
            probs, _ = predict(model, loader, device)

        all_preds.append(probs)
        acc = (probs.argmax(1) == (dev_df['label']=='unstable').astype(int).values).mean()
        print(f"    {label}: acc={acc:.4f}")
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

    # 각 fold별 개별 성능 출력
    if len(all_preds) > 1:
        print("\n  === Per-fold Dev Performance ===")
        for i, pred in enumerate(all_preds):
            fold_ll = sk_logloss(true, pred, labels=[0, 1])
            fold_acc = (pred.argmax(1) == true).mean()
            print(f"    Fold {i}: LogLoss={fold_ll:.6f} Acc={fold_acc:.4f}")


def main():
    p = argparse.ArgumentParser(description="추론 v2: Dual-Crop + Fold Selection")
    p.add_argument("--backbones", nargs="+", default=["dinov2_large"], choices=get_backbone_choices())
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                   help="사용할 fold 목록 (예: --folds 0 2). 기본: 전체")
    p.add_argument("--checkpoint", nargs="+", type=str, default=None,
                   help="체크포인트 파일명 직접 지정 (예: --checkpoint eva_giant_vfa_fold1.pth). "
                        "--folds 대신 사용. 여러개 가능")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--tta", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--front_crop", type=float, default=0.9,
                   help="Front view CenterCrop 비율 (기본: 0.9)")
    p.add_argument("--top_crop", type=float, default=0.7,
                   help="Top view CenterCrop 비율 (기본: 0.7)")
    p.add_argument("--head_type", type=str, default="simple",
                   choices=["attn_gate", "simple"])
    p.add_argument("--validate", action="store_true", help="Dev set 검증 (fold별 성능 출력)")
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    if args.validate:
        validate_on_dev(args)
    else:
        ensemble_predict(args)


if __name__ == "__main__":
    main()
