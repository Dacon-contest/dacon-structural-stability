"""
학습 파이프라인 v2
  Stage 1: ShapeStacks h=6 pretrain (dual-view 카메라 페어링)
  Stage 2: 5-Fold CV finetune (Dacon train + dev)

핵심 기법:
  - Mixup + CutMix (배치 단위 랜덤 선택)
  - Focal Loss + Label Smoothing 결합
  - AMP fp16 + Gradient Accumulation + Gradient Checkpointing
  - Warmup + CosineAnnealing 스케줄러
  - Dev 3× oversample 옵션
  - 비디오 프레임 활용 옵션

Usage (local 12GB):
  python train.py --backbone eva02_large --stage pretrain
  python train.py --backbone eva02_large --stage finetune --fold 0

Usage (Colab A100):
  python train.py --backbone eva_giant --stage both --include_dev --grad_checkpointing
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from tqdm import tqdm

from datasets import (
    DaconDualViewDataset,
    ShapeStacksH6Dataset,
    CombinedDataset,
    get_train_transforms,
    get_val_transforms,
    cutmix_data,
    mixup_data,
)
from models import (
    build_model,
    get_backbone_config,
    get_train_preset,
    get_backbone_choices,
    enable_gradient_checkpointing,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================================
# Losses
# =========================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes=2, smoothing=0.05):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, logits, targets):
        conf = 1.0 - self.smoothing
        sv = self.smoothing / (self.num_classes - 1)
        one_hot = torch.full_like(logits, sv)
        one_hot.scatter_(1, targets.unsqueeze(1), conf)
        return -(one_hot * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def combined_loss(logits, targets, focal, smooth):
    return 0.7 * focal(logits, targets) + 0.3 * smooth(logits, targets)


# =========================================================================
# Helpers
# =========================================================================
def safe_save(obj, path):
    """원자적 저장 — 중간에 끊겨도 기존 체크포인트 보존"""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def fmt(sec):
    s = int(sec)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def log_row(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def unpack_batch(batch, device):
    if len(batch) == 3:
        fr, tp, lb = batch
        return fr.to(device), tp.to(device), None, None, lb.to(device)
    if len(batch) == 5:
        fr, tp, vf, vm, lb = batch
        return fr.to(device), tp.to(device), vf.to(device), vm.to(device), lb.to(device)
    raise ValueError(f"Unexpected batch len {len(batch)}")


# =========================================================================
# Train / Validate
# =========================================================================
def train_one_epoch(model, loader, focal, smooth, optimizer, scaler, device,
                    use_augmix=True, grad_accum=1):
    model.train()
    tot_loss, n_samples, correct, total = 0.0, 0, 0, 0
    n_mix, n_cut, n_skip = 0, 0, 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="Train", leave=False)
    for step, batch in enumerate(pbar, 1):
        front, top, vf, vm, labels = unpack_batch(batch, device)
        B = front.size(0)

        # Random: 30% Mixup, 30% CutMix, 40% clean
        aug_type = "clean"
        if use_augmix:
            r = random.random()
            if r < 0.30:
                aug_type = "mixup"
                front, top, la, lb, lam = mixup_data(front, top, labels)
                n_mix += 1
            elif r < 0.60:
                aug_type = "cutmix"
                front, top, la, lb, lam = cutmix_data(front, top, labels)
                n_cut += 1

        with autocast("cuda", enabled=device.type == "cuda"):
            logits = model(front, top, video_frames=vf, video_mask=vm)
        logits = logits.float()

        if not torch.isfinite(logits).all():
            n_skip += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        if aug_type != "clean":
            loss = lam * combined_loss(logits, la, focal, smooth) + \
                   (1 - lam) * combined_loss(logits, lb, focal, smooth)
        else:
            loss = combined_loss(logits, labels, focal, smooth)

        if not torch.isfinite(loss):
            n_skip += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss / grad_accum).backward()
        if step % grad_accum == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        tot_loss += loss.item() * B
        n_samples += B
        if aug_type == "clean":
            correct += (logits.argmax(1) == labels).sum().item()
            total += B
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         avg=f"{tot_loss / max(n_samples, 1):.4f}",
                         mix=n_mix, cut=n_cut, skip=n_skip)

    return {
        "loss": tot_loss / max(n_samples, 1),
        "acc": correct / total if total > 0 else None,
        "n_mix": n_mix, "n_cut": n_cut, "n_skip": n_skip,
        "n_batches": len(loader),
    }


@torch.no_grad()
def validate(model, loader, focal, smooth, device):
    model.eval()
    probs_list, labels_list = [], []
    tot_loss, total = 0.0, 0

    for batch in tqdm(loader, desc="Val", leave=False):
        front, top, vf, vm, labels = unpack_batch(batch, device)
        logits = model(front, top, video_frames=vf, video_mask=vm).float()
        loss = combined_loss(logits, labels, focal, smooth)
        probs = F.softmax(logits, dim=1)
        probs_list.append(probs.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
        tot_loss += loss.item() * front.size(0)
        total += front.size(0)

    probs = np.concatenate(probs_list)
    labels = np.concatenate(labels_list)
    return {
        "loss": tot_loss / total,
        "logloss": log_loss(labels, probs, labels=[0, 1]),
        "acc": (probs.argmax(1) == labels).mean(),
    }


# =========================================================================
# Build sample lists
# =========================================================================
def build_dacon_samples(data_dir, include_dev=False, dev_oversample=1):
    """Dacon train (+ optional dev) 샘플 목록 생성"""
    samples = []
    for split in ["train"]:
        csv_path = os.path.join(data_dir, "open", f"{split}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        sd = os.path.join(data_dir, "open", split)
        for _, r in df.iterrows():
            label_int = 1 if r["label"] == "unstable" else 0
            samples.append((sd, r["id"], label_int))

    if include_dev:
        csv_path = os.path.join(data_dir, "open", "dev.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            sd = os.path.join(data_dir, "open", "dev")
            for _ in range(dev_oversample):
                for _, r in df.iterrows():
                    label_int = 1 if r["label"] == "unstable" else 0
                    samples.append((sd, r["id"], label_int))
    return samples


def build_dev_samples(data_dir):
    csv_path = os.path.join(data_dir, "open", "dev.csv")
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    sd = os.path.join(data_dir, "open", "dev")
    return [(sd, r["id"], 1 if r["label"] == "unstable" else 0) for _, r in df.iterrows()]


# =========================================================================
# Stage 1: Pretrain
# =========================================================================
def pretrain(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_backbone_config(args.backbone)
    preset = get_train_preset(args.backbone)
    img_size = cfg["img_size"]
    bs = args.batch_size_override or preset["batch_size"]
    ga = args.grad_accum_override or preset["grad_accum"]

    print("=" * 60)
    print(f"  PRETRAIN [{args.backbone}]  img={img_size}  bs={bs}  accum={ga}")
    print("=" * 60)

    train_tf = get_train_transforms(img_size)
    val_tf = get_val_transforms(img_size)

    # --- ShapeStacks h=6 ---
    ss_dir = os.path.join(DATA_DIR, "shapestacks")
    ds_list = []
    if os.path.exists(ss_dir):
        ss = ShapeStacksH6Dataset(ss_dir, transform=train_tf, pairs_per_scenario=4)
        if len(ss) > 0:
            ds_list.append(ss)

    # --- Dacon train+dev (domain alignment) ---
    dacon_samples = build_dacon_samples(DATA_DIR, include_dev=True, dev_oversample=3)
    if dacon_samples:
        dacon_ds = DaconDualViewDataset(dacon_samples, transform=train_tf)
        ds_list.append(dacon_ds)
        print(f"  Dacon samples: {len(dacon_ds)} (train + dev 3×)")

    if not ds_list:
        print("[ERROR] No pretrain data!")
        return

    combined = CombinedDataset(ds_list)
    n = len(combined)
    n_val = max(int(n * 0.1), 100)
    train_sub, val_sub = torch.utils.data.random_split(combined, [n - n_val, n_val])

    # val subset → val_tf (no augmentation), need wrapper
    class ValWrap(torch.utils.data.Dataset):
        def __init__(self, subset, val_tf):
            self.subset = subset
            self.val_tf = val_tf
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            item = self.subset[idx]
            # items are already transformed by the original dataset
            return item

    train_loader = DataLoader(train_sub, batch_size=bs, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_sub, batch_size=bs, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    print(f"  Total: {n}  train: {len(train_sub)}  val: {len(val_sub)}")

    model = build_model(args.backbone, pretrained=True, num_classes=2,
                        drop_rate=0.3, use_video=False)
    if args.grad_checkpointing:
        ok = enable_gradient_checkpointing(model)
        print(f"  Gradient checkpointing: {'ON' if ok else 'N/A'}")
    model = model.to(device)

    warmup_ep = min(5, max(args.pretrain_epochs // 4, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=preset["pretrain_lr"], weight_decay=0.01)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup_ep)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.pretrain_epochs - warmup_ep, 1), eta_min=1e-6)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[warmup_ep])

    focal = FocalLoss(0.25, 2.0)
    smooth = LabelSmoothingLoss(2, 0.05)
    scaler = GradScaler("cuda")
    log_path = os.path.join(SAVE_DIR, f"{args.backbone}_pretrain_log.csv")

    best_ll = float("inf")
    start_ep = 0
    ckpt_path = os.path.join(SAVE_DIR, f"{args.backbone}_pretrain_ckpt.pth")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_ep = ck["epoch"] + 1
        best_ll = ck["best_ll"]
        print(f"  Resumed epoch {start_ep}, best={best_ll:.4f}")

    for ep in range(start_ep, args.pretrain_epochs):
        t0 = time.time()
        tm = train_one_epoch(model, train_loader, focal, smooth, opt, scaler, device,
                             use_augmix=True, grad_accum=ga)
        vm = validate(model, val_loader, focal, smooth, device)
        sched.step()
        lr = opt.param_groups[0]["lr"]
        improved = vm["logloss"] < best_ll
        if improved:
            best_ll = vm["logloss"]

        acc_str = f"acc={tm['acc']:.4f}" if tm['acc'] is not None else "acc=N/A"
        print(f"  Ep [{ep+1}/{args.pretrain_epochs}] lr={lr:.2e} {fmt(time.time()-t0)} | "
              f"train loss={tm['loss']:.4f} {acc_str} mix={tm['n_mix']} cut={tm['n_cut']} | "
              f"val logloss={vm['logloss']:.4f} acc={vm['acc']:.4f} | best={best_ll:.4f}")

        log_row(log_path, {"epoch": ep+1, "lr": lr, "train_loss": tm["loss"],
                           "val_logloss": vm["logloss"], "val_acc": vm["acc"], "best": best_ll})

        if improved:
            safe_save(model.state_dict(), os.path.join(SAVE_DIR, f"{args.backbone}_pretrained.pth"))
            print(f"    >>> Saved (logloss={vm['logloss']:.4f})")

        safe_save({"epoch": ep, "model": model.state_dict(), "optimizer": opt.state_dict(),
                   "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                   "best_ll": best_ll}, ckpt_path)

    print(f"\n  Pretrain done. Best LogLoss: {best_ll:.4f}")


# =========================================================================
# Stage 2: Finetune (5-Fold CV)
# =========================================================================
def finetune(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_backbone_config(args.backbone)
    preset = get_train_preset(args.backbone)
    img_size = cfg["img_size"]
    bs = args.batch_size_override or preset["batch_size"]
    ga = args.grad_accum_override or preset["grad_accum"]

    print("=" * 60)
    print(f"  FINETUNE [{args.backbone}] 5-Fold  img={img_size}  bs={bs}  accum={ga}")
    print("=" * 60)

    # 샘플 목록
    all_samples = build_dacon_samples(DATA_DIR, include_dev=args.include_dev,
                                       dev_oversample=3 if args.include_dev else 1)
    dev_samples = [] if args.include_dev else build_dev_samples(DATA_DIR)

    labels_arr = np.array([s[2] for s in all_samples])
    print(f"  Pool: {len(all_samples)} (stable={sum(labels_arr==0)}, unstable={sum(labels_arr==1)})")
    if dev_samples:
        print(f"  Dev holdout: {len(dev_samples)}")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(all_samples, labels_arr)):
        if args.fold is not None and fold != args.fold:
            continue

        print(f"\n{'='*40}  Fold {fold+1}/{args.n_folds}  {'='*40}")
        trn = [all_samples[i] for i in trn_idx]
        val = [all_samples[i] for i in val_idx]

        train_ds = DaconDualViewDataset(trn, get_train_transforms(img_size),
                                         use_video=args.use_video, num_video_frames=args.num_video_frames)
        val_ds = DaconDualViewDataset(val, get_val_transforms(img_size),
                                       use_video=args.use_video, num_video_frames=args.num_video_frames)

        # Weighted sampler
        trn_labels = np.array([s[2] for s in trn])
        counts = np.bincount(trn_labels)
        wts = 1.0 / counts
        sw = wts[trn_labels]
        sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
        dev_loader = None
        if dev_samples:
            dev_ds = DaconDualViewDataset(dev_samples, get_val_transforms(img_size),
                                           use_video=args.use_video, num_video_frames=args.num_video_frames)
            dev_loader = DataLoader(dev_ds, batch_size=bs, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)

        # Model
        model = build_model(args.backbone, pretrained=True, num_classes=2,
                            drop_rate=0.4, use_video=args.use_video,
                            num_video_frames=args.num_video_frames)
        if args.grad_checkpointing:
            enable_gradient_checkpointing(model)

        pt_path = os.path.join(SAVE_DIR, f"{args.backbone}_pretrained.pth")
        if os.path.exists(pt_path):
            model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True), strict=False)
            print(f"  Loaded pretrained: {pt_path}")
        model = model.to(device)

        # Differential LR
        bb_params, head_params = [], []
        for name, p in model.named_parameters():
            (head_params if any(k in name for k in ["head", "attn_gate"]) else bb_params).append(p)
        opt = torch.optim.AdamW([
            {"params": bb_params, "lr": preset["lr"] * 0.1},
            {"params": head_params, "lr": preset["lr"]},
        ], weight_decay=0.02)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.finetune_epochs, eta_min=1e-7)
        focal = FocalLoss(0.3, 2.0)
        smooth = LabelSmoothingLoss(2, 0.05)
        scaler = GradScaler("cuda")

        best_ll = float("inf")
        patience_cnt = 0
        start_ep = 0
        log_path = os.path.join(SAVE_DIR, f"{args.backbone}_fold{fold}_log.csv")
        ckpt_path = os.path.join(SAVE_DIR, f"{args.backbone}_fold{fold}_ckpt.pth")

        if args.resume and os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
            scaler.load_state_dict(ck["scaler"])
            start_ep = ck["epoch"] + 1
            best_ll = ck["best_ll"]
            patience_cnt = ck.get("patience", 0)
            print(f"  Resumed fold {fold} ep {start_ep}, best={best_ll:.6f}")

        for ep in range(start_ep, args.finetune_epochs):
            t0 = time.time()
            tm = train_one_epoch(model, train_loader, focal, smooth, opt, scaler, device,
                                 use_augmix=True, grad_accum=ga)
            vm = validate(model, val_loader, focal, smooth, device)
            dm = validate(model, dev_loader, focal, smooth, device) if dev_loader else None
            sched.step()
            lr = opt.param_groups[0]["lr"]
            improved = vm["logloss"] < best_ll
            if improved:
                best_ll = vm["logloss"]

            acc_str = f"acc={tm['acc']:.4f}" if tm['acc'] is not None else "acc=N/A"
            line = (f"  Ep [{ep+1}/{args.finetune_epochs}] lr={lr:.2e} {fmt(time.time()-t0)} | "
                    f"train loss={tm['loss']:.4f} {acc_str} | "
                    f"val logloss={vm['logloss']:.4f} acc={vm['acc']:.4f}")
            if dm:
                line += f" | dev logloss={dm['logloss']:.4f} acc={dm['acc']:.4f}"
            line += f" | best={best_ll:.6f}"
            print(line)

            log_row(log_path, {"epoch": ep+1, "lr": lr, "train_loss": tm["loss"],
                               "val_logloss": vm["logloss"], "val_acc": vm["acc"],
                               "dev_logloss": dm["logloss"] if dm else None,
                               "dev_acc": dm["acc"] if dm else None,
                               "best": best_ll})

            safe_save({"epoch": ep, "model": model.state_dict(), "optimizer": opt.state_dict(),
                       "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                       "best_ll": best_ll, "patience": patience_cnt}, ckpt_path)

            if improved:
                safe_save(model.state_dict(), os.path.join(SAVE_DIR, f"{args.backbone}_fold{fold}.pth"))
                print(f"    >>> Saved fold {fold} (logloss={vm['logloss']:.6f})")
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f"    Early stop at ep {ep+1}")
                    break

        fold_results.append(best_ll)
        print(f"  Fold {fold} Best: {best_ll:.6f}")

    if fold_results:
        print(f"\n{'='*60}")
        print(f"  CV Mean LogLoss: {np.mean(fold_results):.6f} ± {np.std(fold_results):.6f}")
        print(f"{'='*60}")


# =========================================================================
# CLI
# =========================================================================
def main():
    p = argparse.ArgumentParser(description="구조물 안정성 예측 학습 v2")
    p.add_argument("--backbone", type=str, default="dinov2_large", choices=get_backbone_choices())
    p.add_argument("--stage", type=str, default="finetune", choices=["pretrain", "finetune", "both"])
    p.add_argument("--pretrain_epochs", type=int, default=15)
    p.add_argument("--finetune_epochs", type=int, default=50)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--batch_size_override", type=int, default=None)
    p.add_argument("--grad_accum_override", type=int, default=None)
    p.add_argument("--grad_checkpointing", action="store_true")
    p.add_argument("--use_video", action="store_true")
    p.add_argument("--num_video_frames", type=int, default=5)
    p.add_argument("--include_dev", action="store_true",
                   help="Dev를 학습 풀에 포함 (3× oversample)")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Backbone: {args.backbone} | Tier: {get_backbone_config(args.backbone)['tier']}")

    if args.stage in ("pretrain", "both"):
        pretrain(args)
    if args.stage in ("finetune", "both"):
        finetune(args)


if __name__ == "__main__":
    main()
