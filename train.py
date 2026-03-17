"""
학습 파이프라인: 외부 데이터 pretrain → Dacon 데이터 finetune
3-stage 학습으로 최대 성능 달성

사용법:
  python train.py --model efficientnet --stage pretrain
  python train.py --model efficientnet --stage finetune
  python train.py --model convnext --stage pretrain
  python train.py --model convnext --stage finetune
  python train.py --model swinv2 --stage pretrain
  python train.py --model swinv2 --stage finetune
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
    ArchiveBlockDataset,
    ShapeStacksDataset,
    CombinedDataset,
    apply_transform_to_crops,
    get_train_transforms,
    get_val_transforms,
    load_video_tensor,
)
from models import build_model

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


def get_model_config(model_type):
    configs = {
        "efficientnet": {
            "model_name": "tf_efficientnetv2_m.in21k_ft_in1k",
            "img_size": 384,
            "batch_size": 8,
            "lr": 2e-4,
            "pretrain_lr": 5e-4,
        },
        "convnext": {
            "model_name": "convnext_base.fb_in22k_ft_in1k",
            "img_size": 384,
            "batch_size": 4,
            "lr": 1e-4,
            "pretrain_lr": 3e-4,
        },
        "swinv2": {
            "model_name": "swinv2_small_window16_256.ms_in1k",
            "img_size": 256,
            "batch_size": 8,
            "lr": 1.5e-4,
            "pretrain_lr": 4e-4,
        },
    }
    return configs[model_type]


def resolve_batch_size(config, args):
    if args.batch_size_override is not None:
        return args.batch_size_override
    return config["batch_size"]


def resolve_stage_num_crops(args, stage):
    if stage == "pretrain" and not args.allow_expensive_pretrain:
        return 1
    return args.num_crops


def enable_gradient_checkpointing(model):
    candidate_modules = [
        getattr(model, "backbone", None),
        getattr(model, "front_backbone", None),
        getattr(model, "top_backbone", None),
        getattr(model, "video_backbone", None),
    ]
    enabled = False
    for module in candidate_modules:
        if module is not None and hasattr(module, "set_grad_checkpointing"):
            module.set_grad_checkpointing(True)
            enabled = True
    return enabled


class FocalLoss(nn.Module):
    """Focal Loss: 경계 샘플에 집중하여 학습"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Cross Entropy"""
    def __init__(self, num_classes=2, smoothing=0.05):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, logits, targets):
        confidence = 1.0 - self.smoothing
        smooth_val = self.smoothing / (self.num_classes - 1)
        one_hot = torch.full_like(logits, smooth_val)
        one_hot.scatter_(1, targets.unsqueeze(1), confidence)
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(one_hot * log_probs).sum(dim=1)
        return loss.mean()


def mixup_data(x1, x2, y, alpha=0.4):
    """MixUp augmentation"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x1.size(0)
    index = torch.randperm(batch_size, device=x1.device)
    mixed_x1 = lam * x1 + (1 - lam) * x1[index]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index]
    y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2, y_a, y_b, lam


def format_seconds(seconds):
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def append_log_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = list(row.keys())
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def print_epoch_summary(epoch, total_epochs, lr, train_metrics, val_metrics,
                        best_logloss, elapsed_sec, monitor_label, extra_metrics=None):
    train_acc = train_metrics['acc'] if train_metrics['acc'] is not None else float('nan')
    print(
        f"  Epoch [{epoch}/{total_epochs}] | lr={lr:.2e} | time={format_seconds(elapsed_sec)}"
    )
    if train_metrics['acc'] is not None:
        print(
            f"    Train  loss={train_metrics['loss']:.4f} | acc={train_acc:.4f} "
            f"(non-mixup {train_metrics['measured_samples']} samples) | "
            f"mixup={train_metrics['mixup_batches']}/{train_metrics['num_batches']} batches | "
            f"accum={train_metrics['grad_accum_steps']}"
        )
    else:
        print(
            f"    Train  loss={train_metrics['loss']:.4f} | "
            f"mixup={train_metrics['mixup_batches']}/{train_metrics['num_batches']} batches | "
            f"accum={train_metrics['grad_accum_steps']}"
        )
    print(
        f"    {monitor_label:<6} loss={val_metrics['loss']:.4f} | "
        f"logloss={val_metrics['logloss']:.4f} | acc={val_metrics['acc']:.4f}"
    )
    if extra_metrics is not None:
        print(
            f"    Dev   loss={extra_metrics['loss']:.4f} | "
            f"logloss={extra_metrics['logloss']:.4f} | acc={extra_metrics['acc']:.4f}"
        )
    print(f"    Best {monitor_label.lower()} logloss so far: {best_logloss:.4f}")


def unpack_supervised_batch(batch, device):
    if len(batch) == 3:
        front, top, labels = batch
        return front.to(device), top.to(device), None, None, labels.to(device)
    if len(batch) == 5:
        front, top, video_frames, video_mask, labels = batch
        return (
            front.to(device),
            top.to(device),
            video_frames.to(device),
            video_mask.to(device),
            labels.to(device),
        )
    raise ValueError(f"Unexpected batch structure with {len(batch)} items")


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_mixup=True,
                    grad_accum_steps=1):
    model.train()
    total_loss = 0.0
    processed_samples = 0
    correct = 0
    total = 0

    mixup_batches = 0
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="Train", leave=False)
    for step, batch in enumerate(pbar, start=1):
        front, top, video_frames, video_mask, labels = unpack_supervised_batch(batch, device)

        mixup_applied = use_mixup and random.random() < 0.5
        if mixup_applied:
            mixup_batches += 1
            front, top, labels_a, labels_b, lam = mixup_data(front, top, labels)
            with autocast('cuda'):
                logits = model(front, top, video_frames=video_frames, video_mask=video_mask)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
        else:
            with autocast('cuda'):
                logits = model(front, top, video_frames=video_frames, video_mask=video_mask)
                loss = criterion(logits, labels)

        loss_for_backward = loss / grad_accum_steps
        scaler.scale(loss_for_backward).backward()

        should_step = (step % grad_accum_steps == 0) or (step == len(loader))
        if should_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * front.size(0)
        processed_samples += front.size(0)
        preds = logits.argmax(dim=1)
        if not mixup_applied:
            correct += (preds == labels).sum().item()
            total += front.size(0)
        avg_loss = total_loss / processed_samples
        postfix = {
            "loss": f"{loss.item():.4f}",
            "avg": f"{avg_loss:.4f}",
            "mixup": f"{mixup_batches}/{step}",
            "accum": f"{grad_accum_steps}",
        }
        if total > 0:
            postfix["acc"] = f"{correct / total:.3f}"
        pbar.set_postfix(postfix)

    total_samples = len(loader.dataset)
    avg_loss = total_loss / total_samples
    return {
        "loss": avg_loss,
        "acc": (correct / total) if total > 0 else None,
        "measured_samples": total,
        "mixup_batches": mixup_batches,
        "num_batches": len(loader),
        "grad_accum_steps": grad_accum_steps,
        "elapsed_sec": time.time() - start_time,
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    total = 0
    start_time = time.time()

    for batch in tqdm(loader, desc="Val", leave=False):
        front, top, video_frames, video_mask, labels = unpack_supervised_batch(batch, device)

        with autocast('cuda'):
            logits = model(front, top, video_frames=video_frames, video_mask=video_mask)
            loss = criterion(logits, labels)

        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        total_loss += loss.item() * front.size(0)
        total += front.size(0)

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    avg_loss = total_loss / total
    logloss = log_loss(all_labels, all_probs, labels=[0, 1])
    accuracy = (all_probs.argmax(axis=1) == all_labels).mean()

    return {
        "loss": avg_loss,
        "logloss": logloss,
        "acc": accuracy,
        "elapsed_sec": time.time() - start_time,
    }


class TransformDataset(torch.utils.data.Dataset):
    """Dataset/Subset 래퍼: raw numpy 출력에 transform 적용"""

    def __init__(self, dataset, transform, num_crops=1):
        self.dataset = dataset
        self.transform = transform
        self.num_crops = num_crops

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        front, top, label = self.dataset[idx]
        front = apply_transform_to_crops(front, self.transform, self.num_crops)
        top = apply_transform_to_crops(top, self.transform, self.num_crops)
        return front, top, label


class FlexibleDualViewDataset(torch.utils.data.Dataset):
    """유연한 Dual-View 데이터셋 (train+dev 혼합 가능)"""

    def __init__(self, samples, transform=None, use_video_frames=False,
                 num_video_frames=3, num_crops=1):
        """samples: list of (data_dir, sample_id, label_int)"""
        self.samples = samples
        self.transform = transform
        self.use_video_frames = use_video_frames
        self.num_video_frames = num_video_frames
        self.num_crops = num_crops

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2
        data_dir, sample_id, label = self.samples[idx]
        sample_dir = os.path.join(data_dir, sample_id)

        front = cv2.imread(os.path.join(sample_dir, "front.png"))
        front = cv2.cvtColor(front, cv2.COLOR_BGR2RGB)
        top = cv2.imread(os.path.join(sample_dir, "top.png"))
        top = cv2.cvtColor(top, cv2.COLOR_BGR2RGB)

        front = apply_transform_to_crops(front, self.transform, self.num_crops)
        top = apply_transform_to_crops(top, self.transform, self.num_crops)

        if self.use_video_frames:
            video_path = os.path.join(sample_dir, "simulation.mp4")
            video_frames = load_video_tensor(video_path, self.transform, self.num_video_frames)
            if video_frames is None:
                if front.dim() == 4:
                    c, h, w = front.shape[1:]
                else:
                    c, h, w = front.shape
                video_frames = torch.zeros((self.num_video_frames, c, h, w), dtype=front.dtype)
                video_mask = torch.tensor(0.0, dtype=torch.float32)
            else:
                video_mask = torch.tensor(1.0, dtype=torch.float32)
            return front, top, video_frames, video_mask, torch.tensor(label, dtype=torch.long)

        return front, top, torch.tensor(label, dtype=torch.long)


def pretrain(args):
    """Stage 1: 외부 데이터로 pretrain"""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_model_config(args.model)
    img_size = config["img_size"]
    batch_size = resolve_batch_size(config, args)
    stage_num_crops = resolve_stage_num_crops(args, "pretrain")

    print("=" * 60)
    print(f" Stage 1: PRETRAIN [{args.model}]")
    print("=" * 60)

    train_tf = get_train_transforms(img_size)
    val_tf = get_val_transforms(img_size)

    # --- 데이터셋 수집 ---
    datasets_list = []

    # 1) Open 데이터 (train + dev) - 무조건 사용
    train_csv = os.path.join(DATA_DIR, "open", "train.csv")
    dev_csv = os.path.join(DATA_DIR, "open", "dev.csv")
    open_samples = []
    for csv_path, source in [(train_csv, "train"), (dev_csv, "dev")]:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                data_dir = os.path.join(DATA_DIR, "open", source)
                label_int = 1 if row["label"] == "unstable" else 0
                open_samples.append((data_dir, row["id"], label_int))
    if open_samples:
        open_ds = FlexibleDualViewDataset(open_samples, transform=None, num_crops=1)
        datasets_list.append(open_ds)
        print(f"  Open dataset (train+dev): {len(open_ds)} samples")

    # 2) Archive 블록 데이터 (default 10,000장)
    archive_csv = os.path.join(DATA_DIR, "archive", "blocks-labels.csv")
    archive_dir = os.path.join(DATA_DIR, "archive")
    if os.path.exists(archive_csv):
        max_arc = args.max_archive_samples if args.max_archive_samples > 0 else None
        archive_ds = ArchiveBlockDataset(archive_csv, archive_dir, transform=None,
                                          max_samples=max_arc, num_crops=1)
        datasets_list.append(archive_ds)
        print(f"  Archive dataset: {len(archive_ds)} samples")

    # 3) ShapeStacks 데이터 (default 전부)
    shapestacks_dir = os.path.join(DATA_DIR, "shapestacks")
    if os.path.exists(shapestacks_dir):
        max_ss = args.max_shapestacks_samples if args.max_shapestacks_samples > 0 else None
        ss_ds = ShapeStacksDataset(shapestacks_dir, transform=None,
                                    max_samples=max_ss, num_crops=1)
        if len(ss_ds) > 0:
            datasets_list.append(ss_ds)
            print(f"  ShapeStacks dataset: {len(ss_ds)} samples")

    if not datasets_list:
        print("[ERROR] No pretrain data found! Skipping pretrain.")
        return

    combined = CombinedDataset(datasets_list)
    print(f"  Total pretrain samples: {len(combined)}")
    print("  Monitor target: mixed pretrain split validation (not competition dev score)")
    if args.num_crops > 1 and stage_num_crops == 1:
        print("  Pretrain crop policy: forcing single-crop for throughput; reserve multi-crop for finetune/inference")
    if args.use_video_frames:
        print("  Pretrain video policy: disabled; video frames are only practical in finetune on Dacon train samples")

    # 90/10 split (transform 없는 상태로 분할 → 각각 다른 transform 적용)
    n = len(combined)
    n_val = max(int(n * 0.1), 100)
    n_train = n - n_val
    train_subset, val_subset = torch.utils.data.random_split(combined, [n_train, n_val])

    train_ds = TransformDataset(train_subset, train_tf, num_crops=stage_num_crops)
    val_ds = TransformDataset(val_subset, val_tf, num_crops=stage_num_crops)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_model(
        args.model,
        pretrained=True,
        num_classes=2,
        drop_rate=0.3,
        use_video=False,
        num_video_frames=args.num_video_frames,
    )
    if args.grad_checkpointing:
        gc_enabled = enable_gradient_checkpointing(model)
        print(f"  Gradient checkpointing: {'enabled' if gc_enabled else 'not supported'}")
    model = model.to(device)
    print(f"  Batch size: {batch_size} | Grad accum: {args.grad_accum_steps} | Num crops: {stage_num_crops}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["pretrain_lr"],
                                   weight_decay=0.01)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=2
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[2]
    )
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    scaler = GradScaler('cuda')
    log_path = os.path.join(SAVE_DIR, f"{args.model}_pretrain_log.csv")

    best_logloss = float("inf")
    start_epoch = 0

    # 체크포인트에서 이어서 학습
    ckpt_path = os.path.join(SAVE_DIR, f"{args.model}_pretrain_ckpt.pth")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_logloss = ckpt["best_logloss"]
        print(f"  >>> Resumed from epoch {start_epoch} (best LogLoss: {best_logloss:.4f})")

    for epoch in range(start_epoch, args.pretrain_epochs):
        print(f"\nEpoch {epoch + 1}/{args.pretrain_epochs}")
        epoch_start = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer,
                                        scaler, device, use_mixup=True,
                                        grad_accum_steps=args.grad_accum_steps)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        improved = val_metrics["logloss"] < best_logloss
        if improved:
            best_logloss = val_metrics["logloss"]

        print_epoch_summary(
            epoch=epoch + 1,
            total_epochs=args.pretrain_epochs,
            lr=current_lr,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_logloss=best_logloss,
            elapsed_sec=time.time() - epoch_start,
            monitor_label="Val",
        )

        append_log_row(log_path, {
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_acc_non_mixup": train_metrics["acc"],
            "mixup_batches": train_metrics["mixup_batches"],
            "val_loss": val_metrics["loss"],
            "val_logloss": val_metrics["logloss"],
            "val_acc": val_metrics["acc"],
            "best_val_logloss": best_logloss,
            "epoch_time_sec": time.time() - epoch_start,
        })

        if improved:
            save_path = os.path.join(SAVE_DIR, f"{args.model}_pretrained.pth")
            torch.save(model.state_dict(), save_path)
            print(f"    >>> Saved best pretrained model (LogLoss: {val_metrics['logloss']:.4f})")

        # 매 에폭 체크포인트 저장 (중단 대비)
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_logloss": best_logloss,
        }, ckpt_path)

    print(f"\n  Best pretrain LogLoss: {best_logloss:.4f}")


def finetune(args):
    """Stage 2: Dacon 대회 데이터로 K-Fold finetune"""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_model_config(args.model)
    img_size = config["img_size"]
    batch_size = resolve_batch_size(config, args)
    stage_num_crops = resolve_stage_num_crops(args, "finetune")

    print("=" * 60)
    print(f" Stage 2: FINETUNE [{args.model}] (5-Fold)")
    print("=" * 60)

    # 기본값은 train만 사용해 honest CV를 측정하고, dev는 외부 holdout으로만 본다.
    train_csv = os.path.join(DATA_DIR, "open", "train.csv")
    dev_csv = os.path.join(DATA_DIR, "open", "dev.csv")
    train_df = pd.read_csv(train_csv)
    dev_df = pd.read_csv(dev_csv)

    train_df["source"] = "train"
    dev_df["source"] = "dev"
    train_df["label_int"] = (train_df["label"] == "unstable").astype(int)
    dev_df["label_int"] = (dev_df["label"] == "unstable").astype(int)

    if args.include_dev_in_finetune:
        finetune_df = pd.concat([train_df, dev_df], ignore_index=True)
        print(f"  Training pool: {len(finetune_df)} samples (train+dev)")
        print("  Dev holdout monitor: disabled because dev is included in training")
    else:
        finetune_df = train_df.copy()
        print(f"  Training pool: {len(finetune_df)} train samples")
        print(f"  External dev holdout: {len(dev_df)} samples (same env as test)")

    print(f"  Label dist: stable={sum(finetune_df['label_int']==0)}, "
          f"unstable={sum(finetune_df['label_int']==1)}")
    print("  Model selection target: fold validation logloss")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(finetune_df, finetune_df["label_int"])):
        if args.fold is not None and fold != args.fold:
            continue

        print(f"\n{'=' * 40}")
        print(f" Fold {fold + 1}/{args.n_folds}")
        print(f"{'=' * 40}")

        fold_train_df = finetune_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = finetune_df.iloc[val_idx].reset_index(drop=True)

        # 데이터셋 구성 (source에 따라 data_dir 결정)
        train_samples = []
        for _, row in fold_train_df.iterrows():
            data_dir = os.path.join(DATA_DIR, "open", row["source"])
            train_samples.append((data_dir, row["id"], row["label_int"]))

        val_samples = []
        for _, row in fold_val_df.iterrows():
            data_dir = os.path.join(DATA_DIR, "open", row["source"])
            val_samples.append((data_dir, row["id"], row["label_int"]))

        train_ds = FlexibleDualViewDataset(
            train_samples,
            get_train_transforms(img_size),
            use_video_frames=args.use_video_frames,
            num_video_frames=args.num_video_frames,
            num_crops=stage_num_crops,
        )
        val_ds = FlexibleDualViewDataset(
            val_samples,
            get_val_transforms(img_size),
            use_video_frames=args.use_video_frames,
            num_video_frames=args.num_video_frames,
            num_crops=stage_num_crops,
        )
        dev_loader = None
        if not args.include_dev_in_finetune:
            dev_samples = []
            for _, row in dev_df.iterrows():
                data_dir = os.path.join(DATA_DIR, "open", row["source"])
                dev_samples.append((data_dir, row["id"], row["label_int"]))
            dev_ds = FlexibleDualViewDataset(
                dev_samples,
                get_val_transforms(img_size),
                use_video_frames=args.use_video_frames,
                num_video_frames=args.num_video_frames,
                num_crops=stage_num_crops,
            )
            dev_loader = DataLoader(dev_ds, batch_size=batch_size,
                                    shuffle=False, num_workers=args.num_workers, pin_memory=True)

        # Weighted sampler for class imbalance
        labels_array = fold_train_df["label_int"].values
        class_counts = np.bincount(labels_array)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels_array]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                       sampler=sampler, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size,
                     shuffle=False, num_workers=args.num_workers, pin_memory=True)

        # 모델 생성 및 pretrained 가중치 로드
        model = build_model(
            args.model,
            pretrained=True,
            num_classes=2,
            drop_rate=0.4,
            use_video=args.use_video_frames if args.model == "swinv2" else False,
            num_video_frames=args.num_video_frames,
        )
        if args.grad_checkpointing:
            gc_enabled = enable_gradient_checkpointing(model)
            print(f"  Gradient checkpointing: {'enabled' if gc_enabled else 'not supported'}")
        pretrained_path = os.path.join(SAVE_DIR, f"{args.model}_pretrained.pth")
        if os.path.exists(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path, map_location="cpu", weights_only=True),
                                  strict=False)
            print(f"  Loaded pretrained weights: {pretrained_path}")
        model = model.to(device)
        print(f"  Batch size: {batch_size} | Grad accum: {args.grad_accum_steps} | Num crops: {stage_num_crops}")

        # Differential learning rate
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if "head" in name or "fusion" in name or "attn_gate" in name or "bilinear" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": config["lr"] * 0.1},
            {"params": head_params, "lr": config["lr"]},
        ], weight_decay=0.02)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.finetune_epochs, eta_min=1e-7
        )

        # Focal + Label Smoothing combined loss
        focal = FocalLoss(alpha=0.3, gamma=2.0)
        smooth = LabelSmoothingLoss(num_classes=2, smoothing=0.05)
        criterion = lambda logits, targets: 0.7 * focal(logits, targets) + 0.3 * smooth(logits, targets)

        scaler = GradScaler('cuda')
        best_logloss = float("inf")
        patience_count = 0
        start_epoch = 0
        log_path = os.path.join(SAVE_DIR, f"{args.model}_fold{fold}_log.csv")

        # 체크포인트에서 이어서 학습
        fold_ckpt_path = os.path.join(SAVE_DIR, f"{args.model}_fold{fold}_ckpt.pth")
        if args.resume and os.path.exists(fold_ckpt_path):
            ckpt = torch.load(fold_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"] + 1
            best_logloss = ckpt["best_logloss"]
            patience_count = ckpt.get("patience_count", 0)
            print(f"  >>> Resumed fold {fold} from epoch {start_epoch} (best LogLoss: {best_logloss:.6f})")

        for epoch in range(start_epoch, args.finetune_epochs):
            print(f"\n  Epoch {epoch + 1}/{args.finetune_epochs}")
            epoch_start = time.time()
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer,
                                            scaler, device, use_mixup=True,
                                            grad_accum_steps=args.grad_accum_steps)
            val_metrics = validate(model, val_loader, criterion, device)
            dev_metrics = validate(model, dev_loader, criterion, device) if dev_loader is not None else None
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            improved = val_metrics["logloss"] < best_logloss
            if improved:
                best_logloss = val_metrics["logloss"]

            print_epoch_summary(
                epoch=epoch + 1,
                total_epochs=args.finetune_epochs,
                lr=current_lr,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_logloss=best_logloss,
                elapsed_sec=time.time() - epoch_start,
                monitor_label="Val",
                extra_metrics=dev_metrics,
            )

            append_log_row(log_path, {
                "epoch": epoch + 1,
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_acc_non_mixup": train_metrics["acc"],
                "mixup_batches": train_metrics["mixup_batches"],
                "fold_val_loss": val_metrics["loss"],
                "fold_val_logloss": val_metrics["logloss"],
                "fold_val_acc": val_metrics["acc"],
                "dev_loss": dev_metrics["loss"] if dev_metrics is not None else None,
                "dev_logloss": dev_metrics["logloss"] if dev_metrics is not None else None,
                "dev_acc": dev_metrics["acc"] if dev_metrics is not None else None,
                "best_fold_logloss": best_logloss,
                "epoch_time_sec": time.time() - epoch_start,
            })

            # 매 에폭 체크포인트 저장 (중단 대비)
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_logloss": best_logloss,
                "patience_count": patience_count,
            }, fold_ckpt_path)

            if improved:
                save_path = os.path.join(SAVE_DIR, f"{args.model}_fold{fold}.pth")
                torch.save(model.state_dict(), save_path)
                print(f"    >>> Saved best fold {fold} model (LogLoss: {val_metrics['logloss']:.6f})")
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= args.patience:
                    print(f"    Early stopping at epoch {epoch + 1}")
                    break

        fold_results.append(best_logloss)
        print(f"\n  Fold {fold} Best LogLoss: {best_logloss:.6f}")

    if fold_results:
        print(f"\n{'=' * 60}")
        print(f"  CV Mean LogLoss: {np.mean(fold_results):.6f} +/- {np.std(fold_results):.6f}")
        print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="구조물 안정성 예측 학습")
    parser.add_argument("--model", type=str, default="efficientnet",
                        choices=["efficientnet", "convnext", "swinv2"])
    parser.add_argument("--stage", type=str, default="finetune",
                        choices=["pretrain", "finetune", "both"])
    parser.add_argument("--pretrain_epochs", type=int, default=15)
    parser.add_argument("--finetune_epochs", type=int, default=50)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None, help="특정 fold만 학습")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_archive_samples", type=int, default=10000)
    parser.add_argument("--max_shapestacks_samples", type=int, default=0,
                        help="0=use all")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size_override", type=int, default=None,
                        help="Override per-step batch size to reduce GPU memory")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps to preserve effective batch size")
    parser.add_argument("--grad_checkpointing", action="store_true",
                        help="Enable gradient checkpointing on supported backbones to reduce VRAM")
    parser.add_argument("--num_crops", type=int, default=3)
    parser.add_argument("--allow_expensive_pretrain", action="store_true",
                        help="Allow multi-crop style expensive pretrain; default keeps pretrain single-crop for throughput")
    parser.add_argument("--use_video_frames", action="store_true",
                        help="Use simulation.mp4 frames when available (mainly useful for finetune on train samples)")
    parser.add_argument("--num_video_frames", type=int, default=3)
    parser.add_argument("--include_dev_in_finetune", action="store_true",
                        help="Use dev in finetune training pool (disables honest dev holdout monitor)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    if args.stage in ("pretrain", "both"):
        pretrain(args)

    if args.stage in ("finetune", "both"):
        finetune(args)


if __name__ == "__main__":
    main()
