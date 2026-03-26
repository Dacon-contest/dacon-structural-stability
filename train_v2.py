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
    get_train_transforms_simple,
    get_val_transforms,
    get_val_transforms_nocrop,
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

import torch.multiprocessing as _mp


def _configure_sharing_strategy():
    preferred = os.environ.get("DACON_SHARING_STRATEGY")
    candidates = [preferred] if preferred else ["file_descriptor", "file_system"]
    for strategy in candidates:
        if not strategy:
            continue
        try:
            _mp.set_sharing_strategy(strategy)
            return strategy
        except RuntimeError:
            continue
    return None


_ACTIVE_SHARING_STRATEGY = _configure_sharing_strategy()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

# Resume ckpt → 로컬(빠르고 디스크 절약), best model → SAVE_DIR(Drive)
LOCAL_CKPT_DIR = os.environ.get("DACON_LOCAL_CKPT", SAVE_DIR)
os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================================
# Layer-wise LR Decay (LLRD) for ViT
# =========================================================================
def get_vit_layer_lr_params(model, bb_lr, head_lr, layer_decay, weight_decay=0.05):
    """Layer-wise LR decay for ViT models (EVA-Giant, DINOv2 등).

    출력에 가까운 층 → 높은 LR, 입력에 가까운 층 → 낮은 LR.
    - backbone.blocks[-1]: bb_lr * layer_decay^0 = bb_lr
    - backbone.blocks[0]:  bb_lr * layer_decay^(N-1)
    - patch_embed, cls_token, pos_embed: bb_lr * layer_decay^N
    - head, attn_gate: head_lr (감쇠 없음)

    bias/norm 파라미터에는 weight_decay=0 적용.
    """
    if not hasattr(model.backbone, 'blocks'):
        return None  # ViT가 아니면 None 반환 → 호출 측에서 기본 2-group 사용

    num_layers = len(model.backbone.blocks)
    layer_scales = {}
    no_decay_keywords = ["bias", "norm", "cls_token", "pos_embed"]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Head / fusion: head_lr, 감쇠 없음
        if any(k in name for k in ["head", "attn_gate"]):
            layer_scales[name] = ("head", 1.0)
            continue

        # Backbone blocks: 깊이 기반 감쇠
        if "backbone.blocks." in name:
            block_idx = int(name.split("backbone.blocks.")[1].split(".")[0])
            depth_from_top = num_layers - block_idx - 1
            layer_scales[name] = ("bb", layer_decay ** depth_from_top)
        else:
            # patch_embed, cls_token, pos_embed, norm → 최대 감쇠
            layer_scales[name] = ("bb", layer_decay ** num_layers)

    # 파라미터 그룹 구성
    param_groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        group_type, scale = layer_scales.get(name, ("bb", 1.0))
        lr = head_lr if group_type == "head" else bb_lr * scale

        # bias, norm 등 → weight decay 0
        is_no_decay = any(kw in name for kw in no_decay_keywords)
        wd = 0.0 if is_no_decay else weight_decay

        key = (round(lr, 12), wd)
        if key not in param_groups:
            param_groups[key] = {"params": [], "lr": lr, "weight_decay": wd}
        param_groups[key]["params"].append(param)

    groups = list(param_groups.values())
    n_params = sum(len(g["params"]) for g in groups)
    lrs = sorted(set(g["lr"] for g in groups))
    print(f"  LLRD: {len(groups)} groups, {n_params} params, "
          f"layer_decay={layer_decay}, {num_layers} layers")
    print(f"  LR range: {lrs[0]:.2e} ~ {lrs[-1]:.2e}")
    return groups


def seed_suffix(seed):
    """seed!=42이면 '_s{seed}' 접미사 반환, 42면 '' (기존 호환)"""
    return '' if seed == 42 else f'_s{seed}'


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
def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device,
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
            loss = lam * loss_fn(logits, la) + (1 - lam) * loss_fn(logits, lb)
        else:
            loss = loss_fn(logits, labels)

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
def validate(model, loader, loss_fn, device):
    model.eval()
    probs_list, labels_list = [], []
    tot_loss, total = 0.0, 0

    for batch in tqdm(loader, desc="Val", leave=False):
        front, top, vf, vm, labels = unpack_batch(batch, device)
        logits = model(front, top, video_frames=vf, video_mask=vm).float()
        loss = loss_fn(logits, labels)
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
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        sd = os.path.join(data_dir, "open", split)
        for _, r in df.iterrows():
            label_int = 1 if r["label"] == "unstable" else 0
            samples.append((sd, r["id"], label_int))

    if include_dev:
        csv_path = os.path.join(data_dir, "open", "dev.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
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
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
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

    use_simple = getattr(args, 'simple_aug', False)
    print("=" * 60)
    print(f"  PRETRAIN [{args.backbone}]  img={img_size}  bs={bs}  accum={ga}")
    print(f"  aug={'simple (no CenterCrop)' if use_simple else 'full'}")
    print("=" * 60)

    if use_simple:
        train_tf = get_train_transforms_simple(img_size)
    else:
        train_tf = get_train_transforms(img_size)
    val_tf = get_val_transforms_nocrop(img_size)

    # --- ShapeStacks h=6 ---
    ss_dir = os.path.join(DATA_DIR, "shapestacks")
    ds_list = []
    if os.path.exists(ss_dir):
        ss = ShapeStacksH6Dataset(ss_dir, transform=train_tf, pairs_per_scenario=4)
        if len(ss) > 0:
            ds_list.append(ss)

    # --- Dacon train+dev (domain alignment, optional) ---
    if not getattr(args, 'no_dacon_pretrain', False):
        dacon_samples = build_dacon_samples(DATA_DIR, include_dev=True, dev_oversample=3)
        if dacon_samples:
            dacon_ds = DaconDualViewDataset(dacon_samples, transform=train_tf)
            ds_list.append(dacon_ds)
            print(f"  Dacon samples: {len(dacon_ds)} (train + dev 3×)")
    else:
        print("  Dacon data: EXCLUDED from pretrain")

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
    opt = torch.optim.AdamW(model.parameters(), lr=preset["pretrain_lr"], weight_decay=0.05)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup_ep)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.pretrain_epochs - warmup_ep, 1), eta_min=1e-6)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[warmup_ep])

    focal = FocalLoss(0.25, 2.0)
    smooth = LabelSmoothingLoss(2, 0.05)
    loss_fn = lambda logits, targets: combined_loss(logits, targets, focal, smooth)
    scaler = GradScaler("cuda")
    sfx = seed_suffix(args.seed)
    log_path = os.path.join(SAVE_DIR, f"{args.backbone}{sfx}_pretrain_log.csv")

    best_ll = float("inf")
    start_ep = 0
    ckpt_path = os.path.join(LOCAL_CKPT_DIR, f"{args.backbone}{sfx}_pretrain_ckpt.pth")
    ckpt_path_fallback = os.path.join(SAVE_DIR, f"{args.backbone}{sfx}_pretrain_ckpt.pth")
    if args.resume and not os.path.exists(ckpt_path) and os.path.exists(ckpt_path_fallback):
        ckpt_path = ckpt_path_fallback
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
            scaler.load_state_dict(ck["scaler"])
            start_ep = ck["epoch"] + 1
            best_ll = ck["best_ll"]
            print(f"  Resumed epoch {start_ep}, best={best_ll:.4f}")
        except RuntimeError:
            print(f"  [WARN] ckpt head 불일치 → backbone만 로드, 처음부터 시작")
            model_sd = model.state_dict()
            filtered = {k: v for k, v in ck["model"].items()
                        if k in model_sd and v.shape == model_sd[k].shape}
            model.load_state_dict(filtered, strict=False)
            print(f"    로드: {len(filtered)}/{len(ck['model'])} params")

    for ep in range(start_ep, args.pretrain_epochs):
        t0 = time.time()
        tm = train_one_epoch(model, train_loader, loss_fn, opt, scaler, device,
                             use_augmix=True, grad_accum=ga)
        vm = validate(model, val_loader, loss_fn, device)
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
            safe_save(model.state_dict(), os.path.join(SAVE_DIR, f"{args.backbone}{sfx}_pretrained.pth"))
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
    drop = args.drop_rate if args.drop_rate is not None else 0.3
    wd = args.weight_decay

    # LR 결정: CLI 우선 → preset 기본값
    head_lr = args.head_lr if args.head_lr else preset["lr"]
    bb_lr = args.bb_lr if args.bb_lr else preset["lr"]

    print("=" * 60)
    print(f"  FINETUNE [{args.backbone}] 5-Fold  img={img_size}  bs={bs}  accum={ga}")
    print(f"  Loss={args.loss} | Mixup={'OFF' if args.no_mixup else 'ON'} | "
          f"Sched={args.scheduler} | Head={args.head_type}")
    print(f"  head_lr={head_lr:.1e} | bb_lr={bb_lr:.1e} | wd={wd} | drop={drop}")
    if getattr(args, 'layer_decay', None):
        print(f"  LLRD: layer_decay={args.layer_decay} | warmup={getattr(args, 'warmup_epochs', 0)}ep")
    if getattr(args, 'video_frame_aug', False):
        print(f"  video_frame_aug: ON (50% 확률로 front 대신 0.1초 프레임)")
    if getattr(args, 'init_from_best', False):
        print(f"  init_from_best: ON (기존 fold best 가중치에서 이어서 학습)")
    if args.merge_dev:
        print(f"merge_dev: train+dev 합쳐서 K-Fold")
    print("=" * 60)

    # 데이터 전략 결정
    if args.merge_dev:
        # train+dev 전체 합침 → K-Fold
        all_samples = build_dacon_samples(DATA_DIR, include_dev=True, dev_oversample=1)
        dev_samples = []
        dev_aug_samples = []
    else:
        all_samples = build_dacon_samples(DATA_DIR, include_dev=args.include_dev,
                                           dev_oversample=3 if args.include_dev else 1)
        dev_samples = [] if args.include_dev else build_dev_samples(DATA_DIR)
        dev_aug_samples = []
        if args.include_dev_aug and not args.include_dev:
            dev_aug_samples = build_dev_samples(DATA_DIR) * args.dev_aug_repeat
            print(f"  Dev augment-only: {len(dev_aug_samples)} (×{args.dev_aug_repeat} oversample)")

    labels_arr = np.array([s[2] for s in all_samples])
    print(f"  Pool: {len(all_samples)} (stable={sum(labels_arr==0)}, unstable={sum(labels_arr==1)})")
    if dev_samples:
        print(f"  Dev holdout: {len(dev_samples)}")

    # 증강 선택
    if args.simple_aug:
        train_tf = get_train_transforms_simple(img_size)
        print(f"  Augmentation: simple")
    else:
        train_tf = get_train_transforms(img_size)

    sfx = seed_suffix(args.seed)
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(all_samples, labels_arr)):
        if args.fold is not None and fold != args.fold:
            continue

        ckpt_path = os.path.join(LOCAL_CKPT_DIR, f"{args.backbone}{sfx}_fold{fold}_ckpt.pth")
        ckpt_fallback = os.path.join(SAVE_DIR, f"{args.backbone}{sfx}_fold{fold}_ckpt.pth")
        best_path = os.path.join(SAVE_DIR, f"{args.backbone}{sfx}_fold{fold}.pth")

        if args.skip_completed and os.path.exists(best_path) \
                and not os.path.exists(ckpt_path) and not os.path.exists(ckpt_fallback):
            print(f"  Fold {fold} 이미 완료 (best 모델 존재) → 건너뜀")
            fold_results.append(None)
            continue

        print(f"\n{'='*40}  Fold {fold+1}/{args.n_folds}  {'='*40}")
        trn = [all_samples[i] for i in trn_idx]
        val = [all_samples[i] for i in val_idx]

        # dev 증강 학습: fold split 후 train에만 추가 (val 오염 없음)
        if dev_aug_samples:
            trn = trn + dev_aug_samples
            print(f"  Train: {len(trn)} (원본 {len(trn)-len(dev_aug_samples)} + dev_aug {len(dev_aug_samples)})")

        vfa = getattr(args, 'video_frame_aug', False)
        train_ds = DaconDualViewDataset(trn, train_tf,
                                         use_video=args.use_video, num_video_frames=args.num_video_frames,
                                         video_frame_aug=vfa)
        val_ds = DaconDualViewDataset(val, get_val_transforms(img_size),
                                       use_video=args.use_video, num_video_frames=args.num_video_frames)

        # merge_dev + 균형 데이터면 shuffle, 아니면 WeightedRandomSampler
        trn_labels = np.array([s[2] for s in trn])
        counts = np.bincount(trn_labels)
        is_balanced = len(counts) == 2 and abs(counts[0] - counts[1]) / max(counts) < 0.1

        if is_balanced or args.merge_dev:
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=True, drop_last=True)
        else:
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

        # Model (head_type 지원)
        model = build_model(args.backbone, pretrained=True, num_classes=2,
                            drop_rate=drop, use_video=args.use_video,
                            num_video_frames=args.num_video_frames,
                            head_type=args.head_type)
        if args.grad_checkpointing:
            enable_gradient_checkpointing(model)

        pt_path = os.path.join(SAVE_DIR, f"{args.backbone}_pretrained.pth")
        has_pretrained = os.path.exists(pt_path)
        if has_pretrained:
            saved = torch.load(pt_path, map_location="cpu", weights_only=True)
            # head/attn_gate 크기가 다를 수 있으므로 backbone만 로드
            model_sd = model.state_dict()
            filtered = {k: v for k, v in saved.items()
                        if k in model_sd and v.shape == model_sd[k].shape}
            skipped = set(saved.keys()) - set(filtered.keys())
            model.load_state_dict(filtered, strict=False)
            print(f"  Loaded pretrained: {pt_path} ({len(filtered)} params, skipped {len(skipped)})")
        else:
            print(f"  No pretrained ckpt — using ImageNet weights")

        # init_from_best: 기존 fold best 가중치에서 이어서 학습
        if getattr(args, 'init_from_best', False):
            fold_best = os.path.join(SAVE_DIR, f"{args.backbone}{sfx}_fold{fold}.pth")
            if os.path.exists(fold_best):
                saved = torch.load(fold_best, map_location="cpu", weights_only=True)
                model_sd = model.state_dict()
                filtered = {k: v for k, v in saved.items()
                            if k in model_sd and v.shape == model_sd[k].shape}
                model.load_state_dict(filtered, strict=False)
                print(f"  Loaded fold {fold} best weights for continued fine-tuning ({len(filtered)} params)")
            else:
                print(f"  [WARN] init_from_best: fold {fold} best 가중치 없음 → pretrained/ImageNet start")

        model = model.to(device)

        # Optimizer: LLRD (layer-wise LR decay) 또는 기존 2-group
        bb_lr_scale = 0.1 if has_pretrained else 1.0
        layer_decay = getattr(args, 'layer_decay', None)

        if layer_decay and layer_decay < 1.0:
            # LLRD: ViT 전용 per-layer param groups
            llrd_groups = get_vit_layer_lr_params(
                model, bb_lr=bb_lr * bb_lr_scale, head_lr=head_lr,
                layer_decay=layer_decay, weight_decay=wd)
            if llrd_groups:
                opt = torch.optim.AdamW(llrd_groups)
            else:
                print(f"  [WARN] LLRD not supported for {args.backbone} → 기본 2-group 사용")
                bb_params, head_params = [], []
                for name, p in model.named_parameters():
                    (head_params if any(k in name for k in ["head", "attn_gate"]) else bb_params).append(p)
                opt = torch.optim.AdamW([
                    {"params": bb_params, "lr": bb_lr * bb_lr_scale},
                    {"params": head_params, "lr": head_lr},
                ], weight_decay=wd)
        else:
            # 기존: backbone vs head 2-group
            bb_params, head_params = [], []
            for name, p in model.named_parameters():
                (head_params if any(k in name for k in ["head", "attn_gate"]) else bb_params).append(p)
            opt = torch.optim.AdamW([
                {"params": bb_params, "lr": bb_lr * bb_lr_scale},
                {"params": head_params, "lr": head_lr},
            ], weight_decay=wd)

        # Scheduler (warmup 지원)
        warmup_ep = getattr(args, 'warmup_epochs', 0)
        remaining_ep = max(args.finetune_epochs - warmup_ep, 1)

        if args.scheduler == "cosine_wr":
            base_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=10, T_mult=2)
        else:
            base_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=remaining_ep, eta_min=1e-7)

        if warmup_ep > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.01, total_iters=warmup_ep)
            sched = torch.optim.lr_scheduler.SequentialLR(
                opt, [warmup_sched, base_sched], milestones=[warmup_ep])
            print(f"  Warmup: {warmup_ep} epochs (LinearLR 0.01x → 1x)")
        else:
            sched = base_sched

        # Loss
        if args.loss == "ce":
            loss_fn = nn.CrossEntropyLoss()
        else:
            focal = FocalLoss(0.3, 2.0)
            smooth = LabelSmoothingLoss(2, 0.05)
            loss_fn = lambda logits, targets: combined_loss(logits, targets, focal, smooth)

        scaler = GradScaler("cuda")

        best_ll = float("inf")
        patience_cnt = 0
        start_ep = 0
        log_path = os.path.join(SAVE_DIR, f"{args.backbone}{sfx}_fold{fold}_log.csv")
        if not os.path.exists(ckpt_path) and os.path.exists(ckpt_fallback):
            ckpt_path = ckpt_fallback

        if args.resume and os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            # head 구조가 다른 ckpt면 backbone만 로드하고 처음부터
            try:
                model.load_state_dict(ck["model"])
                opt.load_state_dict(ck["optimizer"])
                sched.load_state_dict(ck["scheduler"])
                scaler.load_state_dict(ck["scaler"])
                start_ep = ck["epoch"] + 1
                best_ll = ck["best_ll"]
                patience_cnt = ck.get("patience", 0)
                print(f"  Resumed fold {fold} ep {start_ep}, best={best_ll:.6f}")
            except RuntimeError as e:
                print(f"  [WARN] ckpt head 불일치 → backbone만 로드, 처음부터 시작")
                model_sd = model.state_dict()
                filtered = {k: v for k, v in ck["model"].items()
                            if k in model_sd and v.shape == model_sd[k].shape}
                model.load_state_dict(filtered, strict=False)
                print(f"    로드: {len(filtered)}/{len(ck['model'])} params")

        for ep in range(start_ep, args.finetune_epochs):
            t0 = time.time()
            tm = train_one_epoch(model, train_loader, loss_fn, opt, scaler, device,
                                 use_augmix=not args.no_mixup, grad_accum=ga)
            vm = validate(model, val_loader, loss_fn, device)
            dm = validate(model, dev_loader, loss_fn, device) if dev_loader else None
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
                safe_save(model.state_dict(), os.path.join(SAVE_DIR, f"{args.backbone}{sfx}_fold{fold}.pth"))
                print(f"    >>> Saved fold {fold} (logloss={vm['logloss']:.6f})")
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f"    Early stop at ep {ep+1}")
                    break

        # 완료된 fold의 resume ckpt 삭제 (디스크 절약)
        local_ckpt = os.path.join(LOCAL_CKPT_DIR, f"{args.backbone}{sfx}_fold{fold}_ckpt.pth")
        if os.path.exists(local_ckpt):
            os.remove(local_ckpt)
            print(f"  Resume ckpt 삭제: {local_ckpt}")

        fold_results.append(best_ll)
        print(f"  Fold {fold} Best: {best_ll:.6f}")

    valid_results = [r for r in fold_results if r is not None]
    if valid_results:
        print(f"\n{'='*60}")
        print(f"  CV Mean LogLoss: {np.mean(valid_results):.6f} ± {np.std(valid_results):.6f}")
        print(f"{'='*60}")


# =========================================================================
# CLI
# =========================================================================
def main():
    p = argparse.ArgumentParser(description="구조물 안정성 예측 학습 v3")
    p.add_argument("--backbone", type=str, default="dinov2_large", choices=get_backbone_choices())
    p.add_argument("--stage", type=str, default="finetune", choices=["pretrain", "finetune", "both"])
    p.add_argument("--pretrain_epochs", type=int, default=15)
    p.add_argument("--finetune_epochs", type=int, default=30)
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
                   help="Dev를 학습 풀에 포함 (3× oversample, val fold에도 섞임)")
    p.add_argument("--include_dev_aug", action="store_true",
                   help="Dev를 증강해서 train에만 추가 (val fold 오염 없음)")
    p.add_argument("--dev_aug_repeat", type=int, default=3,
                   help="dev_aug oversample 횟수 (기본 3×)")
    p.add_argument("--merge_dev", action="store_true",
                   help="train+dev 합쳐서 K-Fold (1100개 전체 활용)")
    p.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"],
                   help="Loss 함수 (ce: CrossEntropy, focal: Focal+LabelSmoothing)")
    p.add_argument("--no_mixup", action="store_true",
                   help="Mixup/CutMix 비활성화")
    p.add_argument("--head_lr", type=float, default=None,
                   help="Head LR 직접 지정 (미지정 시 preset 사용)")
    p.add_argument("--bb_lr", type=float, default=None,
                   help="Backbone LR 직접 지정 (미지정 시 preset 사용)")
    p.add_argument("--weight_decay", type=float, default=0.01,
                   help="Weight decay (기본 0.01)")
    p.add_argument("--layer_decay", type=float, default=None,
                   help="Layer-wise LR decay for ViT (EVA-Giant: 0.9, Large: 0.8)")
    p.add_argument("--warmup_epochs", type=int, default=0,
                   help="Linear warmup epochs (기본 0)")
    p.add_argument("--scheduler", type=str, default="cosine",
                   choices=["cosine", "cosine_wr"],
                   help="스케줄러 (cosine_wr: WarmRestarts T0=10)")
    p.add_argument("--head_type", type=str, default="simple",
                   choices=["attn_gate", "simple"],
                   help="Head 구조 (simple: concat+MLP)")
    p.add_argument("--drop_rate", type=float, default=None,
                   help="Dropout rate (미지정 시 0.3)")
    p.add_argument("--simple_aug", action="store_true",
                   help="단순 증강 사용")
    p.add_argument("--video_frame_aug", action="store_true",
                   help="학습 시 50%% 확률로 front.png 대신 영상 0.1초 프레임 사용 (데이터 증강)")
    p.add_argument("--no_dacon_pretrain", action="store_true",
                   help="Pretrain에서 Dacon 대회 데이터 제외 (ShapeStacks만 사용)")
    p.add_argument("--init_from_best", action="store_true",
                   help="기존 fold best 가중치에서 이어서 학습 (fresh optimizer)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip_completed", action="store_true",
                   help="이미 best 모델이 있고 resume ckpt 없는 fold 건너뛰기")
    args = p.parse_args()

    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Backbone: {args.backbone} | Tier: {get_backbone_config(args.backbone)['tier']}")

    if args.stage in ("pretrain", "both"):
        pretrain(args)
    if args.stage in ("finetune", "both"):
        finetune(args)


if __name__ == "__main__":
    main()
