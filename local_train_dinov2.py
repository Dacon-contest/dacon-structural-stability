"""
local_train_dinov2.py — DINOv2 로컬 학습 & 추론 & 시각화 (Windows + 12GB GPU)

사용법:
  # 학습
  python local_train_dinov2.py train --fold 0              # fold 0 학습
  python local_train_dinov2.py train --fold 0 --resume     # 이어서 학습
  python local_train_dinov2.py train --fold 0 --continue_best
  python local_train_dinov2.py train                       # 전체 5-fold 순차 학습
  python local_train_dinov2.py train --fold 1 && python local_train_dinov2.py train --fold 2 && python local_train_dinov2.py train --fold 3 && python local_train_dinov2.py train --fold 4

  # 추론 (5-fold 앙상블 → 제출 CSV)
  python local_train_dinov2.py infer                       # 다수결+best loss 앙상블
  #   앙상블 방식: ENSEMBLE_METHOD 설정 참고
  #   - "majority_best": 다수결 방향 → 해당 방향 모델 중 best val_logloss 확률 사용
  #   - "mean": 단순 평균 (비권장)

  # 시각화 (Grad-CAM)
  python local_train_dinov2.py preview --fold 0            # test 15장 Grad-CAM
  python local_train_dinov2.py preview --fold 0 --split dev
  python local_train_dinov2.py preview --fold 0 -n 30 --random
"""
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  학습 CONFIG                                                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── 백본 ─────────────────────────────────────────────────────────────
BACKBONE = "dinov2_large"        # dinov2_large (336px, 304M params)

# ── 학습 설정 ────────────────────────────────────────────────────────
EPOCHS          = 15             # 총 에폭 수
PATIENCE        = 15             # early stop patience
SEED            = 42

# ── 하이퍼파라미터 (DINOv2-L 최적화) ─────────────────────────────────
HEAD_LR         = 1e-4           # Head 학습률
BB_LR           = 1e-5           # Backbone 학습률 (자기지도 pretrain → 보수적)
WEIGHT_DECAY    = 0.05           # weight decay (ViT 권장 0.05)
DROP_RATE       = 0.3            # dropout rate
WARMUP_EPOCHS   = 2              # warmup 에폭
LAYER_DECAY     = 0.75           # DINOv2 LLRD (층별 학습률 감소)

# ── 학습 전략 ────────────────────────────────────────────────────────
LOSS            = "ce"           # ce | focal
SCHEDULER       = "cosine"       # cosine | cosine_wr
NO_MIXUP        = True           # True = Mixup/CutMix 끄기
SIMPLE_AUG      = True           # True = 단순 증강
MERGE_DEV       = True           # True = train+dev 1100개 합쳐서 K-Fold

# ── 하드웨어 (12GB GPU 기준) ─────────────────────────────────────────
BATCH_SIZE      = 2              # DINOv2-L 336px → 12GB에서 bs=2
GRAD_ACCUM      = 16             # gradient accumulation (eff_bs = 2×16 = 32)
NUM_WORKERS     = 4              # DataLoader 워커 수
GRAD_CKPT       = True           # Gradient checkpointing (304M → 필수)

# ── 기타 ─────────────────────────────────────────────────────────────
HEAD_TYPE       = "cross_attn"   # simple | attn_gate | cross_attn (token-level fusion)
FUSION_LAYERS   = 4              # Cross-view transformer layers (cross_attn용)
FUSION_HEADS    = 8              # Attention heads in fusion
USE_VIDEO       = False

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  추론 CONFIG                                                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

# --- 모델/Fold 설정 ---
INFER_FOLDS = [0, 1, 2, 3, 4]   # fold 번호로 자동 탐색
INFER_CHECKPOINT = None          # 직접 지정 시: ["dinov2_large_fold0.pth"]

# --- 앙상블 ---
# "majority_best": 다수결 방향 → 해당 방향 모델 중 best loss 확률 사용 (기본)
# "mean": 단순 평균
ENSEMBLE_METHOD  = "mean"

# --- Dual-Crop ---
FRONT_CROP = 0.9                 # Front view CenterCrop 비율
TOP_CROP   = 0.9                 # Top view CenterCrop 비율 (학습과 동일)

# --- TTA ---
USE_TTA = True                   # Test-Time Augmentation (4가지 변환)

# --- Power Sharpening ---
ALPHA = 1.0                      # alpha > 1: 날카롭게 | 1.0: 미적용

# ════════════════════════════════════════════════════════════════════════

import argparse
import datetime
import os
import random
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from train_v2 import pretrain, finetune  # noqa: E402
from models import (build_model, get_backbone_config,  # noqa: E402
                    get_train_preset, enable_gradient_checkpointing)
from inference_v2 import (make_dual_transforms, make_dual_tta_transforms,  # noqa: E402
                          DualCropDataset, predict, predict_tta)

DATA_DIR = os.path.join(BASE_DIR, "data")
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
PREVIEW_DIR = os.path.join(BASE_DIR, "previews")
OUTPUT_DIR = os.path.join(BASE_DIR, "submissions")


# =====================================================================
# Grad-CAM — 모델 주목 영역 시각화
# =====================================================================
def _to_spatial(feat, backbone_key, img_size):
    """Feature map → NCHW spatial format"""
    if feat.dim() == 4:
        return feat
    # DINOv2 (reg4): 1 CLS + 4 register = 5 prefix tokens
    n_prefix = 5 if "dinov2" in backbone_key else 1
    patches = feat[:, n_prefix:, :]
    grid = img_size // 14
    B, N, D = patches.shape
    return patches.reshape(B, grid, grid, D).permute(0, 3, 1, 2).contiguous()


def _build_cam(spatial, grad):
    """Grad-CAM: gradient-weighted activation map"""
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * spatial).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam.squeeze().detach().cpu().numpy()
    cam -= cam.min()
    cam /= max(cam.max(), 1e-8)
    return cam


def _overlay_cam(img_bgr, cam, alpha=0.35):
    """Grad-CAM / Attention heatmap을 이미지 위에 오버레이"""
    h, w = img_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(cam_resized * 255), cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)


def _compute_gradcam(model, front_t, top_t, backbone_key, img_size, target_class):
    """Dual-view Grad-CAM 계산 (simple/attn_gate 전용)"""
    bb = model.backbone
    with torch.enable_grad():
        feat_f = bb.forward_features(front_t)
        feat_t = bb.forward_features(top_t)

        sp_f = _to_spatial(feat_f, backbone_key, img_size)
        sp_t = _to_spatial(feat_t, backbone_key, img_size)
        sp_f.retain_grad()
        sp_t.retain_grad()

        gate_weights = None

        if model.head_type == "attn_gate":
            vec_f = sp_f.mean(dim=(2, 3))
            vec_t = sp_t.mean(dim=(2, 3))
            gate = model.attn_gate(torch.cat([vec_f, vec_t], dim=1))
            fused = gate[:, 0:1] * vec_f + gate[:, 1:2] * vec_t
            gate_weights = gate[0].detach().cpu().numpy()
        else:
            vec_f = sp_f.mean(dim=(2, 3))
            vec_t = sp_t.mean(dim=(2, 3))
            fused = torch.cat([vec_f, vec_t], dim=1)

        logits = model.head(fused)
        model.zero_grad()
        logits[0, target_class].backward()

    cam_f = _build_cam(sp_f, sp_f.grad)
    cam_t = _build_cam(sp_t, sp_t.grad)
    return cam_f, cam_t, gate_weights


def _compute_attention_rollout(model, front_t, top_t, backbone_key, img_size):
    """Cross-view Attention Rollout 시각화.

    CLS 토큰이 front/top 각 패치를 얼마나 attend하는지 계산.
    마지막 layer의 CLS attention + 전체 rollout 블렌딩으로
    선명하고 해석 가능한 히트맵 생성.

    Returns:
        cam_f: (H, W) front attention map (0~1)
        cam_t: (H, W) top attention map (0~1)
        probs: (2,) softmax 확률
        pred_class: int
    """
    with torch.no_grad():
        tok_f = model.get_patch_tokens(front_t)  # (1, Nf, D)
        tok_t = model.get_patch_tokens(top_t)    # (1, Nt, D)

        cls_out, attentions = model.fusion.forward_with_attention(tok_f, tok_t)
        logits = model.head(cls_out)
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()[0]
        pred_class = int(logits.argmax(1).item())

    n_f = tok_f.shape[1]
    grid = img_size // 14

    # 마지막 layer의 CLS attention (가장 task-specific)
    last_attn = attentions[-1][0].detach().mean(dim=0).cpu().float()  # (N, N)
    last_cls = last_attn[0, 1:].numpy()

    # Rollout (전체 layer 누적)
    rollout = None
    for attn_w in attentions:
        attn = attn_w[0].detach().mean(dim=0).cpu().float()
        eye = torch.eye(attn.size(0))
        attn = 0.5 * eye + 0.5 * attn
        attn = attn / attn.sum(dim=-1, keepdim=True)
        rollout = attn if rollout is None else (attn @ rollout)
    rollout_cls = rollout[0, 1:].numpy()

    # 블렌딩: 70% 마지막 layer + 30% rollout
    cls_attn = 0.7 * last_cls + 0.3 * rollout_cls

    def _normalize_cam(raw):
        # percentile clip → 상위 1% 이상은 1로 clamp (극단값 제거)
        p99 = np.percentile(raw, 99)
        v = np.clip(raw, raw.min(), max(p99, raw.min() + 1e-8))
        v = v - v.min()
        v = v / max(v.max(), 1e-8)
        return v.reshape(grid, grid)

    cam_f = _normalize_cam(cls_attn[:n_f])
    cam_t = _normalize_cam(cls_attn[n_f:])

    return cam_f, cam_t, probs, pred_class


# =====================================================================
# Train
# =====================================================================
def cmd_train(cli_args):
    """CONFIG → train_v2.finetune() namespace 변환 후 실행"""
    args = argparse.Namespace(
        backbone=BACKBONE,
        finetune_epochs=EPOCHS,
        patience=PATIENCE,
        seed=SEED,
        head_lr=HEAD_LR,
        bb_lr=BB_LR,
        weight_decay=WEIGHT_DECAY,
        drop_rate=DROP_RATE,
        warmup_epochs=WARMUP_EPOCHS,
        layer_decay=LAYER_DECAY,
        loss=LOSS,
        scheduler=SCHEDULER,
        no_mixup=NO_MIXUP,
        simple_aug=SIMPLE_AUG,
        merge_dev=MERGE_DEV,
        batch_size_override=BATCH_SIZE,
        grad_accum_override=GRAD_ACCUM,
        num_workers=NUM_WORKERS,
        grad_checkpointing=GRAD_CKPT,
        head_type=HEAD_TYPE,
        fusion_layers=FUSION_LAYERS,
        fusion_heads=FUSION_HEADS,
        use_video=USE_VIDEO,
        num_video_frames=5,
        fold=cli_args.fold,
        resume=cli_args.resume,
        init_from_best=cli_args.continue_best,
        stage="finetune",
        n_folds=5,
        pretrain_epochs=15,
        include_dev=False,
        include_dev_aug=False,
        dev_aug_repeat=3,
        video_frame_aug=False,
        no_dacon_pretrain=False,
        skip_completed=False,
    )

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {gpu} ({vram:.1f} GB)")

    eff = BATCH_SIZE * GRAD_ACCUM
    print(f"┌──────────────────────────────────────────────┐")
    print(f"│  {BACKBONE} (DINOv2 ViT-L, 304M)")
    print(f"│  epochs={EPOCHS}  bs={BATCH_SIZE}×{GRAD_ACCUM}={eff}  fold={cli_args.fold}")
    print(f"│  head_lr={HEAD_LR:.1e}  bb_lr={BB_LR:.1e}  wd={WEIGHT_DECAY}")
    print(f"│  drop={DROP_RATE}  warmup={WARMUP_EPOCHS}  LLRD={LAYER_DECAY}")
    print(f"│  loss={LOSS}  mixup={'OFF' if NO_MIXUP else 'ON'}  sched={SCHEDULER}")
    print(f"│  merge_dev={MERGE_DEV}  aug={'simple' if SIMPLE_AUG else 'full'}")
    print(f"│  head={HEAD_TYPE}  fusion_layers={FUSION_LAYERS}  fusion_heads={FUSION_HEADS}")
    print(f"│  grad_ckpt={GRAD_CKPT}  workers={NUM_WORKERS}")
    if cli_args.resume:
        print(f"│  ★ RESUME: optimizer/scheduler 이어받기")
    if cli_args.continue_best:
        print(f"│  ★ CONTINUE_BEST: best 가중치 → fresh optimizer")
    print(f"└──────────────────────────────────────────────┘")

    finetune(args)


# =====================================================================
# Inference — 추론 + 제출 파일 생성
# =====================================================================
def cmd_infer(cli_args):
    """Fold 앙상블 추론 → 제출 CSV 생성"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_csv = os.path.join(DATA_DIR, "open", "sample_submission.csv")
    test_dir = os.path.join(DATA_DIR, "open", "test")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cfg = get_backbone_config(BACKBONE)
    img_size = cfg["img_size"]
    preset = get_train_preset(BACKBONE)
    bs = preset["batch_size"] * 2

    # --- 체크포인트 수집 ---
    ckpt_list = []
    if INFER_CHECKPOINT:
        ckpts = INFER_CHECKPOINT if isinstance(INFER_CHECKPOINT, list) else [INFER_CHECKPOINT]
        for c in ckpts:
            p = os.path.join(SAVE_DIR, c)
            if os.path.exists(p):
                ckpt_list.append((p, c))
            else:
                print(f"  [WARN] 체크포인트 없음: {p}")
    else:
        for fold in INFER_FOLDS:
            p = os.path.join(SAVE_DIR, f"{BACKBONE}_fold{fold}.pth")
            if os.path.exists(p):
                ckpt_list.append((p, f"{BACKBONE}_fold{fold}"))
            else:
                print(f"  [WARN] 체크포인트 없음: {p}")

    if not ckpt_list:
        print("[ERROR] 사용 가능한 체크포인트가 없습니다!")
        return

    print(f"{'='*60}")
    print(f"추론 설정")
    print(f"  백본: {BACKBONE} | 이미지: {img_size}px | 배치: {bs}")
    print(f"  Front crop: {FRONT_CROP} | Top crop: {TOP_CROP}")
    print(f"  TTA: {USE_TTA} | Alpha: {ALPHA}")
    print(f"  앙상블: {ENSEMBLE_METHOD} | 모델 수: {len(ckpt_list)}")
    for _, label in ckpt_list:
        print(f"    - {label}")
    print(f"{'='*60}\n")

    # --- Fold별 val_logloss 읽기 (majority_best에 사용) ---
    fold_losses = {}  # label → best_val_logloss
    for _, label in ckpt_list:
        log_csv = os.path.join(SAVE_DIR, f"{label}_log.csv")
        if os.path.exists(log_csv):
            log_df = pd.read_csv(log_csv)
            fold_losses[label] = log_df["val_logloss"].min()
            print(f"  {label}: best_val_logloss={fold_losses[label]:.6f}")
        else:
            fold_losses[label] = float("inf")
            print(f"  {label}: log 없음 (val_logloss=inf)")

    # --- 모델별 추론 ---
    all_preds = []
    all_labels = []

    for ckpt_path, label in ckpt_list:
        print(f"\n▶ Loading: {label}")
        model = build_model(BACKBONE, pretrained=False, num_classes=2,
                            drop_rate=0.0, head_type=HEAD_TYPE,
                            fusion_layers=FUSION_LAYERS, fusion_heads=FUSION_HEADS)
        saved = torch.load(ckpt_path, map_location=device, weights_only=True)
        model_sd = model.state_dict()
        filtered = {k: v for k, v in saved.items()
                    if k in model_sd and v.shape == model_sd[k].shape}
        model.load_state_dict(filtered, strict=False)
        model = model.to(device)

        nw = NUM_WORKERS
        if USE_TTA:
            probs, ids = predict_tta(model, test_csv, test_dir, img_size, device,
                                     FRONT_CROP, TOP_CROP, bs=bs, nw=nw)
        else:
            ftf, ttf = make_dual_transforms(img_size, FRONT_CROP, TOP_CROP)
            ds = DualCropDataset(test_csv, test_dir, ftf, ttf)
            loader = DataLoader(ds, batch_size=bs, shuffle=False,
                                num_workers=nw, pin_memory=True)
            probs, ids = predict(model, loader, device)

        all_preds.append(probs)
        all_labels.append(label)

        _print_distribution(label, probs[:, 1])

        del model
        torch.cuda.empty_cache()

    # --- 앙상블 ---
    print(f"\n{'='*60}")
    print(f"앙상블 결합 ({ENSEMBLE_METHOD})")
    print(f"{'='*60}")

    if ENSEMBLE_METHOD == "majority_best":
        ens = _majority_best_ensemble(all_preds, all_labels, fold_losses)
    else:
        ens = np.mean(all_preds, axis=0)

    p_unstable_orig = ens[:, 1]
    _print_distribution("앙상블 결과", p_unstable_orig)

    # --- Power Sharpening ---
    if ALPHA != 1.0:
        p_unstable_sharp = _power_sharpen(p_unstable_orig, ALPHA)
        _print_distribution(f"Alpha={ALPHA} 보정 후", p_unstable_sharp)
        p_unstable_final = p_unstable_sharp
    else:
        print(f"\n  Alpha=1.0 → Power Sharpening 미적용 (원본 그대로)")
        p_unstable_final = p_unstable_orig

    # --- 제출 파일 ---
    sub = pd.read_csv(test_csv, encoding='utf-8-sig')
    eps = 1e-7
    sub["unstable_prob"] = np.clip(p_unstable_final, eps, 1 - eps)
    sub["stable_prob"] = 1.0 - sub["unstable_prob"]

    tag = BACKBONE
    if INFER_CHECKPOINT:
        ckpt_tag = '_'.join(os.path.splitext(os.path.basename(c))[0]
                            for c in (INFER_CHECKPOINT if isinstance(INFER_CHECKPOINT, list) else [INFER_CHECKPOINT]))
        tag = ckpt_tag
    else:
        if len(INFER_FOLDS) < 5:
            tag += f"_f{''.join(str(f) for f in INFER_FOLDS)}"
        else:
            tag += "_5fold"
    tta_tag = "_tta" if USE_TTA else ""
    alpha_tag = f"_a{str(ALPHA).replace('.', '')}" if ALPHA != 1.0 else ""
    ts = datetime.datetime.now().strftime("%m%d_%H%M")
    out_name = f"submission_{tag}{tta_tag}{alpha_tag}_{ts}.csv"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    sub[["id", "unstable_prob", "stable_prob"]].to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"제출 파일 저장 완료")
    print(f"{'='*60}")
    print(f"  파일: {out_name}")
    print(f"  경로: {out_path}")
    print(f"  샘플 수: {len(sub)}")
    print(f"  unstable_prob: mean={sub['unstable_prob'].mean():.10f}")
    print(f"  unstable_prob: std ={sub['unstable_prob'].std():.10f}")
    print(f"  예측 unstable: {(sub['unstable_prob'] > 0.5).sum()}개")
    print(f"  예측 stable:   {(sub['unstable_prob'] <= 0.5).sum()}개")


# =====================================================================
# Preview — Grad-CAM 주목 영역 + 추론 확률 시각화
# =====================================================================
def cmd_preview(cli_args):
    backbone = BACKBONE
    fold = cli_args.fold
    cfg = get_backbone_config(backbone)
    img_size = cfg["img_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sfx = '' if SEED == 42 else f'_s{SEED}'
    ckpt_path = os.path.join(SAVE_DIR, f"{backbone}{sfx}_fold{fold}.pth")
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] 체크포인트 없음: {ckpt_path}")
        print(f"  먼저 학습: python local_train_dinov2.py train --fold {fold}")
        return

    model = build_model(backbone, pretrained=False, num_classes=2, drop_rate=0.0,
                        head_type=HEAD_TYPE,
                        fusion_layers=FUSION_LAYERS, fusion_heads=FUSION_HEADS)
    saved = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(saved, strict=False)
    model = model.to(device).eval()
    print(f"[Preview] 모델: {ckpt_path}")

    split = cli_args.split
    if split == "test":
        csv_path = os.path.join(DATA_DIR, "open", "sample_submission.csv")
        data_dir = os.path.join(DATA_DIR, "open", "test")
        has_label = False
    elif split == "dev":
        csv_path = os.path.join(DATA_DIR, "open", "dev.csv")
        data_dir = os.path.join(DATA_DIR, "open", "dev")
        has_label = True
    else:
        csv_path = os.path.join(DATA_DIR, "open", "train.csv")
        data_dir = os.path.join(DATA_DIR, "open", "train")
        has_label = True

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    n = min(cli_args.n, len(df))
    if cli_args.random:
        random.seed(SEED)
        indices = sorted(random.sample(range(len(df)), n))
    else:
        indices = list(range(n))

    front_tf, top_tf = make_dual_transforms(img_size, front_crop_ratio=FRONT_CROP, top_crop_ratio=TOP_CROP)

    tag = f"{backbone}_fold{fold}_{split}"
    out_dir = os.path.join(PREVIEW_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    results = []
    use_attn_rollout = (HEAD_TYPE == "cross_attn")
    vis_method = "Attention Rollout" if use_attn_rollout else "Grad-CAM"
    print(f"[Preview] {split} 데이터 {n}장 — {vis_method} 시각화")
    print(f"{'─'*90}")

    for i, idx in enumerate(indices):
        row = df.iloc[idx]
        sid = row["id"]
        d = os.path.join(data_dir, sid)

        front_raw = cv2.imread(os.path.join(d, "front.png"))
        top_raw = cv2.imread(os.path.join(d, "top.png"))
        if front_raw is None or top_raw is None:
            print(f"  [SKIP] {sid}")
            continue

        front_rgb = cv2.cvtColor(front_raw, cv2.COLOR_BGR2RGB)
        top_rgb = cv2.cvtColor(top_raw, cv2.COLOR_BGR2RGB)

        front_t = front_tf(image=front_rgb)["image"].unsqueeze(0).to(device)
        top_t = top_tf(image=top_rgb)["image"].unsqueeze(0).to(device)

        cam_f = cam_t = gate = None

        if use_attn_rollout:
            # Attention Rollout: CLS → 패치 attention
            try:
                cam_f, cam_t, probs, pred_class = _compute_attention_rollout(
                    model, front_t, top_t, backbone, img_size)
                pred_label = "unstable" if pred_class == 1 else "stable"
                confidence = probs[pred_class]
            except Exception as e:
                print(f"  [WARN] Attention Rollout 실패 ({sid}): {e}")
                with torch.no_grad():
                    logits = model(front_t, top_t)
                probs = F.softmax(logits.float(), dim=1).cpu().numpy()[0]
                pred_class = int(logits.argmax(1).item())
                pred_label = "unstable" if pred_class == 1 else "stable"
                confidence = probs[pred_class]
        else:
            # Grad-CAM: simple/attn_gate
            with torch.no_grad():
                logits = model(front_t, top_t)
            probs = F.softmax(logits.float(), dim=1).cpu().numpy()[0]
            pred_class = int(logits.argmax(1).item())
            pred_label = "unstable" if pred_class == 1 else "stable"
            confidence = probs[pred_class]

            try:
                cam_f, cam_t, gate = _compute_gradcam(
                    model, front_t, top_t, backbone, img_size, pred_class)
            except Exception as e:
                print(f"  [WARN] Grad-CAM 실패 ({sid}): {e}")

        # 3. 결과 기록
        if has_label:
            true_label = row["label"]
            correct = pred_label == true_label
        else:
            true_label = None
            correct = None

        results.append({
            "id": sid, "pred": pred_label,
            "conf": confidence, "true": true_label, "correct": correct,
            "p_stable": probs[0], "p_unstable": probs[1],
        })

        # 4. 콘솔 출력 (소숫점 10자리)
        line = (f"  {i+1:03d} {sid}: "
                f"stable={probs[0]:.10f}  unstable={probs[1]:.10f}  "
                f"-> {pred_label}")
        if gate is not None:
            line += f"  [gate: F={gate[0]:.3f} T={gate[1]:.3f}]"
        if has_label:
            mark = "O" if correct else "X"
            line += f"  GT={true_label} {mark}"
        print(line)

        # 5. 이미지: [Front+CAM | Top+CAM | 정보 패널]
        h = 256
        panel_w = 300
        front_show = cv2.resize(front_raw, (h, h))
        top_show = cv2.resize(top_raw, (h, h))

        if cam_f is not None:
            front_show = _overlay_cam(front_show, cam_f)
            top_show = _overlay_cam(top_show, cam_t)

        panel = np.ones((h, panel_w, 3), dtype=np.uint8) * 245
        pred_color = (0, 0, 200) if pred_label == "unstable" else (0, 160, 0)
        cv2.putText(panel, pred_label.upper(), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)
        cv2.putText(panel, f"{confidence:.1%}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)

        bar_y, bar_w = 70, panel_w - 20
        cv2.rectangle(panel, (10, bar_y), (10 + bar_w, bar_y + 16), (220, 220, 220), -1)
        cv2.rectangle(panel, (10, bar_y), (10 + int(bar_w * probs[1]), bar_y + 16),
                      (0, 0, 200), -1)

        cv2.putText(panel, f"unstable {probs[1]:.10f}", (10, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 180), 1)
        cv2.putText(panel, f"stable   {probs[0]:.10f}", (10, 126),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 140, 0), 1)

        y_cur = 150
        if gate is not None:
            cv2.putText(panel, f"gate: Front={gate[0]:.4f} Top={gate[1]:.4f}",
                        (10, y_cur), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (100, 50, 0), 1)
            y_cur += 22

        if has_label:
            gt_color = (0, 160, 0) if correct else (0, 0, 200)
            cv2.putText(panel, f"GT: {true_label}", (10, y_cur),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, gt_color, 2)
            y_cur += 25
            cv2.putText(panel, "CORRECT" if correct else "WRONG", (10, y_cur),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, gt_color, 2)

        cv2.putText(panel, sid, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        canvas = np.concatenate([front_show, top_show, panel], axis=1)
        border_color = (200, 100, 0) if pred_label == "unstable" else (100, 100, 100)
        if has_label:
            border_color = (0, 180, 0) if correct else (0, 0, 220)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1]-1, canvas.shape[0]-1), border_color, 3)
        f_label = "FRONT + ATTN" if use_attn_rollout else "FRONT + CAM"
        t_label = "TOP + ATTN" if use_attn_rollout else "TOP + CAM"
        cv2.putText(canvas, f_label, (5, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(canvas, t_label, (h + 5, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imwrite(os.path.join(out_dir, f"{i+1:03d}_{sid}_{pred_label}.png"), canvas)

    # 요약
    print(f"{'─'*90}")
    n_done = len(results)
    print(f"[Preview] {n_done}장 저장 -> {out_dir}")
    if has_label and results:
        n_correct = sum(1 for r in results if r["correct"])
        print(f"  정확도: {n_correct}/{n_done} ({n_correct/n_done:.1%})")
        wrong = [r for r in results if not r["correct"]]
        if wrong:
            print(f"  오답 ({len(wrong)}건):")
            for r in wrong:
                print(f"    {r['id']}: pred={r['pred']}  "
                      f"stable={r['p_stable']:.10f}  "
                      f"unstable={r['p_unstable']:.10f}  GT={r['true']}")
    else:
        n_unstable = sum(1 for r in results if r["pred"] == "unstable")
        print(f"  예측 분포: stable={n_done - n_unstable} unstable={n_unstable}")

    _save_grid(out_dir)


# =====================================================================
# 유틸리티
# =====================================================================
def _majority_best_ensemble(all_preds, all_labels, fold_losses):
    """다수결 방향 확인 → 해당 방향 모델 중 best loss 모델의 확률 사용.

    각 샘플마다:
      1. 모든 모델의 예측 방향(unstable/stable) 개수 세기
      2. 다수결 방향 결정
      3. 다수결 방향으로 예측한 모델 중 val_logloss가 가장 낮은 모델의 확률 채택
      4. 만장일치 시에도 동일 로직 (best loss 모델 확률)
    """
    n_models = len(all_preds)
    n_samples = all_preds[0].shape[0]
    ens = np.zeros((n_samples, 2))

    # val_logloss 순서로 정렬 (낮을수록 좋음)
    losses = np.array([fold_losses.get(lbl, float("inf")) for lbl in all_labels])
    print(f"\n  모델별 val_logloss:")
    for lbl, loss in zip(all_labels, losses):
        print(f"    {lbl}: {loss:.6f}")

    # 각 모델의 unstable 확률 (N_models, N_samples)
    p_unstable = np.array([pred[:, 1] for pred in all_preds])  # (M, N)

    # 각 모델의 투표: unstable(1) / stable(0)
    votes = (p_unstable > 0.5).astype(int)  # (M, N)
    unstable_votes = votes.sum(axis=0)       # (N,)
    majority_unstable = unstable_votes > (n_models / 2)  # True = 다수가 unstable

    # 통계
    n_unanimous_unstable = (unstable_votes == n_models).sum()
    n_unanimous_stable = (unstable_votes == 0).sum()
    n_split = n_samples - n_unanimous_unstable - n_unanimous_stable
    print(f"\n  투표 결과:")
    print(f"    만장일치 unstable: {n_unanimous_unstable}개")
    print(f"    만장일치 stable:   {n_unanimous_stable}개")
    print(f"    분할 투표:         {n_split}개")

    for i in range(n_samples):
        if majority_unstable[i]:
            # 다수가 unstable → unstable로 투표한 모델 중 best loss
            mask = votes[:, i] == 1
        else:
            # 다수가 stable → stable로 투표한 모델 중 best loss
            mask = votes[:, i] == 0

        # mask된 모델 중 loss가 가장 낮은 모델 선택
        candidate_losses = np.where(mask, losses, float("inf"))
        best_idx = candidate_losses.argmin()
        ens[i] = all_preds[best_idx][i]

    # 사용 모델 빈도
    print(f"\n  모델별 최종 채택 빈도:")
    for m_idx in range(n_models):
        count = 0
        for i in range(n_samples):
            if majority_unstable[i]:
                mask = votes[:, i] == 1
            else:
                mask = votes[:, i] == 0
            candidate_losses = np.where(mask, losses, float("inf"))
            if candidate_losses.argmin() == m_idx:
                count += 1
        print(f"    {all_labels[m_idx]}: {count}회 ({count/n_samples:.1%})")

    return ens


def _power_sharpen(p, alpha):
    """p^α / (p^α + (1-p)^α)"""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    q = 1.0 - p
    pa, qa = p ** alpha, q ** alpha
    return pa / (pa + qa)


def _print_distribution(name, probs_unstable, decimals=10):
    """확률 분포 상세 출력"""
    fmt = f".{decimals}f"
    p = probs_unstable
    print(f"\n{'─'*60}")
    print(f"  [{name}] 확률 분포")
    print(f"{'─'*60}")
    print(f"  mean   = {p.mean():{fmt}}")
    print(f"  std    = {p.std():{fmt}}")
    print(f"  min    = {p.min():{fmt}}")
    print(f"  max    = {p.max():{fmt}}")
    print(f"  median = {np.median(p):{fmt}}")

    bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    print(f"\n  구간별 분포:")
    for lo, hi in bins:
        cnt = np.sum((p >= lo) & (p < hi)) if hi < 1.0 else np.sum((p >= lo) & (p <= hi))
        bar = "█" * (cnt // max(1, len(p) // 50))
        print(f"    [{lo:.1f}~{hi:.1f}): {cnt:5d}  {bar}")

    print(f"  unstable 예측 (>0.5): {(p > 0.5).sum()}개 / {len(p)}개")
    print(f"  stable   예측 (≤0.5): {(p <= 0.5).sum()}개 / {len(p)}개")


def _save_grid(out_dir):
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".png") and f != "grid.png")
    if not files:
        return
    imgs = [cv2.imread(os.path.join(out_dir, f)) for f in files]
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        return

    cols = 5
    h, w = imgs[0].shape[:2]
    while len(imgs) % cols != 0:
        imgs.append(np.ones((h, w, 3), dtype=np.uint8) * 240)

    rows = []
    for r in range((len(imgs) + cols - 1) // cols):
        rows.append(np.concatenate(imgs[r*cols:(r+1)*cols], axis=1))
    grid = np.concatenate(rows, axis=0)
    path = os.path.join(out_dir, "grid.png")
    cv2.imwrite(path, grid)
    print(f"  그리드: {path}")


# =====================================================================
# CLI
# =====================================================================
def main():
    p = argparse.ArgumentParser(
        description="DINOv2 로컬 학습 & 추론 & 시각화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용법:
  python local_train_dinov2.py train --fold 0              # 학습
  python local_train_dinov2.py train --fold 0 --resume     # 이어서 학습
  python local_train_dinov2.py train --fold 0 --continue_best
  python local_train_dinov2.py train                       # 전체 5-fold
  python local_train_dinov2.py infer                       # 추론 + 제출 CSV
  python local_train_dinov2.py preview --fold 0            # test 15장 Grad-CAM
  python local_train_dinov2.py preview --fold 0 --split dev

설정 변경은 파일 상단 CONFIG 섹션 직접 수정.
""",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # train
    t = sub.add_parser("train", help="학습 (설정은 파일 상단 CONFIG)")
    t.add_argument("--fold", type=int, default=None, help="fold 번호 (미지정=전체)")
    t.add_argument("--resume", action="store_true", help="optimizer/scheduler 이어받기")
    t.add_argument("--continue_best", action="store_true", help="best 가중치 → fresh optimizer")

    # infer
    sub.add_parser("infer", help="추론 + 제출 파일 생성 (설정은 파일 상단 추론 CONFIG)")

    # preview
    v = sub.add_parser("preview", help="Attention/Grad-CAM 시각화 (이미지 저장)")
    v.add_argument("--fold", type=int, required=True, help="fold 번호")
    v.add_argument("--split", default="test", choices=["test", "dev", "train"])
    v.add_argument("-n", type=int, default=20, help="샘플 수 (기본: 20)")
    v.add_argument("--random", action="store_true", default=True, help="랜덤 샘플링 (기본: ON)")

    args = p.parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "infer":
        cmd_infer(args)
    elif args.command == "preview":
        cmd_preview(args)


if __name__ == "__main__":
    main()
