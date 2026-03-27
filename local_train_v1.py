"""
local_train_v1.py — 로컬 학습 & 시각화 (Windows + 12GB GPU)

사용법:
  python local_train_v1.py train --fold 0           # 학습 시작
  python local_train_v1.py train --fold 0 --resume  # 이어서 학습
  python local_train_v1.py train                    # 전체 5-fold
  python local_train_v1.py preview --fold 0         # test 15장 시각화
  python local_train_v1.py preview --fold 0 --split dev   # dev 정답 비교
  python local_train_v1.py preview --fold 0 -n 30 --random
"""
# ── 백본 ─────────────────────────────────────────────────────────────
BACKBONE = "convnext_small"      # convnext_small | eva02_large | dinov2_large | eva_giant

# ── 학습 설정 ────────────────────────────────────────────────────────
EPOCHS          = 10             # 총 에폭 수
PATIENCE        = 10             # early stop patience
SEED            = 42

# ── 하이퍼파라미터 ───────────────────────────────────────────────────
HEAD_LR         = 2e-4           # Head 학습률
BB_LR           = 2e-5           # Backbone 학습률
WEIGHT_DECAY    = 0.01           # weight decay (ViT: 0.05 권장)
DROP_RATE       = 0.3            # dropout rate
WARMUP_EPOCHS   = 3              # warmup 에폭 (0이면 OFF)
LAYER_DECAY     = None           # ViT LLRD (dinov2: 0.75, eva02: 0.8, eva_giant: 0.9)

# ── 학습 전략 ────────────────────────────────────────────────────────
LOSS            = "ce"           # ce | focal
SCHEDULER       = "cosine"       # cosine | cosine_wr
NO_MIXUP        = False          # True = Mixup/CutMix 끄기
SIMPLE_AUG      = False          # True = 단순 증강 (CenterCrop 없는 가벼운 증강)
MERGE_DEV       = False          # True = train+dev 1100개 합쳐서 K-Fold

# ── 하드웨어 (12GB GPU 기준) ─────────────────────────────────────────
BATCH_SIZE      = 12             # 배치 크기 (VRAM에 맞게 조절)
GRAD_ACCUM      = 2              # gradient accumulation (eff_bs = BATCH_SIZE × GRAD_ACCUM)
NUM_WORKERS     = 4              # DataLoader 워커 수
GRAD_CKPT       = False          # Gradient checkpointing (VRAM 부족 시 True)

# ── 기타 ─────────────────────────────────────────────────────────────
HEAD_TYPE       = "simple"       # simple | attn_gate
USE_VIDEO       = False

import argparse
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from train_v2 import pretrain, finetune  # noqa: E402
from models import build_model, get_backbone_config, get_backbone_choices  # noqa: E402
from inference_v2 import make_dual_transforms  # noqa: E402

DATA_DIR = os.path.join(BASE_DIR, "data")
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
PREVIEW_DIR = os.path.join(BASE_DIR, "previews")


# =====================================================================
# Grad-CAM — 모델 주목 영역 시각화
# =====================================================================
def _to_spatial(feat, backbone_key, img_size):
    """Feature map → NCHW spatial format"""
    if feat.dim() == 4:
        return feat  # ConvNeXt: already NCHW
    # ViT: (B, T, D) → (B, D, H, W)
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


def _overlay_cam(img_bgr, cam, alpha=0.45):
    """Grad-CAM heatmap을 이미지 위에 오버레이"""
    h, w = img_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(cam_resized * 255), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)


def _compute_gradcam(model, front_t, top_t, backbone_key, img_size, target_class):
    """Dual-view Grad-CAM 계산"""
    bb = model.backbone
    with torch.enable_grad():
        feat_f = bb.forward_features(front_t)
        feat_t = bb.forward_features(top_t)

        sp_f = _to_spatial(feat_f, backbone_key, img_size)
        sp_t = _to_spatial(feat_t, backbone_key, img_size)
        sp_f.retain_grad()
        sp_t.retain_grad()

        vec_f = sp_f.mean(dim=(2, 3))
        vec_t = sp_t.mean(dim=(2, 3))

        gate_weights = None
        if model.head_type == "attn_gate":
            gate = model.attn_gate(torch.cat([vec_f, vec_t], dim=1))
            fused = gate[:, 0:1] * vec_f + gate[:, 1:2] * vec_t
            gate_weights = gate[0].detach().cpu().numpy()
        else:
            fused = torch.cat([vec_f, vec_t], dim=1)

        logits = model.head(fused)
        model.zero_grad()
        logits[0, target_class].backward()

    cam_f = _build_cam(sp_f, sp_f.grad)
    cam_t = _build_cam(sp_t, sp_t.grad)
    return cam_f, cam_t, gate_weights


# =====================================================================
# Train
# =====================================================================
def cmd_train(cli_args):
    """CONFIG 값을 train_v2.finetune()가 기대하는 namespace로 변환 후 실행"""
    args = argparse.Namespace(
        # CONFIG에서 가져오기
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
        use_video=USE_VIDEO,
        num_video_frames=5,
        # CLI에서만 설정하는 것들
        fold=cli_args.fold,
        resume=cli_args.resume,
        init_from_best=cli_args.continue_best,
        # 고정값
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

    # GPU 정보
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {gpu} ({vram:.1f} GB)")

    eff = BATCH_SIZE * GRAD_ACCUM
    print(f"┌──────────────────────────────────────────────┐")
    print(f"│  {BACKBONE}")
    print(f"│  epochs={EPOCHS}  bs={BATCH_SIZE}×{GRAD_ACCUM}={eff}  fold={cli_args.fold}")
    print(f"│  head_lr={HEAD_LR:.1e}  bb_lr={BB_LR:.1e}  wd={WEIGHT_DECAY}")
    print(f"│  drop={DROP_RATE}  warmup={WARMUP_EPOCHS}  layer_decay={LAYER_DECAY}")
    print(f"│  loss={LOSS}  mixup={'OFF' if NO_MIXUP else 'ON'}  sched={SCHEDULER}")
    print(f"│  merge_dev={MERGE_DEV}  aug={'simple' if SIMPLE_AUG else 'full'}")
    if cli_args.resume:
        print(f"│  ★ RESUME: optimizer/scheduler 이어받기")
    if cli_args.continue_best:
        print(f"│  ★ CONTINUE_BEST: best 가중치 → fresh optimizer")
    print(f"└──────────────────────────────────────────────┘")

    finetune(args)


# =====================================================================
# Preview — Grad-CAM 주목 영역 + 추론 확률 시각화
# =====================================================================
def cmd_preview(cli_args):
    import pandas as pd

    backbone = BACKBONE
    fold = cli_args.fold
    cfg = get_backbone_config(backbone)
    img_size = cfg["img_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sfx = '' if SEED == 42 else f'_s{SEED}'
    ckpt_path = os.path.join(SAVE_DIR, f"{backbone}{sfx}_fold{fold}.pth")
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] 체크포인트 없음: {ckpt_path}")
        print(f"  먼저 학습: python local_train_v1.py train --fold {fold}")
        return

    model = build_model(backbone, pretrained=False, num_classes=2, drop_rate=0.0,
                        head_type=HEAD_TYPE)
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

    front_tf, top_tf = make_dual_transforms(img_size, front_crop_ratio=0.9, top_crop_ratio=0.9)

    tag = f"{backbone}_fold{fold}_{split}"
    out_dir = os.path.join(PREVIEW_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    results = []
    print(f"[Preview] {split} 데이터 {n}장 — Grad-CAM 시각화")
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

        # ── 1. 정확한 추론 확률 (no_grad) ──
        with torch.no_grad():
            logits = model(front_t, top_t)
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()[0]
        pred_class = int(logits.argmax(1).item())
        pred_label = "unstable" if pred_class == 1 else "stable"
        confidence = probs[pred_class]

        # ── 2. Grad-CAM 히트맵 ──
        try:
            cam_f, cam_t, gate = _compute_gradcam(
                model, front_t, top_t, backbone, img_size, pred_class)
        except Exception as e:
            print(f"  [WARN] Grad-CAM 실패 ({sid}): {e}")
            cam_f = cam_t = gate = None

        # ── 3. 결과 기록 ──
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

        # ── 4. 콘솔 출력 (소숫점 10자리) ──
        line = (f"  {i+1:03d} {sid}: "
                f"stable={probs[0]:.10f}  unstable={probs[1]:.10f}  "
                f"-> {pred_label}")
        if gate is not None:
            line += f"  [gate: F={gate[0]:.3f} T={gate[1]:.3f}]"
        if has_label:
            mark = "O" if correct else "X"
            line += f"  GT={true_label} {mark}"
        print(line)

        # ── 5. 이미지: [Front+CAM | Top+CAM | 정보 패널] ──
        h = 256
        panel_w = 300
        front_show = cv2.resize(front_raw, (h, h))
        top_show = cv2.resize(top_raw, (h, h))

        if cam_f is not None:
            front_show = _overlay_cam(front_show, cam_f)
            top_show = _overlay_cam(top_show, cam_t)

        # 정보 패널
        panel = np.ones((h, panel_w, 3), dtype=np.uint8) * 245

        pred_color = (0, 0, 200) if pred_label == "unstable" else (0, 160, 0)
        cv2.putText(panel, pred_label.upper(), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)
        cv2.putText(panel, f"{confidence:.1%}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)

        # 확률 바
        bar_y, bar_w = 70, panel_w - 20
        cv2.rectangle(panel, (10, bar_y), (10 + bar_w, bar_y + 16), (220, 220, 220), -1)
        cv2.rectangle(panel, (10, bar_y), (10 + int(bar_w * probs[1]), bar_y + 16),
                      (0, 0, 200), -1)

        # 확률 (10자리)
        cv2.putText(panel, f"unstable {probs[1]:.10f}", (10, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 180), 1)
        cv2.putText(panel, f"stable   {probs[0]:.10f}", (10, 126),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 140, 0), 1)

        # 게이트 가중치 (attn_gate)
        y_cur = 150
        if gate is not None:
            cv2.putText(panel, f"gate: Front={gate[0]:.4f} Top={gate[1]:.4f}",
                        (10, y_cur), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (100, 50, 0), 1)
            y_cur += 22

        # 정답 라벨
        if has_label:
            gt_color = (0, 160, 0) if correct else (0, 0, 200)
            cv2.putText(panel, f"GT: {true_label}", (10, y_cur),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, gt_color, 2)
            y_cur += 25
            cv2.putText(panel, "CORRECT" if correct else "WRONG", (10, y_cur),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, gt_color, 2)

        cv2.putText(panel, sid, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        # 캔버스 조합
        canvas = np.concatenate([front_show, top_show, panel], axis=1)
        border_color = (200, 100, 0) if pred_label == "unstable" else (100, 100, 100)
        if has_label:
            border_color = (0, 180, 0) if correct else (0, 0, 220)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1]-1, canvas.shape[0]-1), border_color, 3)
        cv2.putText(canvas, "FRONT + CAM", (5, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(canvas, "TOP + CAM", (h + 5, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imwrite(os.path.join(out_dir, f"{i+1:03d}_{sid}_{pred_label}.png"), canvas)

    # ── 요약 ──
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
# CLI — 명령어는 최소한만
# =====================================================================
def main():
    p = argparse.ArgumentParser(
        description="로컬 학습 & 시각화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용법:
  python local_train_v1.py train --fold 0             # 학습
  python local_train_v1.py train --fold 0 --resume    # 이어서 학습 (optimizer 포함)
  python local_train_v1.py train --fold 0 --continue_best  # best → 추가 학습
  python local_train_v1.py preview --fold 0            # test 15장 시각화
  python local_train_v1.py preview --fold 0 --split dev    # dev 정답 비교

설정 변경은 파일 상단 CONFIG 섹션 직접 수정.
""",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # train: fold, resume, continue_best만
    t = sub.add_parser("train", help="학습 (설정은 파일 상단 CONFIG)")
    t.add_argument("--fold", type=int, default=None, help="fold 번호 (미지정=전체)")
    t.add_argument("--resume", action="store_true", help="optimizer/scheduler 이어받기")
    t.add_argument("--continue_best", action="store_true", help="best 가중치 → fresh optimizer")

    # preview: fold, split, n, random만
    v = sub.add_parser("preview", help="예측 시각화 (이미지 저장)")
    v.add_argument("--fold", type=int, required=True, help="fold 번호")
    v.add_argument("--split", default="test", choices=["test", "dev", "train"])
    v.add_argument("-n", type=int, default=15, help="샘플 수 (기본: 15)")
    v.add_argument("--random", action="store_true", help="랜덤 샘플링")

    args = p.parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "preview":
        cmd_preview(args)


if __name__ == "__main__":
    main()
