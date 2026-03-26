"""
local_train_v1.py — 로컬 소규모 모델 학습 스크립트 (Windows + 12GB GPU)

사용 대상 백본 (로컬 tier):
  - convnext_small  : 50M,  384px, CNN 기반 — 빠르고 가벼움 (추천)
  - eva02_large     : 305M, 448px, ViT     — 높은 정밀도, VRAM 많이 사용
  - dinov2_large    : 304M, 336px, ViT     — 공간 구조 특화

핵심 차이점 (train_v2.py 대비):
  - 로컬 12GB GPU에 맞는 기본 배치 크기 / grad_accum
  - cudnn.benchmark=True + matmul precision=high 자동 적용
  - Windows multiprocessing 안전 가드 (if __name__ == '__main__')
  - 심볼릭 링크 없는 단순 체크포인트 저장 구조

Usage:
  python local_train_v1.py --backbone convnext_small --stage finetune --fold 0
  python local_train_v1.py --backbone convnext_small --stage finetune --merge_dev --no_mixup --simple_aug --resume
  python local_train_v1.py --backbone eva02_large --stage finetune --fold 0 --layer_decay 0.8 --warmup_epochs 2
  python local_train_v1.py --backbone dinov2_large --stage finetune --fold 0 --init_from_best --resume
"""

import argparse
import os
import sys

import torch

# ── 로컬 GPU 최적화 ──────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True       # CNN 속도 향상 (ConvNeXt 등)
    torch.set_float32_matmul_precision("high")  # Tensor Core 활용 (Ampere+)
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[LocalTrain] GPU: {gpu_name} ({vram_gb:.1f} GB)")
    if vram_gb < 10:
        print("[WARN] VRAM < 10GB — 배치 크기를 더 낮추거나 grad_accum을 늘리세요")
else:
    print("[LocalTrain] GPU 없음 — CPU 모드 (학습 매우 느림)")

# ── train_v2 핵심 함수 재사용 ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_v2 import pretrain, finetune  # noqa: E402
from models import get_backbone_choices  # noqa: E402

# ── 로컬 12GB GPU 기본 배치/누적 설정 ────────────────────────────────────────
# effective_bs = batch_size × grad_accum
# VRAM 12GB 기준 보수적 설정; 여유 있으면 batch_size를 늘리세요
_LOCAL_BATCH = {
    "convnext_small": 24,   # eff_bs=48  (가벼워서 넉넉함)
    "eva02_large":     2,   # eff_bs=32  (448px + ViT → VRAM 빠듯)
    "dinov2_large":    4,   # eff_bs=32  (336px + ViT → 적당)
    "siglip_so400m":   2,   # eff_bs=4   (A100 tier, 로컬은 비추)
    "eva_giant":       1,   # eff_bs=4   (사실상 로컬 학습 불가)
}
_LOCAL_ACCUM = {
    "convnext_small": 2,
    "eva02_large":    16,
    "dinov2_large":   8,
    "siglip_so400m":  2,
    "eva_giant":      4,
}


def main():
    p = argparse.ArgumentParser(
        description="로컬 소규모 모델 학습 (Windows 12GB GPU 최적화)"
    )
    # ── 백본 / 스테이지 ──────────────────────────────────────────────────────
    p.add_argument("--backbone", type=str, default="convnext_small",
                   choices=get_backbone_choices(),
                   help="기본값: convnext_small (로컬에서 가장 빠름)")
    p.add_argument("--stage", type=str, default="finetune",
                   choices=["pretrain", "finetune", "both"])
    p.add_argument("--pretrain_epochs", type=int, default=15)
    p.add_argument("--finetune_epochs", type=int, default=30)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--fold", type=int, default=None,
                   help="특정 fold만 실행 (None이면 전체)")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    # ── 하드웨어 설정 ────────────────────────────────────────────────────────
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader 워커 수 (로컬: 4, OOM 시 2로 낮추기)")
    p.add_argument("--batch_size_override", type=int, default=None,
                   help="None이면 로컬 기본값 자동 적용")
    p.add_argument("--grad_accum_override", type=int, default=None,
                   help="None이면 로컬 기본값 자동 적용")
    p.add_argument("--grad_checkpointing", action="store_true",
                   help="VRAM 부족 시 ON (속도 ~20%% 감소)")

    # ── 데이터 전략 ──────────────────────────────────────────────────────────
    p.add_argument("--use_video", action="store_true")
    p.add_argument("--num_video_frames", type=int, default=5)
    p.add_argument("--include_dev", action="store_true")
    p.add_argument("--include_dev_aug", action="store_true")
    p.add_argument("--dev_aug_repeat", type=int, default=3)
    p.add_argument("--merge_dev", action="store_true",
                   help="train+dev 합쳐서 K-Fold (추천)")

    # ── 학습 하이퍼파라미터 ───────────────────────────────────────────────────
    p.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"])
    p.add_argument("--no_mixup", action="store_true")
    p.add_argument("--head_lr", type=float, default=None)
    p.add_argument("--bb_lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--layer_decay", type=float, default=None,
                   help="ViT용 LLRD (eva02_large: 0.8, dinov2_large: 0.8)")
    p.add_argument("--warmup_epochs", type=int, default=0)
    p.add_argument("--scheduler", type=str, default="cosine",
                   choices=["cosine", "cosine_wr"])
    p.add_argument("--head_type", type=str, default="simple",
                   choices=["attn_gate", "simple"])
    p.add_argument("--drop_rate", type=float, default=None)
    p.add_argument("--simple_aug", action="store_true")
    p.add_argument("--video_frame_aug", action="store_true")
    p.add_argument("--no_dacon_pretrain", action="store_true")

    # ── 체크포인트 ───────────────────────────────────────────────────────────
    p.add_argument("--init_from_best", action="store_true",
                   help="기존 fold best 가중치에서 이어서 학습 (fresh optimizer)")
    p.add_argument("--resume", action="store_true",
                   help="Resume ckpt 있으면 이어서 학습")
    p.add_argument("--skip_completed", action="store_true",
                   help="best 모델 있고 ckpt 없는 fold 건너뛰기")

    args = p.parse_args()

    # ── 로컬 배치/누적 기본값 적용 ────────────────────────────────────────────
    if args.batch_size_override is None:
        args.batch_size_override = _LOCAL_BATCH.get(args.backbone, 4)
        print(f"[LocalTrain] batch_size 자동 설정: {args.batch_size_override} "
              f"(--batch_size_override 로 덮어쓰기 가능)")

    if args.grad_accum_override is None:
        args.grad_accum_override = _LOCAL_ACCUM.get(args.backbone, 8)
        print(f"[LocalTrain] grad_accum 자동 설정: {args.grad_accum_override}")

    eff_bs = args.batch_size_override * args.grad_accum_override
    print(f"[LocalTrain] 유효 배치 크기: {args.batch_size_override} × "
          f"{args.grad_accum_override} = {eff_bs}")

    # ── 실행 ─────────────────────────────────────────────────────────────────
    if args.stage in ("pretrain", "both"):
        pretrain(args)
    if args.stage in ("finetune", "both"):
        finetune(args)


# Windows multiprocessing 안전 가드 — 반드시 필요 (DataLoader spawn)
if __name__ == "__main__":
    main()
