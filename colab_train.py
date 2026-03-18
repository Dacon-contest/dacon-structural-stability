"""
Google Colab A100 학습 스크립트
=================================
이 파일을 Colab에 업로드한 뒤 셀 단위로 실행하세요.

필요 파일 (Google Drive 또는 직접 업로드):
  - data.zip  (대회 data 폴더 압축)
  - models.py, datasets.py, train.py, inference.py

셀 실행 순서:
  1. setup_environment()      — 패키지 설치
  2. extract_data()           — zip 추출
  3. prepare_shapestacks()    — ShapeStacks 다운로드 & h=6 필터
  4. run_pretrain()           — Stage 1
  5. run_finetune()           — Stage 2 (5-Fold)
  6. run_inference()          — 제출 파일 생성
"""
import os
import subprocess
import sys

WORK_DIR = "/content/dacon-structural-stability"
DATA_DIR = os.path.join(WORK_DIR, "data")


# =========================================================================
# Cell 1: 환경 설정
# =========================================================================
def setup_environment():
    """패키지 설치 + 작업 디렉토리 구성"""
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "timm>=1.0.20", "albumentations>=1.4.0",
        "opencv-python-headless", "scikit-learn", "pandas", "tqdm",
    ])
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(WORK_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(WORK_DIR, "submissions"), exist_ok=True)

    # 소스 파일 복사 (Colab /content/ 에 업로드한 경우)
    for f in ["models.py", "datasets.py", "train.py", "inference.py"]:
        src = os.path.join("/content", f)
        dst = os.path.join(WORK_DIR, f)
        if os.path.exists(src) and not os.path.exists(dst):
            import shutil
            shutil.copy2(src, dst)
            print(f"  Copied {f} → {WORK_DIR}")

    print("Environment setup complete.")
    print(f"Working directory: {WORK_DIR}")

    # GPU 확인
    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected!")


# =========================================================================
# Cell 2: 데이터 추출
# =========================================================================
def extract_data(zip_path="/content/data.zip"):
    """대회 데이터 zip 추출
    zip 내부 구조가 data/ 폴더와 동일해야 합니다:
      open/train.csv, open/dev.csv, open/sample_submission.csv
      open/train/TRAIN_xxx/, open/dev/DEV_xxx/, open/test/TEST_xxxx/
    """
    import zipfile

    if not os.path.exists(zip_path):
        print(f"[ERROR] {zip_path} not found!")
        print("Google Drive에서 마운트하거나 직접 업로드하세요:")
        print("  from google.colab import files")
        print("  files.upload()")
        return

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # zip 내부에 data/ 폴더가 있는지 확인
        names = zf.namelist()
        has_data_prefix = any(n.startswith("data/") for n in names)

        if has_data_prefix:
            zf.extractall(WORK_DIR)
            print(f"  Extracted to {WORK_DIR}/data/")
        else:
            # data/ 접두사 없으면 data/ 폴더에 직접 추출
            zf.extractall(DATA_DIR)
            print(f"  Extracted to {DATA_DIR}/")

    # 검증
    for f in ["open/train.csv", "open/dev.csv", "open/sample_submission.csv"]:
        p = os.path.join(DATA_DIR, f)
        if os.path.exists(p):
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} NOT FOUND")


# =========================================================================
# Cell 3: ShapeStacks 준비
# =========================================================================
def prepare_shapestacks(zip_path=None):
    """ShapeStacks 데이터 준비
    옵션 1: zip_path 지정 → 추출 후 h=6 필터
    옵션 2: 이미 data/shapestacks/ 에 있으면 필터만 수행
    옵션 3: ShapeStacks 없이 진행 (pretrain 스킵)
    """
    ss_dir = os.path.join(DATA_DIR, "shapestacks", "shapestacks", "recordings")

    if zip_path and os.path.exists(zip_path):
        import zipfile
        print(f"Extracting ShapeStacks from {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(os.path.join(DATA_DIR, "shapestacks"))
        print("  Done.")

    if not os.path.exists(ss_dir):
        print("[INFO] ShapeStacks not found. Pretrain will use Dacon data only.")
        return

    # h=6 통계
    total, h6, h6_stable, h6_unstable = 0, 0, 0, 0
    for name in os.listdir(ss_dir):
        if not os.path.isdir(os.path.join(ss_dir, name)):
            continue
        total += 1
        if "h=6" in name:
            h6 += 1
            is_unstable = False
            for p in name.split("-"):
                if p.startswith("vcom=") or p.startswith("vpsf="):
                    if int(p.split("=")[1]) > 0:
                        is_unstable = True
                        break
            if is_unstable:
                h6_unstable += 1
            else:
                h6_stable += 1

    print(f"  Total scenarios: {total}")
    print(f"  h=6 scenarios: {h6} (stable={h6_stable}, unstable={h6_unstable})")
    print(f"  학습에는 h=6만 사용합니다 (코드에서 자동 필터링)")


# =========================================================================
# Cell 4: Pretrain
# =========================================================================
def run_pretrain(backbone="eva_giant", epochs=15, extra_args=""):
    """Stage 1: ShapeStacks h=6 + Dacon pretrain"""
    cmd = (f"cd {WORK_DIR} && {sys.executable} train.py "
           f"--backbone {backbone} --stage pretrain "
           f"--pretrain_epochs {epochs} --grad_checkpointing "
           f"--resume {extra_args}")
    print(f"Running: {cmd}")
    os.system(cmd)


# =========================================================================
# Cell 5: Finetune
# =========================================================================
def run_finetune(backbone="eva_giant", epochs=50, fold=None, include_dev=True, extra_args=""):
    """Stage 2: 5-Fold CV finetune"""
    cmd = (f"cd {WORK_DIR} && {sys.executable} train.py "
           f"--backbone {backbone} --stage finetune "
           f"--finetune_epochs {epochs} --grad_checkpointing "
           f"--resume")
    if include_dev:
        cmd += " --include_dev"
    if fold is not None:
        cmd += f" --fold {fold}"
    cmd += f" {extra_args}"
    print(f"Running: {cmd}")
    os.system(cmd)


# =========================================================================
# Cell 6: Inference
# =========================================================================
def run_inference(backbones=None, tta=True, temperature=1.0):
    """추론 + submission 생성"""
    if backbones is None:
        backbones = ["eva_giant"]
    bk_str = " ".join(backbones)
    cmd = (f"cd {WORK_DIR} && {sys.executable} inference.py "
           f"--backbones {bk_str}")
    if tta:
        cmd += " --tta"
    if temperature != 1.0:
        cmd += f" --temperature {temperature}"
    print(f"Running: {cmd}")
    os.system(cmd)


# =========================================================================
# Full Pipeline (한번에 실행)
# =========================================================================
def full_pipeline(backbone="eva_giant", pretrain_epochs=15, finetune_epochs=50):
    """전체 파이프라인: pretrain → finetune (5-fold) → inference"""
    print("=" * 60)
    print(f"  Full Pipeline: {backbone}")
    print(f"  Pretrain: {pretrain_epochs} ep → Finetune: {finetune_epochs} ep × 5 folds")
    print("=" * 60)

    run_pretrain(backbone, pretrain_epochs)
    run_finetune(backbone, finetune_epochs, include_dev=True)
    run_inference([backbone], tta=True)

    print("\n  Pipeline complete!")
    print(f"  제출: {WORK_DIR}/submissions/")


# =========================================================================
# Multi-backbone Ensemble Pipeline
# =========================================================================
def multi_backbone_pipeline():
    """다중 백본 앙상블 파이프라인 (A100 권장)"""
    backbones = ["eva_giant", "dinov3_huge", "siglip_so400m"]

    for bk in backbones:
        print(f"\n{'='*60}")
        print(f"  Training: {bk}")
        print(f"{'='*60}")
        run_pretrain(bk, epochs=10)
        run_finetune(bk, epochs=40, include_dev=True)

    # Final ensemble
    run_inference(backbones, tta=True)
    print("\n  Multi-backbone ensemble complete!")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--setup", action="store_true")
    p.add_argument("--extract", type=str, default=None, help="data.zip path")
    p.add_argument("--shapestacks", type=str, default=None, help="shapestacks.zip path")
    p.add_argument("--train", action="store_true")
    p.add_argument("--inference", action="store_true")
    p.add_argument("--backbone", type=str, default="eva_giant")
    p.add_argument("--full", action="store_true", help="Full pipeline")
    p.add_argument("--multi", action="store_true", help="Multi-backbone ensemble")
    args = p.parse_args()

    if args.setup:
        setup_environment()
    if args.extract:
        extract_data(args.extract)
    if args.shapestacks:
        prepare_shapestacks(args.shapestacks)
    if args.full:
        full_pipeline(args.backbone)
    elif args.multi:
        multi_backbone_pipeline()
    elif args.train:
        run_pretrain(args.backbone)
        run_finetune(args.backbone, include_dev=True)
    elif args.inference:
        run_inference([args.backbone], tta=True)
