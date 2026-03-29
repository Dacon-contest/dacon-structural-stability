"""Generate colab_train11.ipynb — 2-Stage Pipeline: DINOv2 Masking → EVA Classification"""
import json, os

cells = []

def add_code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.split("\n") if isinstance(source, str) else source
    })

# Fix source format: join lines with \n but keep as list of strings with \n
def add_cell(source_text):
    lines = source_text.split("\n")
    source = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        source.append(lines[-1])
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    })

# ══════════════════════════════════════════════════════════════
# Cell 1: 환경 설정
# ══════════════════════════════════════════════════════════════
add_cell(r'''# ═══════════════════════════════════════════════════════════════════
# 1. 환경 설정 (Drive 마운트 + 패키지 설치 + 캐시를 Drive로)
# ═══════════════════════════════════════════════════════════════════
# colab_train11: 2-Stage Pipeline (DINOv2 Masking → EVA Classification)
# Stage 1: DINOv2 Attention으로 배경 제거 (체커보드/하늘)
# Stage 2: 배경 없는 순수 구조물만 EVA로 분류
# ═══════════════════════════════════════════════════════════════════
import os, subprocess, sys, shutil

from google.colab import drive
drive.mount('/content/drive')

DRIVE_DIR  = "/content/drive/MyDrive/dacon"
WORK_DIR   = "/content/dacon"
LOCAL_DATA = os.path.join(WORK_DIR, "data")
CACHE_DIR  = os.path.join(DRIVE_DIR, ".cache")
LOCAL_CKPT = "/content/dacon/local_ckpt"

for d in [
    DRIVE_DIR,
    os.path.join(DRIVE_DIR, "checkpoints"),
    os.path.join(DRIVE_DIR, "submissions"),
    CACHE_DIR, WORK_DIR, LOCAL_CKPT,
]:
    os.makedirs(d, exist_ok=True)

os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "huggingface")
os.environ["TORCH_HOME"] = os.path.join(CACHE_DIR, "torch")
os.environ["XDG_CACHE_HOME"] = CACHE_DIR
os.environ["PIP_CACHE_DIR"] = os.path.join(CACHE_DIR, "pip")
os.environ["TMPDIR"] = os.path.join(CACHE_DIR, "tmp")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)
os.environ["DACON_LOCAL_CKPT"] = LOCAL_CKPT

import torch
if torch.cuda.is_available():
    gpu  = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu} ({vram:.1f} GB)")
else:
    print("GPU 없음! 런타임 > 런타임 유형 변경 > GPU 선택")

free = shutil.disk_usage('/content').free / 1e9
print(f"로컬 디스크 여유: {free:.1f} GB")

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "timm>=1.0.20", "albumentations>=1.4.0",
    "opencv-python-headless", "scikit-learn", "pandas", "tqdm",
])
print("패키지 설치 완료")''')

# ══════════════════════════════════════════════════════════════
# Cell 2: 소스 복사 + 체크포인트 (v10 EVA 백업 + 초기화)
# ══════════════════════════════════════════════════════════════
add_cell(r'''# ═══════════════════════════════════════════════════════════════════
# 2. 소스 파일 복사 + v10 EVA 체크포인트 백업 (새 학습 준비)
# ═══════════════════════════════════════════════════════════════════
import os, shutil

DRIVE_DIR = "/content/drive/MyDrive/dacon"
WORK_DIR  = "/content/dacon"
DRIVE_CKPT = os.path.join(DRIVE_DIR, "checkpoints")
LOCAL_CKPT_BEST = os.path.join(WORK_DIR, "checkpoints")
DRIVE_SUB  = os.path.join(DRIVE_DIR, "submissions")

# --- 소스 파일 복사 ---
SRC_FILES = ["models.py", "datasets.py", "train.py", "train_v2.py",
             "inference.py", "inference_v2.py"]
for f in SRC_FILES:
    src = os.path.join(DRIVE_DIR, f)
    dst = os.path.join(WORK_DIR, f)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  [OK] {f}")
    else:
        print(f"  [MISSING] {f} - Drive에 업로드 필요: {DRIVE_DIR}/")

# --- 체크포인트 디렉토리 ---
os.makedirs(LOCAL_CKPT_BEST, exist_ok=True)

# --- submissions → Drive symlink ---
sub_link = os.path.join(WORK_DIR, "submissions")
if os.path.islink(sub_link):
    os.unlink(sub_link)
elif os.path.isdir(sub_link):
    shutil.rmtree(sub_link)
os.symlink(DRIVE_SUB, sub_link)
print(f"  submissions/ -> Drive")

# ═══════════════════════════════════════════════════════════════════
# v10 EVA 체크포인트 백업 + 로컬 초기화
# (마스킹 이미지로 처음부터 재학습하므로 기존 가중치 사용 불가)
# ═══════════════════════════════════════════════════════════════════
backup_dir = os.path.join(DRIVE_CKPT, "v10_backup")
os.makedirs(backup_dir, exist_ok=True)
backed_up = 0
for f in sorted(os.listdir(DRIVE_CKPT)):
    if f.startswith("eva_giant") and f.endswith(".pth"):
        src = os.path.join(DRIVE_CKPT, f)
        dst = os.path.join(backup_dir, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            backed_up += 1
            print(f"  v10 백업: {f} → v10_backup/")
if backed_up > 0:
    print(f"  {backed_up}개 v10 EVA 가중치 백업 완료")
else:
    print("  v10 EVA 가중치 백업 완료 (이미 존재)")

# 로컬 EVA 체크포인트 삭제 (새 학습용)
cleared = 0
for f in os.listdir(LOCAL_CKPT_BEST):
    if f.startswith("eva_giant") and (f.endswith(".pth") or f.endswith(".csv")):
        os.remove(os.path.join(LOCAL_CKPT_BEST, f))
        cleared += 1
if cleared > 0:
    print(f"  로컬 EVA 체크포인트 {cleared}개 삭제 (처음부터 재학습)")
else:
    print("  로컬 EVA 체크포인트 없음 (깨끗한 상태)")

# --- Drive 동기화 헬퍼 ---
def sync_best_to_drive(backbone=None):
    count = 0
    for f in sorted(os.listdir(LOCAL_CKPT_BEST)):
        if not f.endswith('.pth') or '_ckpt' in f:
            continue
        if backbone and not f.startswith(backbone):
            continue
        src = os.path.join(LOCAL_CKPT_BEST, f)
        dst = os.path.join(DRIVE_CKPT, f)
        src_size = os.path.getsize(src)
        if os.path.exists(dst) and os.path.getsize(dst) == src_size:
            continue
        mb = src_size / 1e6
        print(f"  -> Drive: {f} ({mb:.0f} MB)")
        shutil.copy2(src, dst)
        count += 1
    if count == 0:
        print("  동기화할 새 가중치 없음")
    else:
        print(f"  {count}개 가중치 Drive 동기화 완료")

print(f"\n준비 완료: EVA를 마스킹 이미지로 처음부터 학습합니다.")''')

# ══════════════════════════════════════════════════════════════
# Cell 3: 패치 (train_v2.py 검증 + datasets.py 증강)
# ══════════════════════════════════════════════════════════════
cell3_code = (
    '# ═══════════════════════════════════════════════════════════════════\n'
    '# 3. 코랩 패치 (train_v2.py 검증 + datasets.py TTA 확장)\n'
    '# ═══════════════════════════════════════════════════════════════════\n'
    'from pathlib import Path\n'
    '\n'
    '# 1) train_v2.py 검증\n'
    'train_v2 = Path("/content/dacon/train_v2.py")\n'
    'if not train_v2.exists():\n'
    '    raise FileNotFoundError(\n'
    '        "train_v2.py가 없습니다! Drive에 업로드하세요.\\n"\n'
    '        f"경로: {DRIVE_DIR}/train_v2.py"\n'
    '    )\n'
    'tv2_text = train_v2.read_text(encoding="utf-8")\n'
    'missing = []\n'
    'for required in ["--layer_decay", "--warmup_epochs", "cross_attn"]:\n'
    '    if required not in tv2_text:\n'
    '        missing.append(required)\n'
    'if missing:\n'
    '    raise RuntimeError(f"train_v2.py에 필수 기능 누락: {missing}")\n'
    'print("✓ train_v2.py 검증 완료")\n'
    '\n'
    'print("\\n패치 요약:")\n'
    'print("  1. train_v2.py 필수 인자 검증 완료")\n'
)
add_cell(cell3_code)

# ══════════════════════════════════════════════════════════════
# Cell 4: 데이터 추출
# ══════════════════════════════════════════════════════════════
add_cell(r'''# ═══════════════════════════════════════════════════════════════════
# 4. 데이터 압축 해제 (Drive data.zip → 로컬 SSD)
# ═══════════════════════════════════════════════════════════════════
import os, shutil, zipfile, time

DRIVE_DIR  = "/content/drive/MyDrive/dacon"
WORK_DIR   = "/content/dacon"
LOCAL_DATA = os.path.join(WORK_DIR, "data")
ZIP_PATH   = os.path.join(DRIVE_DIR, "data.zip")

if os.path.islink(LOCAL_DATA):
    os.unlink(LOCAL_DATA)
elif os.path.isdir(LOCAL_DATA):
    shutil.rmtree(LOCAL_DATA)
os.makedirs(LOCAL_DATA, exist_ok=True)

if not os.path.exists(ZIP_PATH):
    print(f"[ERROR] {ZIP_PATH} 를 찾을 수 없습니다.")
else:
    zip_mb = os.path.getsize(ZIP_PATH) / 1e6
    print(f"data.zip: {zip_mb:.0f} MB → 해제 중...")
    t0 = time.time()
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(LOCAL_DATA)
    print(f"  해제 완료: {time.time() - t0:.1f}초")

for f in ["open/train.csv", "open/dev.csv", "open/sample_submission.csv"]:
    p = os.path.join(LOCAL_DATA, f)
    ok = "[OK]" if os.path.exists(p) else "[FAIL]"
    print(f"  {ok} {f}")

for split in ["train", "dev", "test"]:
    d = os.path.join(LOCAL_DATA, "open", split)
    if os.path.isdir(d):
        n = len([x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))])
        print(f"  {split}/: {n}개 샘플")

free = shutil.disk_usage('/content').free / 1e9
print(f"\n로컬 디스크 여유: {free:.1f} GB")''')

# ══════════════════════════════════════════════════════════════
# Cell 5: 환경 검증
# ══════════════════════════════════════════════════════════════
add_cell(r'''# ═══════════════════════════════════════════════════════════════════
# 5. 환경 검증 (v11: 2-Stage Pipeline)
# ═══════════════════════════════════════════════════════════════════
import os, shutil
WORK_DIR = "/content/dacon"

print("=" * 60)
print("  v11 환경 검증 (DINOv2 Masking → EVA Classification)")
print("=" * 60)

for f in ["models.py", "datasets.py", "train_v2.py", "inference_v2.py"]:
    ok = os.path.exists(os.path.join(WORK_DIR, f))
    print(f"  {'[OK]' if ok else '[FAIL]'} {f}")

for f in ["data/open/train.csv", "data/open/dev.csv"]:
    ok = os.path.exists(os.path.join(WORK_DIR, f))
    print(f"  {'[OK]' if ok else '[FAIL]'} {f}")

ckpt_dir = os.path.join(WORK_DIR, "checkpoints")
eva_ckpts = [f for f in os.listdir(ckpt_dir)
             if f.startswith("eva_giant") and f.endswith('.pth')]
print(f"\n  EVA 기존 체크포인트: {len(eva_ckpts)}개 (0이어야 정상)")
if eva_ckpts:
    print("  [WARN] 기존 EVA 체크포인트 발견! Cell 2에서 삭제 필요")
else:
    print("  [OK] 깨끗 — 처음부터 학습 준비 완료")

free = shutil.disk_usage('/content').free / 1e9
print(f"\n  디스크 여유: {free:.1f} GB")
print("=" * 60)''')

# ══════════════════════════════════════════════════════════════
# Cell 6: ★ DINOv2 Attention Masking Engine (THE CORE)
# ══════════════════════════════════════════════════════════════
add_cell(r'''# ═══════════════════════════════════════════════════════════════════
# 6. ★ DINOv2 Attention Masking Engine — 배경 제거
# ═══════════════════════════════════════════════════════════════════
# 원리:
#   1) DINOv2의 CLS 토큰이 패치 토큰에 보내는 attention = "어디가 중요한가"
#   2) self-supervised 학습된 DINOv2는 전경 객체에 자연스럽게 집중
#   3) attention이 높은 영역 = 구조물, 낮은 영역 = 배경(체커보드/하늘)
#   4) threshold → binary mask → 배경을 검은색으로 제거
#   5) EVA는 순수한 구조물만 보고 안정성 판단
#
# v10 Cell 19 attention rollout 오류 수정:
#   - DINOv2-reg4는 CLS(1) + Register(4) + Patch(576) = 581 토큰
#   - CLS→patch attention 추출 시 register 토큰을 반드시 건너뛰어야 함
#   - v10은 result[0, 0, 1:]로 register 포함 → reshape(24,24) 실패 (580≠576)
#   - v11은 result[0, 0, n_prefix:]로 register 건너뜀 → 정확히 576개
# ═══════════════════════════════════════════════════════════════════
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
import cv2
from PIL import Image
from tqdm import tqdm

os.chdir("/content/dacon")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────
# 1) DINOv2 로드 (pretrained, 학습 불필요 — 마스킹 전용)
# ────────────────────────────────────────────────────────────
DINO_NAME = "vit_large_patch14_reg4_dinov2.lvd142m"
MASK_IMG_SIZE = 336          # 336/14 = 24 (깔끔한 패치 그리드)
MASK_PATCH_SIZE = 14
MASK_GRID = MASK_IMG_SIZE // MASK_PATCH_SIZE  # 24
N_ATTN_LAYERS = 4            # 마지막 4개 레이어 평균 (more robust)
MASK_DILATE_ITER = 2          # 마스크 팽창 (구조물 경계 포함)
MASK_KERNEL_SIZE = 7
BG_FILL = (0, 0, 0)          # 배경 → 검은색

print(f"Loading {DINO_NAME} for attention masking...")
dino = timm.create_model(DINO_NAME, pretrained=True)
dino = dino.to(device).eval()

# Register token 수 확인 (DINOv2-reg4: CLS=1 + REG=4 = 5)
n_prefix = getattr(dino, 'num_prefix_tokens', 5)
n_patches = MASK_GRID * MASK_GRID
print(f"  Prefix tokens: {n_prefix} (CLS + {n_prefix-1} registers)")
print(f"  Patch grid: {MASK_GRID}x{MASK_GRID} = {n_patches} patches")
print(f"  Attention layers: last {N_ATTN_LAYERS}")
print(f"  Mask dilation: {MASK_DILATE_ITER} iterations, kernel={MASK_KERNEL_SIZE}")

# ────────────────────────────────────────────────────────────
# 2) Attention 추출 함수
# ────────────────────────────────────────────────────────────
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 전체 이미지를 보기 위해 crop 없이 resize만
mask_tf = A.Compose([
    A.Resize(MASK_IMG_SIZE, MASK_IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def get_dino_mask(model, img_rgb, orig_h, orig_w):
    """
    DINOv2 마지막 N 레이어의 CLS→patch attention으로 binary mask 생성.
    
    Args:
        model: DINOv2 모델
        img_rgb: (H, W, 3) RGB uint8 numpy array
        orig_h, orig_w: 원본 이미지 크기
    
    Returns:
        mask: (orig_h, orig_w) binary mask (1=구조물, 0=배경)
        heatmap: (orig_h, orig_w) attention heatmap [0,1]
    """
    tensor = mask_tf(image=img_rgb)["image"].unsqueeze(0).to(device)
    
    # Hook: 마지막 N 레이어에서 CLS→patch attention 추출
    cls_attns = []
    hooks = []
    target_blocks = model.blocks[-N_ATTN_LAYERS:]
    
    for blk in target_blocks:
        def _make_hook():
            def _hook(module, inputs, output):
                x = inputs[0]
                B, N, C = x.shape
                num_heads = module.num_heads
                head_dim = C // num_heads
                
                qkv = module.qkv(x)  # (B, N, 3*C)
                qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, hd)
                q, k, _ = qkv.unbind(0)
                
                attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
                attn = attn.softmax(dim=-1)  # (B, heads, N, N)
                
                # CLS(idx=0) → patch tokens (skip prefix)
                cls_patch = attn[0, :, 0, n_prefix:]  # (heads, n_patches)
                cls_attns.append(cls_patch.mean(dim=0).detach().cpu())
            return _hook
        hooks.append(blk.attn.register_forward_hook(_make_hook()))
    
    with torch.no_grad(), torch.amp.autocast("cuda"):
        model(tensor)
    
    for h in hooks:
        h.remove()
    
    # 레이어 평균 → spatial heatmap
    avg_attn = torch.stack(cls_attns).mean(dim=0).numpy()  # (n_patches,)
    heatmap = avg_attn.reshape(MASK_GRID, MASK_GRID)
    
    # Normalize [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Upsample to original image resolution
    heatmap_full = cv2.resize(heatmap.astype(np.float32),
                              (orig_w, orig_h),
                              interpolation=cv2.INTER_CUBIC)
    heatmap_full = np.clip(heatmap_full, 0, 1)
    
    # Otsu threshold (자동으로 전경/배경 분리)
    heatmap_u8 = (heatmap_full * 255).astype(np.uint8)
    _, mask = cv2.threshold(heatmap_u8, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (mask > 0).astype(np.uint8)
    
    # Morphological post-processing
    kernel = np.ones((MASK_KERNEL_SIZE, MASK_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # 구멍 메우기
    mask = cv2.dilate(mask, kernel, iterations=MASK_DILATE_ITER)  # 경계 확장
    
    return mask, heatmap_full

# ────────────────────────────────────────────────────────────
# 3) 전체 데이터셋 마스킹
# ────────────────────────────────────────────────────────────
train_df = pd.read_csv("data/open/train.csv")
dev_df = pd.read_csv("data/open/dev.csv")
sub_df = pd.read_csv("data/open/sample_submission.csv")

datasets_to_mask = [
    ("train", "data/open/train", train_df["id"].tolist()),
    ("dev",   "data/open/dev",   dev_df["id"].tolist()),
    ("test",  "data/open/test",  sub_df["id"].tolist()),
]

total_images = sum(len(ids) * 2 for _, _, ids in datasets_to_mask)
print(f"\n총 {total_images}개 이미지 마스킹 시작...")
print(f"  배경 → 검은색 ({BG_FILL})")

fg_ratios = []
t0 = time.time()

for split_name, data_dir, sample_ids in datasets_to_mask:
    for sid in tqdm(sample_ids, desc=f"  {split_name}"):
        for view in ["front", "top"]:
            img_path = os.path.join(data_dir, sid, f"{view}.png")
            
            # 원본 로드
            img = np.array(Image.open(img_path))  # RGB
            h, w = img.shape[:2]
            
            # 마스크 생성
            mask, _ = get_dino_mask(dino, img, h, w)
            
            # 마스크 적용: 배경 → 검은색
            masked = img.copy()
            masked[mask == 0] = BG_FILL
            
            # 덮어쓰기 저장 (원본은 data.zip에 보존)
            Image.fromarray(masked).save(img_path)
            
            fg_ratios.append(mask.sum() / mask.size)

elapsed = time.time() - t0
fg_arr = np.array(fg_ratios)

print(f"\n{'='*60}")
print(f"  마스킹 완료! ({elapsed:.0f}초)")
print(f"{'='*60}")
print(f"  처리: {total_images}개 이미지")
print(f"  전경 비율: 평균 {fg_arr.mean():.1%}, "
      f"최소 {fg_arr.min():.1%}, 최대 {fg_arr.max():.1%}")

# Sanity check
if fg_arr.mean() < 0.05:
    print("  [WARN] 전경 비율이 너무 낮음! 마스크 품질 확인 필요")
elif fg_arr.mean() > 0.80:
    print("  [WARN] 전경 비율이 너무 높음! 배경 제거가 불충분할 수 있음")
else:
    print("  [OK] 전경 비율 정상 범위")

# DINOv2 메모리 해제 (EVA 학습을 위해)
del dino
torch.cuda.empty_cache()
import gc; gc.collect()
print("\nDINOv2 메모리 해제 완료 → EVA 학습 준비")''')

# ══════════════════════════════════════════════════════════════
# Cell 7: 마스크 품질 시각화
# ══════════════════════════════════════════════════════════════
add_cell(r'''# ═══════════════════════════════════════════════════════════════════
# 7. 마스크 품질 시각화 (마스킹된 이미지 확인)
# ═══════════════════════════════════════════════════════════════════
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image

# 한글 폰트
import subprocess
_noto = "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.otf"
if not os.path.exists(_noto):
    subprocess.run(["apt-get", "-qq", "install", "-y", "fonts-noto-cjk"], check=False)
    fm._load_fontmanager(try_read_cache=False)
for f in fm.findSystemFonts():
    if "NotoSansCJK" in f or "NotoSansKR" in f:
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

os.chdir("/content/dacon")

# 마스킹된 이미지 확인 (이미 덮어쓴 파일)
import pandas as pd
train_df = pd.read_csv("data/open/train.csv")
sample_ids = train_df["id"].tolist()

# label별 샘플 선택
stable_ids = train_df[train_df["label"] == "stable"]["id"].tolist()[:3]
unstable_ids = train_df[train_df["label"] == "unstable"]["id"].tolist()[:3]
show_ids = stable_ids + unstable_ids
labels = ["stable"] * 3 + ["unstable"] * 3

fig, axes = plt.subplots(2, len(show_ids), figsize=(4 * len(show_ids), 7))
fig.suptitle("DINOv2 Attention Masking 결과 (배경 제거됨)", fontsize=14, fontweight="bold")

for j, (sid, label) in enumerate(zip(show_ids, labels)):
    d = os.path.join("data/open/train", sid)
    
    # Front (masked)
    fr = np.array(Image.open(os.path.join(d, "front.png")))
    axes[0, j].imshow(fr)
    axes[0, j].set_title(f"{sid}\nFront ({label})", fontsize=9)
    axes[0, j].axis("off")
    
    # Top (masked)
    tp = np.array(Image.open(os.path.join(d, "top.png")))
    axes[1, j].imshow(tp)
    axes[1, j].set_title(f"Top ({label})", fontsize=9)
    axes[1, j].axis("off")

    # 전경 비율 표시
    fr_fg = (fr.sum(axis=2) > 0).mean()
    tp_fg = (tp.sum(axis=2) > 0).mean()
    axes[0, j].text(5, 375, f"FG:{fr_fg:.0%}", color="lime", fontsize=8,
                    bbox=dict(facecolor="black", alpha=0.7))
    axes[1, j].text(5, 375, f"FG:{tp_fg:.0%}", color="lime", fontsize=8,
                    bbox=dict(facecolor="black", alpha=0.7))

plt.tight_layout()
plt.show()

print("위 이미지에서 확인할 것:")
print("  1. 구조물(탑)이 온전히 보이는가? → 마스크가 구조물을 잘랐으면 문제")
print("  2. 체커보드 바닥이 제거되었는가? → 검은색이면 성공")
print("  3. 하늘/배경이 제거되었는가? → 검은색이면 성공")
print("  ★ 구조물 경계가 약간 거칠어도 OK — EVA는 내부 패턴에 집중")''')

# ══════════════════════════════════════════════════════════════
# Cell 8: EVA-Giant 학습 설정
# ══════════════════════════════════════════════════════════════
add_cell(r'''# ═══════════════════════════════════════════════════════════════════
# 8. EVA-Giant 학습 설정 (마스킹 이미지 전용 — 처음부터 학습)
# ═══════════════════════════════════════════════════════════════════
# 핵심: DINOv2가 배경을 완벽히 제거했으므로
#       EVA는 오직 구조물의 형태/꺾임/기울기에만 집중
#       → 체커보드 과적합 원천 차단
# ═══════════════════════════════════════════════════════════════════
import os, sys, shlex, subprocess, shutil
os.chdir("/content/dacon")

EVA_CONFIG = {
    "backbone": "eva_giant",
    "epochs": 30,             # v10에서 fold3이 ep20 상한 도달 → 30으로 확장
    "patience": 15,
    "head_lr": 1e-4,
    "bb_lr": 1e-5,
    "weight_decay": 0.05,
    "drop_rate": 0.3,
    "layer_decay": 0.9,
    "warmup_epochs": 2,
    "merge_dev": True,
    "loss": "ce",
    "no_mixup": True,
    "simple_aug": True,
    "head_type": "simple",
    "scheduler": "cosine",
    "batch_size": 16,
}

print(f"{'='*60}")
print(f"  EVA-Giant 학습 설정 (마스킹 이미지)")
print(f"{'='*60}")
for k, v in EVA_CONFIG.items():
    print(f"  {k:>16}: {v}")
print(f"\n  [핵심] 배경 제거된 이미지로 처음부터 학습")
print(f"  [변경] epochs: 20 → 30 (v10 fold3 상한 도달 대응)")
print(f"  [기대] 체커보드 과적합 제거 → Test Loss 대폭 개선")

# ═══════════════════════════════════════════════════════════════════
# 학습 헬퍼 함수
# ═══════════════════════════════════════════════════════════════════
def run_finetune_fold(cfg, fold, sync_fn=None):
    """단일 fold finetune 학습을 subprocess로 실행 (train_v2.py)"""
    os.chdir("/content/dacon")
    backbone = cfg["backbone"]

    cmd = [
        sys.executable, "-u", "train_v2.py",
        "--backbone", backbone,
        "--stage", "finetune",
        "--finetune_epochs", str(cfg["epochs"]),
        "--patience", str(cfg["patience"]),
        "--fold", str(fold),
        "--loss", cfg.get("loss", "ce"),
        "--scheduler", cfg.get("scheduler", "cosine"),
        "--head_type", cfg.get("head_type", "simple"),
        "--head_lr", str(cfg["head_lr"]),
        "--bb_lr", str(cfg["bb_lr"]),
        "--weight_decay", str(cfg["weight_decay"]),
        "--drop_rate", str(cfg["drop_rate"]),
        "--layer_decay", str(cfg["layer_decay"]),
        "--warmup_epochs", str(cfg.get("warmup_epochs", 2)),
        "--grad_checkpointing", "--resume", "--init_from_best",
        "--num_workers", "0",
    ]
    if cfg.get("merge_dev"):   cmd += ["--merge_dev"]
    if cfg.get("no_mixup"):    cmd += ["--no_mixup"]
    if cfg.get("simple_aug"):  cmd += ["--simple_aug"]
    cmd += ["--batch_size_override", str(cfg.get("batch_size", 16))]

    cmd_str = shlex.join(cmd)
    print(f"CMD: {cmd_str}\n" + "=" * 60)

    env = os.environ.copy()
    env["TQDM_FORCE_TTY"] = "1"
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env
    )
    while True:
        char = process.stdout.read(1)
        if not char and process.poll() is not None:
            break
        sys.stdout.write(char)
        sys.stdout.flush()
    process.wait()

    if process.returncode != 0:
        print(f"\n[ERROR] {backbone} Fold {fold} 실패 (exit {process.returncode})")
    else:
        free = shutil.disk_usage('/content').free / 1e9
        print(f"\n디스크 여유: {free:.1f} GB")
        if sync_fn:
            print("Drive 동기화...")
            sync_fn(backbone)

for fold in range(5):
    best = os.path.join("checkpoints", f"eva_giant_fold{fold}.pth")
    status = "✓ 완료" if os.path.exists(best) else "✗ 미완료"
    print(f"  Fold {fold}: {status}")''')

# ══════════════════════════════════════════════════════════════
# Cells 9-13: EVA Fold 0-4
# ══════════════════════════════════════════════════════════════
for fold in range(5):
    add_cell(f'# 9-{fold}. EVA-Giant Fold {fold} (마스킹 이미지)\nrun_finetune_fold(EVA_CONFIG, fold={fold}, sync_fn=sync_best_to_drive)')

# ══════════════════════════════════════════════════════════════
# Cell 14: 학습 곡선 분석
# ══════════════════════════════════════════════════════════════
add_cell(r'''# ═══════════════════════════════════════════════════════════════════
# 10. EVA-Giant 학습 곡선 분석 (마스킹 이미지 학습)
# ═══════════════════════════════════════════════════════════════════
import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

os.chdir("/content/dacon")

def plot_learning_curves(backbone, title_suffix=""):
    ckpt_dir = "checkpoints"
    log_files = sorted(glob.glob(os.path.join(ckpt_dir, f"{backbone}_fold*_log.csv")))
    if not log_files:
        print(f"[WARN] {backbone} 로그 없음")
        return []

    n_folds = len(log_files)
    fig, axes = plt.subplots(2, n_folds, figsize=(5 * n_folds, 8), squeeze=False)
    fig.suptitle(f"{backbone} 학습 곡선 {title_suffix}", fontsize=14, fontweight="bold")

    summaries = []
    for i, log_path in enumerate(log_files):
        df = pd.read_csv(log_path)
        fold_name = os.path.basename(log_path).replace("_log.csv", "")
        epochs = df["epoch"].values
        train_loss = df["train_loss"].values
        val_loss = df["val_logloss"].values

        ax1 = axes[0][i]
        ax1.plot(epochs, train_loss, 'b-o', markersize=3, label="Train", linewidth=1.5)
        ax1.plot(epochs, val_loss, 'r-s', markersize=3, label="Val", linewidth=1.5)
        best_idx = val_loss.argmin()
        best_ep = epochs[best_idx]
        ax1.axvline(best_ep, color='green', linestyle='--', alpha=0.7,
                    label=f"Best ep={best_ep}")
        ax1.scatter([best_ep], [val_loss[best_idx]], color='green', s=100,
                    zorder=5, marker='*')
        ax1.set_title(fold_name, fontsize=10)
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.legend(fontsize=7); ax1.set_yscale("log"); ax1.grid(True, alpha=0.3)

        ax2 = axes[1][i]
        gap = val_loss - train_loss
        colors = ['green' if g < 0.02 else 'orange' if g < 0.05 else 'red'
                  for g in gap]
        ax2.bar(epochs, gap, color=colors, alpha=0.7, edgecolor='gray', linewidth=0.5)
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.axhline(0.02, color='green', linestyle=':', alpha=0.5)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val-Train Gap")
        ax2.set_title("과적합 갭", fontsize=9); ax2.grid(True, alpha=0.3)

        summaries.append({
            "fold": fold_name, "total_ep": len(epochs),
            "best_ep": best_ep, "best_val": val_loss[best_idx],
            "best_train": train_loss[best_idx],
            "gap": gap[best_idx],
        })

    plt.tight_layout(); plt.show()

    print(f"\n{'='*80}")
    print(f"  {'Fold':<28} {'Best Ep':>8} {'Val Loss':>12} {'Train':>12} {'Gap':>10}")
    print(f"{'='*80}")
    for s in summaries:
        status = "OK" if s["gap"] < 0.02 else "WARN" if s["gap"] < 0.05 else "OVERFIT"
        print(f"  {s['fold']:<28} {s['best_ep']:>8} {s['best_val']:>12.7f} "
              f"{s['best_train']:>12.7f} {s['gap']:>10.4f}  {status}")
    avg_gap = np.mean([s["gap"] for s in summaries])
    print(f"\n  평균 과적합 갭: {avg_gap:.4f}")
    if avg_gap > 0.05:
        print("  [!] 과적합 → regularization 강화 필요")
    elif avg_gap > 0.02:
        print("  [~] 경미한 과적합 → 모니터링")
    else:
        print("  [v] 양호")
    return summaries

eva_summaries = plot_learning_curves("eva_giant", "(마스킹 이미지 학습)")''')

# ══════════════════════════════════════════════════════════════
# Cell 15: Dev 평가
# ══════════════════════════════════════════════════════════════
add_cell(r'''# ═══════════════════════════════════════════════════════════════════
# 11. Dev 평가 (마스킹 이미지 + EVA)
# ═══════════════════════════════════════════════════════════════════
import torch, os, numpy as np, pandas as pd
import torch.nn.functional as TF
from torch.amp import autocast
from sklearn.metrics import log_loss as sk_logloss
import albumentations as A
from albumentations.pytorch import ToTensorV2

os.chdir("/content/dacon")
from models import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_size = 336
norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
crop = int(img_size * 0.9)
val_tf = A.Compose([A.CenterCrop(crop, crop), A.Resize(img_size, img_size),
                     A.Normalize(**norm), ToTensorV2()])

def load_split(csv_path, data_dir, tf):
    df = pd.read_csv(csv_path)
    fronts, tops, labels = [], [], []
    for _, row in df.iterrows():
        sid = row["id"]
        d = os.path.join(data_dir, sid)
        fr = np.array(Image.open(os.path.join(d, "front.png")))
        tp = np.array(Image.open(os.path.join(d, "top.png")))
        fronts.append(tf(image=fr)["image"])
        tops.append(tf(image=tp)["image"])
        labels.append(1 if row["label"] == "unstable" else 0)
    return torch.stack(fronts), torch.stack(tops), np.array(labels)

from PIL import Image

print("Loading masked dev data...")
dev_fronts, dev_tops, dev_labels = load_split(
    "data/open/dev.csv", "data/open/dev", val_tf)
print(f"  Dev: {len(dev_labels)} (unstable={dev_labels.sum()}, stable={(1-dev_labels).sum()})")

# 모든 fold 평가
ckpt_dir = "checkpoints"
results = []
all_probs = []

for fold in range(5):
    ckpt_path = os.path.join(ckpt_dir, f"eva_giant_fold{fold}.pth")
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] fold {fold}")
        continue

    model = build_model("eva_giant", pretrained=False, num_classes=2,
                       drop_rate=0.0, head_type="simple")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
    model.load_state_dict(filtered, strict=False)
    model = model.to(device).eval()

    fold_probs = []
    with torch.no_grad():
        for i in range(0, len(dev_labels), 32):
            fr = dev_fronts[i:i+32].to(device)
            tp = dev_tops[i:i+32].to(device)
            with autocast("cuda"):
                logits = model(fr, tp)
            probs = TF.softmax(logits.float(), dim=1).cpu().numpy()
            fold_probs.append(probs)
    fold_probs = np.concatenate(fold_probs)
    all_probs.append(fold_probs)

    eps = 1e-7
    ll = sk_logloss(dev_labels, np.clip(fold_probs, eps, 1-eps), labels=[0, 1])
    acc = (fold_probs.argmax(1) == dev_labels).mean()
    print(f"  fold {fold}: Dev LL={ll:.6f} Acc={acc:.4f}")

    del model; torch.cuda.empty_cache()
    results.append({"fold": fold, "dev_ll": ll, "dev_acc": acc})

if all_probs:
    mean_probs = np.mean(all_probs, axis=0)
    eps = 1e-7
    ens_ll = sk_logloss(dev_labels, np.clip(mean_probs, eps, 1-eps), labels=[0, 1])
    ens_acc = (mean_probs.argmax(1) == dev_labels).mean()
    
    print(f"\n{'='*60}")
    print(f"  EVA (마스크) 앙상블: Dev LL={ens_ll:.6f} Acc={ens_acc:.4f}")
    print(f"{'='*60}")
    print(f"  v10 EVA (원본): Dev LL ≈ 0.006 (비교)")
    print(f"  → 마스킹 효과: {'개선' if ens_ll < 0.006 else '추가 분석 필요'}")''')

# ══════════════════════════════════════════════════════════════
# Cell 16: 테스트 추론 + 제출
# ══════════════════════════════════════════════════════════════
add_cell(r'''# ═══════════════════════════════════════════════════════════════════
# 12. 테스트 추론 + 제출 파일 생성
# ═══════════════════════════════════════════════════════════════════
import os, datetime
import numpy as np, pandas as pd, torch
import torch.nn.functional as TF
from torch.amp import autocast
from torch.utils.data import DataLoader

os.chdir("/content/dacon")
from models import build_model, get_backbone_config
from inference_v2 import (
    DualCropDataset, make_dual_transforms, make_dual_tta_transforms,
    predict, predict_tta,
)

USE_FOLDS = [0, 1, 2, 3, 4]
USE_TTA = True
FRONT_CROP = 0.9
TOP_CROP = 0.9
NUM_WORKERS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_csv = "data/open/sample_submission.csv"
test_dir = "data/open/test"

# --- EVA 추론 (마스킹된 테스트 이미지) ---
print(f"{'='*60}")
print(f"  EVA-Giant 추론 (마스킹 이미지) {'(TTA)' if USE_TTA else ''}")
print(f"{'='*60}")

cfg = get_backbone_config("eva_giant")
img_size = cfg["img_size"]

all_preds = []
for fold in USE_FOLDS:
    ckpt_path = os.path.join("checkpoints", f"eva_giant_fold{fold}.pth")
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] fold {fold}")
        continue

    model = build_model("eva_giant", pretrained=False, num_classes=2,
                       drop_rate=0.0, head_type="simple")
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
    model.load_state_dict(filtered, strict=False)
    model = model.to(device)

    if USE_TTA:
        probs, ids = predict_tta(model, test_csv, test_dir, img_size, device,
                                 FRONT_CROP, TOP_CROP, bs=32, nw=NUM_WORKERS)
    else:
        ftf, ttf = make_dual_transforms(img_size, FRONT_CROP, TOP_CROP)
        ds = DualCropDataset(test_csv, test_dir, ftf, ttf)
        loader = DataLoader(ds, batch_size=32, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)
        probs, ids = predict(model, loader, device)

    all_preds.append(probs)
    print(f"  fold {fold}: P(unstable) mean={probs[:, 1].mean():.6f}")
    del model; torch.cuda.empty_cache()

if not all_preds:
    raise RuntimeError("체크포인트 없음!")

mean_probs = np.mean(all_preds, axis=0)

# --- 확률 분포 ---
p = mean_probs[:, 1]
print(f"\n  [EVA Masked Ensemble]")
print(f"    mean={p.mean():.8f} std={p.std():.8f}")
print(f"    min={p.min():.8f}  max={p.max():.8f}")
print(f"    unstable(>0.5): {(p > 0.5).sum()} | stable: {(p <= 0.5).sum()}")

# --- 제출 파일 ---
out_dir = "submissions"
os.makedirs(out_dir, exist_ok=True)
sub_df = pd.read_csv("data/open/sample_submission.csv", encoding="utf-8-sig")
ts = datetime.datetime.now().strftime("%m%d_%H%M")
eps = 1e-7

tta_tag = "_tta" if USE_TTA else ""
fname = f"submission_eva_masked{tta_tag}_{ts}.csv"
fpath = os.path.join(out_dir, fname)

sub = sub_df.copy()
sub["unstable_prob"] = np.clip(mean_probs[:, 1], eps, 1 - eps)
sub["stable_prob"] = 1.0 - sub["unstable_prob"]
sub[["id", "unstable_prob", "stable_prob"]].to_csv(fpath, index=False)

# 검증
assert not np.isnan(mean_probs).any(), "NaN!"
assert np.allclose(mean_probs.sum(axis=1), 1.0, atol=1e-5), "확률 합 != 1"

print(f"\n{'='*60}")
print(f"  제출 파일: {fname}")
print(f"  NaN: 없음 ✓  확률합: 1.0 ✓")
print(f"{'='*60}")
print(f"\n  ★ 이 파일로 제출하세요!")
print(f"  → v10 XGB(0.0476) 대비 개선이 목표")
print(f"  → 배경 제거로 과적합 감소 기대")''')

# ══════════════════════════════════════════════════════════════
# Cell 17: Drive 동기화
# ══════════════════════════════════════════════════════════════
add_cell(r'''# ═══════════════════════════════════════════════════════════════════
# 13. Drive 동기화 + 최종 요약
# ═══════════════════════════════════════════════════════════════════
import os, shutil

os.chdir("/content/dacon")
DRIVE_DIR = "/content/drive/MyDrive/dacon"
DRIVE_CKPT = os.path.join(DRIVE_DIR, "checkpoints")

print(f"{'='*60}")
print(f"  최종 동기화")
print(f"{'='*60}")

# EVA (마스크 학습) 체크포인트 → Drive
print("\n[1] EVA (masked) 체크포인트 → Drive")
sync_best_to_drive("eva_giant")

# 로그 파일 → Drive
for f in os.listdir("checkpoints"):
    if f.startswith("eva_giant") and f.endswith("_log.csv"):
        src = os.path.join("checkpoints", f)
        dst = os.path.join(DRIVE_CKPT, f)
        shutil.copy2(src, dst)
        print(f"  로그: {f}")

# 현황
print(f"\n{'='*60}")
print(f"  v11 최종 요약")
print(f"{'='*60}")
ckpts = sorted([f for f in os.listdir(DRIVE_CKPT)
                if f.startswith("eva_giant") and f.endswith('.pth')
                and '_ckpt' not in f and 'v10_backup' not in f])
folds_done = [f"fold{i}" for i in range(5) if f"eva_giant_fold{i}.pth" in ckpts]
print(f"  EVA (masked) 완료: {len(folds_done)}/5 folds {folds_done}")

sub_dir = os.path.join(DRIVE_DIR, "submissions")
subs = sorted([f for f in os.listdir(sub_dir) if f.endswith('.csv')])
print(f"  제출 파일: {len(subs)}개")
for s in subs[-3:]:
    print(f"    {s}")

# v10 백업 확인
backup_dir = os.path.join(DRIVE_CKPT, "v10_backup")
if os.path.isdir(backup_dir):
    v10_ckpts = [f for f in os.listdir(backup_dir) if f.endswith('.pth')]
    print(f"\n  v10 백업: {len(v10_ckpts)}개 ({backup_dir})")

free = shutil.disk_usage('/content').free / 1e9
print(f"\n  디스크 여유: {free:.1f} GB")
print(f"\n✅ v11 완료! 2-Stage Pipeline (DINOv2 Masking → EVA)")
print(f"  → DINOv2가 배경 제거 → EVA가 순수 구조물만 분류")
print(f"  → v10 대비 과적합 감소 기대")''')

# ══════════════════════════════════════════════════════════════
# Write notebook
# ══════════════════════════════════════════════════════════════
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "colab_train11.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Created: {output_path}")
print(f"Cells: {len(cells)}")
for i, c in enumerate(cells):
    first_line = c["source"][0].strip() if c["source"] else ""
    print(f"  Cell {i+1}: {first_line[:70]}")
