# dacon-structural-stability

멀티뷰 이미지 기반 구조물 안정성 예측 — 대규모 ViT 백본 + 최대 증강 파이프라인

---

## 사용 모델

| 키 | 모델명 | 파라미터 | 입력 | Tier | 설명 |
|---|---|---|---|---|---|
| `eva_giant` | EVA-Giant/14 | 1.0B | 336px | A100 | ImageNet #1급, IN-22K → IN-1K 파인튜닝 |
| `dinov3_huge` | DINOv3 ViT-H+/16 | 840M | 384px | A100 | 자기지도학습 SOTA, RoPE 유연 해상도 |
| `siglip_so400m` | SigLIP SO400M/14 | 428M | 384px | A100 | Vision-Language 대비학습 |
| `eva02_large` | EVA02-Large/14 | 305M | 448px | Local | 상위 솔루션 검증 완료 |
| `dinov2_large` | DINOv2 ViT-L/14 reg4 | 304M | 336px | Local | 공간 구조 이해 특화 |

## 적용 기법 요약

### 모델 아키텍처
- **공유 백본 Dual-View**: front/top 이미지를 동일 백본으로 인코딩
- **Attention Gate Fusion**: 두 뷰 특징의 가중 합산 (학습 가능 어텐션 게이트)
- **MLP Head**: LayerNorm → 512 → 256 → 2 (Dropout 0.3/0.15)
- **비디오 프레임 융합**: 시뮬레이션 영상 프레임 temporal mean pooling (선택)
- **Gradient Checkpointing**: VRAM 절약용 체크포인팅

### 증강
- RandomResizedCrop, HorizontalFlip, Affine (translate/scale/rotate)
- Perspective, ColorJitter, HSV, BrightnessContrast
- GaussianBlur, MotionBlur, GaussNoise, ISONoise
- RandomShadow, RandomFog, CoarseDropout, RandomGrayscale
- **CutMix** (30%) + **Mixup** (30%) + Clean (40%) — 배치 단위 랜덤

### 손실 함수
- **Focal Loss** (α=0.25~0.3, γ=2.0) × 0.7 + **Label Smoothing** (ε=0.05) × 0.3

### 학습 전략
- Stage 1: **ShapeStacks h=6 Pretrain** — 같은 시나리오 카메라 앵글 페어링 (dual-view)
- Stage 2: **5-Fold CV Finetune** — Dacon train + dev 3× oversample
- Differential LR (백본 0.1× / 헤드 1×)
- Cosine Annealing + Warmup 2 epoch
- AMP fp16, Gradient Accumulation, WeightedRandomSampler

### 추론
- **멀티 백본 앙상블**: 여러 백본 × 5-fold 체크포인트 평균
- **TTA**: 원본 + HFlip + Brightness + CenterCrop (4종)
- **Temperature Scaling**: 확률 보정

---

## 환경 설정 (Setup)

```powershell
cd c:\Pyg\Projects\dacon\dacon-structural-stability
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

---

## 로컬 학습 (12GB GPU — RTX 4070 Ti)

### Stage 1: Pretrain

```powershell
python train.py --backbone eva02_large --stage pretrain --pretrain_epochs 15 --grad_checkpointing --resume
python train.py --backbone dinov2_large --stage pretrain --pretrain_epochs 15 --grad_checkpointing --resume
```

### Stage 2: Finetune (5-Fold)

```powershell
python train.py --backbone eva02_large --stage finetune --finetune_epochs 50 --include_dev --grad_checkpointing --resume
python train.py --backbone dinov2_large --stage finetune --finetune_epochs 50 --include_dev --grad_checkpointing --resume
```

### Pretrain + Finetune 한번에

```powershell
python train.py --backbone eva02_large --stage both --pretrain_epochs 15 --finetune_epochs 50 --include_dev --grad_checkpointing --resume
```

---

## 추론 (Inference)

### Dev 성능 검증

```powershell
python inference.py --backbones eva02_large --tta --validate
python inference.py --backbones eva02_large dinov2_large --tta --validate
```

### 최종 제출 파일 생성

```powershell
python inference.py --backbones eva02_large dinov2_large --tta --temperature 1.0
```

결과물: `submissions/` 폴더에 CSV 파일 생성

---

## Google Colab (A100) 사용법

> `colab_train.ipynb` 를 Colab에 업로드하여 사용합니다.

### 1. 사전 준비

Google Drive `내 드라이브/dacon/` 폴더에 다음 파일 업로드:
- `models.py`, `datasets.py`, `train.py`, `inference.py`
- `data.zip` (대회 data 폴더 압축)
- (선택) `shapestacks.zip` — ShapeStacks 원본, h=6만 자동 추출

### 2. Colab에서 실행

1. `colab_train.ipynb` 업로드 후 열기
2. 런타임 → 런타임 유형 변경 → **GPU: A100**
3. 셀 1~5 순서대로 실행 (환경 설정 + 데이터 추출)
4. 학습 설정 셀에서 백본 선택 → Pretrain → Finetune 실행

### 3. 재접속 시

> 셀 1~5 재실행 → 학습 셀 실행 → **자동으로 마지막 체크포인트에서 이어서 학습**

- 체크포인트: Google Drive에 심볼릭 링크로 자동 저장 (끊겨도 안전)
- 원자적 저장: `.tmp` → `rename` 방식으로 저장 중 끊겨도 기존 체크포인트 보존
- ShapeStacks h=6: Drive에 1회 추출 후 다음 세션부터 심볼릭 링크만 (재추출 불필요)

### 4. 디스크 절약

- ShapeStacks: 전체 33GB 대신 **h=6만 ~11GB** 선택 추출
- 대회 data.zip: 로컬 복사 후 추출 → 임시 zip 자동 삭제
- zip + 추출 이중 점유로 69GB 차지하는 문제 해결

---

## 전체 파이프라인 (로컬)

```powershell
.\venv\Scripts\activate

# ShapeStacks 다운로드 (최초 1회)
python download_shapestacks.py

# 백본 1: EVA02-Large
python train.py --backbone eva02_large --stage both --pretrain_epochs 15 --finetune_epochs 50 --include_dev --grad_checkpointing --resume

# 백본 2: DINOv2-Large
python train.py --backbone dinov2_large --stage both --pretrain_epochs 15 --finetune_epochs 50 --include_dev --grad_checkpointing --resume

# 앙상블 추론
python inference.py --backbones eva02_large dinov2_large --tta --validate
python inference.py --backbones eva02_large dinov2_large --tta
```
