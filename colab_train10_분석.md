# colab_train10.ipynb 심층 분석

## 2-Stage Cascaded Ensemble: DINOv2 + EVA-Giant

> **목표**: 구조물 안정성 이진 분류 (stable / unstable)  
> **전략**: 서로 다른 강점을 가진 두 Vision Transformer를 독립 학습 → 메타러너로 최적 결합  
> **대회**: Dacon 구조물 안정성 예측 경진대회

---

## 목차

1. [전체 아키텍처 개요](#1-전체-아키텍처-개요)
2. [데이터 구조](#2-데이터-구조)
3. [환경 설정 (Cell 1~5)](#3-환경-설정-cell-15)
4. [DINOv2 학습 파이프라인 (Cell 6~13)](#4-dinov2-학습-파이프라인-cell-613)
5. [EVA-Giant 학습 파이프라인 (Cell 14~20)](#5-eva-giant-학습-파이프라인-cell-1420)
6. [모델 비교 분석 (Cell 21)](#6-모델-비교-분석-cell-21)
7. [Stacking Meta-Learner (Cell 22)](#7-stacking-meta-learner-cell-22)
8. [2-Stage Ensemble 추론 (Cell 23)](#8-2-stage-ensemble-추론-cell-23)
9. [앙상블 가중치 최적화 (Cell 24)](#9-앙상블-가중치-최적화-cell-24)
10. [제출 파일 생성 (Cell 25)](#10-제출-파일-생성-cell-25)
11. [시각화 및 분석 (Cell 26~28)](#11-시각화-및-분석-cell-2628)
12. [Drive 동기화 (Cell 29)](#12-drive-동기화-cell-29)
13. [핵심 기법 정리](#13-핵심-기법-정리)
14. [하이퍼파라미터 비교표](#14-하이퍼파라미터-비교표)
15. [개선 포인트 및 회고](#15-개선-포인트-및-회고)

---

## 1. 전체 아키텍처 개요

### 왜 2-Stage Cascaded Ensemble인가?

단일 모델로는 한계가 있다. 이 노트북의 핵심 가설:

| 모델 | 강점 | 약점 |
|------|------|------|
| **DINOv2-Large** | 구조물 형태 인식 (spatial feature) | unstable 세부 분류 약함 |
| **EVA-Giant** | fine-grained 분류 정밀도 | 배경 의존 → 과적합 위험 |

**DINOv2**는 self-supervised로 사전학습되어 이미지 내 공간적 구조를 잘 이해하지만, "이 구조물이 불안정한가?"라는 세밀한 판단에는 약하다. 반면 **EVA-Giant**는 1.0B 파라미터의 거대 모델로 분류 정밀도가 높지만, 배경 텍스처에 의존하는 경향이 있다.

→ **두 모델을 독립 학습 후 메타러너로 결합**하면 양쪽 강점을 활용할 수 있다.

### 전체 흐름도

```
┌─────────────────────────────────────────────────────────┐
│                    환경 설정 (Cell 1~5)                    │
│  Drive 마운트 → 패키지 설치 → 소스 복사 → 데이터 해제 → 검증  │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Stage 1: DINOv2-Large (Cell 6~13)           │
│                                                          │
│  ┌──────────┐   5-Fold CV   ┌──────────┐               │
│  │ DINOv2   │──────────────▶│ 5개 Best │               │
│  │ cross_attn│  (Fold 0~4)  │ 가중치    │               │
│  │ head      │              └──────────┘               │
│  └──────────┘               │  학습 곡선 분석            │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Stage 2: EVA-Giant (Cell 14~20)             │
│                                                          │
│  ┌──────────┐   5-Fold CV   ┌──────────┐               │
│  │ EVA-Giant│──────────────▶│ 5개 Best │               │
│  │ simple   │  (Fold 0~4)  │ 가중치    │               │
│  │ head     │              └──────────┘                │
│  └──────────┘               │  학습 곡선 분석            │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│                 모델 비교 (Cell 21)                       │
│  Train/Dev Loss 비교 → Unstable/Stable per-class 분석    │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│           Stacking Meta-Learner (Cell 22)                │
│                                                          │
│  OOF 예측 생성 → 최적 가중치 탐색 (Grid Search)           │
│                → Logistic Regression 스태킹              │
│                → XGBoost 스태킹                          │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│         2-Stage Ensemble 추론 (Cell 23)                  │
│                                                          │
│  DINOv2 5-Fold TTA ──┐                                  │
│                       ├─→ 가중평균 / LR Stack / XGB Stack │
│  EVA 5-Fold TTA  ────┘                                  │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│             제출 + 시각화 (Cell 24~29)                    │
│  가중치 최적화 → 제출 생성 → 성능 비교 → Attention 분석     │
│  → 오류 분석 → Drive 동기화                               │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 데이터 구조

### 입력 데이터

각 샘플은 **두 장의 이미지**(front view + top view)로 구성:

```
data/open/
├── train.csv          # 학습 데이터 라벨 (id, label)
├── dev.csv            # 개발 데이터 라벨
├── sample_submission.csv
├── train/
│   ├── TRAIN_001/
│   │   ├── front.png   ← 건물 정면 사진
│   │   └── top.png     ← 건물 위에서 본 사진
│   └── ...
├── dev/
│   └── DEV_001/ ...
└── test/
    └── TEST_001/ ...
```

### 라벨

- `stable`: 구조물이 안정적
- `unstable`: 구조물이 불안정

### 왜 Dual-View인가?

정면(front)만으로는 안 보이는 기울어짐이나 균열이 위에서(top) 보면 드러날 수 있다. 모델은 두 뷰의 정보를 **동시에** 받아 판단해야 한다.

---

## 3. 환경 설정 (Cell 1~5)

### Cell 1: 환경 설정

```python
# 핵심 전략: 캐시를 Drive로 보내 재연결 시 재다운로드 방지
os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
os.environ["TORCH_HOME"] = os.path.join(CACHE_DIR, "torch")
```

**왜 캐시를 Drive에?**  
Colab은 런타임이 끊기면 로컬 디스크가 초기화된다. 모델 가중치 (DINOv2 ~1.2GB, EVA ~4GB)를 매번 다운로드하면 시간 낭비. Drive 캐시에 저장하면 재연결 시 바로 사용 가능.

**체크포인트 전략: 2-Tier 저장**
- **로컬 SSD** (`/content/dacon/checkpoints/`): 빠른 저장/로드 (학습 중 사용)
- **Drive** (`/content/drive/MyDrive/dacon/checkpoints/`): 영구 백업 (세션 종료 후에도 유지)

### Cell 2: 소스 파일 복사 + 체크포인트 설정

```
Drive → 로컬 SSD로 소스 파일 복사
models.py, datasets.py, train_v2.py, inference_v2.py 등
```

핵심 함수: `sync_best_to_drive(backbone)` — 학습 완료 후 best 가중치만 Drive로 복사 (resume ckpt는 제외하여 용량 절약)

### Cell 3: 코랩 경쟁 모드 패치

이 셀이 가장 중요한 환경 패치를 수행한다:

#### 1) train_v2.py 검증
```python
for required in ["--layer_decay", "--warmup_epochs", "cross_attn"]:
    if required not in tv2_text:
        missing.append(required)
```
**왜 검증이 필요한가?** Drive에 올라간 train_v2.py가 **구버전**일 수 있다. 이 노트북은 `--layer_decay`, `--warmup_epochs`, `--head_type cross_attn`을 사용하므로, 이들이 없으면 학습 시 argparse 에러가 발생한다.

#### 2) 증강 강화 패치

기존 → 강화 패치 비교:

| 증강 | 기존 | 강화 |
|------|------|------|
| RandomResizedCrop scale | (0.8, 1.0) | **(0.72, 1.0)** |
| Affine rotate | (-10, 10) | **(-12, 12)** |
| CoarseDropout holes | (1, 6) | **(1, 8)** |
| + CLAHE | 없음 | **추가** |
| + Sharpen | 없음 | **추가** |
| + ImageCompression | 없음 | **추가** |

#### 3) TTA 6개로 확장

기존 4개 → 6개:

| TTA | 설명 |
|-----|------|
| 0 | Center Crop (기본) |
| 1 | Horizontal Flip |
| 2 | Brightness/Contrast 조정 |
| 3 | **CLAHE** (새로 추가) |
| 4 | **미세 Affine 변환** (새로 추가) |
| 5 | Tighter Crop (81%) |

### Cell 4: 데이터 압축 해제

```
Drive의 data.zip → 로컬 SSD로 압축 해제
```
로컬 SSD가 Drive 대비 **10~50배 빠른 I/O**이므로, 데이터를 로컬에 풀어 학습 속도를 극대화한다.

### Cell 5: 환경 검증

모든 필수 파일의 존재와 백본별 체크포인트 현황을 점검. 학습 시작 전 **최종 체크리스트** 역할.

---

## 4. DINOv2 학습 파이프라인 (Cell 6~13)

### Cell 6: DINOv2 설정 + 학습 함수

#### DINOv2-Large 모델 소개

| 항목 | 값 |
|------|-----|
| 모델명 | `vit_large_patch14_reg4_dinov2.lvd142m` |
| 파라미터 | ~304M |
| 입력 크기 | 336 × 336 |
| 패치 크기 | 14 × 14 |
| 사전학습 | Self-supervised (DINOv2) |

**DINOv2의 특징**: Facebook의 self-supervised 학습으로 ImageNet 라벨 없이도 강력한 시각적 특징을 학습. 특히 **공간적 구조(spatial structure)**를 잘 이해하므로, 건물과 같은 구조물 인식에 강하다.

#### DINO_CONFIG 상세

```python
DINO_CONFIG = {
    "backbone": "dinov2_large",
    "epochs": 20,
    "patience": 10,           # Early Stopping: 10 에포크 동안 개선 없으면 중단
    "head_lr": 1e-4,          # Classification head 학습률
    "bb_lr": 1e-5,            # Backbone (ViT) 학습률 — 10배 낮음
    "weight_decay": 0.05,     # L2 정규화
    "drop_rate": 0.3,         # Dropout
    "layer_decay": 0.75,      # ★ Layer-wise LR Decay
    "warmup_epochs": 2,       # Linear warmup
    "head_type": "cross_attn",# ★ CrossViewFusion head
    "scheduler": "cosine",    # Cosine Annealing
    "batch_size": 16,
}
```

#### Layer-wise Learning Rate Decay (LLRD)

```
Layer 0 (입력에 가까움): lr × 0.75^N
Layer 1:                  lr × 0.75^(N-1)
...
Layer N (출력에 가까움): lr × 0.75^0 = lr
```

**원리**: 하위 레이어(일반적 특징)는 천천히, 상위 레이어(task-specific 특징)는 빠르게 학습. 사전학습된 일반 지식을 보존하면서 fine-tuning.

DINOv2에서 `layer_decay=0.75`는 상당히 공격적인 감쇠로, 하위 레이어를 거의 고정에 가깝게 유지한다.

#### CrossViewFusion Head (cross_attn)

```
Front 패치 토큰 ──┐
                   ├──→ Transformer Cross-Attention ──→ Classification
Top 패치 토큰  ───┘
```

**왜 cross_attn?**  
일반적인 `simple` head는 front와 top의 CLS 토큰만 concat하여 분류한다. 하지만 `cross_attn`은 **패치 레벨에서 두 뷰를 교차 융합**한다:

1. Front의 각 패치가 Top의 모든 패치에 attention
2. Top의 각 패치가 Front의 모든 패치에 attention
3. 융합된 표현으로 최종 분류

→ "정면에서 보이는 이 기둥이 위에서 보면 기울어져 있다" 같은 **3D 구조적 관계**를 포착할 수 있다.

#### HEAD_TYPE_MAP

```python
HEAD_TYPE_MAP = {
    "dinov2_large": "cross_attn",
    "eva_giant": "simple",
}
```

**왜 EVA는 simple?** EVA-Giant는 1.0B 파라미터로 이미 충분한 표현력을 가지므로, 추가적인 cross-attention head가 오히려 과적합을 유발할 수 있다. DINOv2(304M)는 상대적으로 작아서 cross-attention이 효과적.

#### run_finetune_fold() 함수

```python
def run_finetune_fold(cfg, fold, sync_fn=None):
    cmd = [
        sys.executable, "-u", "train_v2.py",  # ★ train.py가 아님!
        "--backbone", backbone,
        "--stage", "finetune",
        "--layer_decay", str(cfg["layer_decay"]),
        "--warmup_epochs", str(cfg.get("warmup_epochs", 2)),
        "--grad_checkpointing",  # VRAM 절약
        "--resume", "--init_from_best",
        "--num_workers", "0",    # Colab에서는 0이 안정적
        ...
    ]
```

**중요 포인트**:
- `train_v2.py` 사용 (train.py는 `--layer_decay`, `--warmup_epochs` 미지원)
- `--grad_checkpointing`: EVA-Giant 1.0B 모델을 A100 40~80GB에 올리기 위해 필수
- `--num_workers 0`: Colab은 SHM(shared memory)이 작아 worker>0이면 크래시
- `--resume --init_from_best`: 이전 best에서 이어서 학습 (세션 끊김 대응)

### Cell 7: DINOv2 Pretrain (선택)

```python
DINO_PRETRAIN_EPOCHS = 0  # 0이면 건너뜀
```

**Pretrain stage**는 backbone의 feature extractor만 먼저 학습시키는 단계. 경쟁 모드에서는 시간 절약을 위해 스킵하고, 바로 finetune으로 진입.

### Cell 8~12: DINOv2 5-Fold 학습

각 셀이 하나의 fold를 학습:

```python
# Cell 8:  run_finetune_fold(DINO_CONFIG, fold=0, sync_fn=sync_best_to_drive)
# Cell 9:  run_finetune_fold(DINO_CONFIG, fold=1, sync_fn=sync_best_to_drive)
# Cell 10: run_finetune_fold(DINO_CONFIG, fold=2, sync_fn=sync_best_to_drive)
# Cell 11: run_finetune_fold(DINO_CONFIG, fold=3, sync_fn=sync_best_to_drive)
# Cell 12: run_finetune_fold(DINO_CONFIG, fold=4, sync_fn=sync_best_to_drive)
```

**5-Fold Stratified CV 전략**:
- `merge_dev=True` → train + dev 전체 데이터를 5등분
- 각 fold: 4/5로 학습, 1/5로 검증
- 결과: 5개의 독립 모델 → 앙상블 시 각각의 "시각"을 제공

**왜 fold별로 셀을 분리?**  
Colab은 90분 연속 학습 시 timeout될 수 있다. 각 fold를 별도 셀로 분리하면:
1. 하나씩 실행 가능 (런타임 관리 용이)
2. 특정 fold만 재학습 가능
3. fold 완료 시 즉시 Drive 동기화 (`sync_best_to_drive`)

### Cell 13: DINOv2 학습 곡선 분석

```python
dino_summaries = plot_learning_curves("dinov2_large", "(Stage 1: 구조물 인식)")
```

**분석 항목**:
1. **Train Loss vs Val Loss 곡선**: 양쪽 곡선의 궤적 비교
2. **과적합 갭 (Val - Train)**: 
   - < 0.02: 양호 (초록)
   - 0.02~0.05: 경고 (주황)
   - `> 0.05`: 과적합 (빨강)
3. **Best Epoch**: 가장 낮은 Val Loss를 기록한 에포크
4. **Gap 배율**: Val/Train 비율 — 1.0에 가까울수록 좋음

이 분석으로 **다음 학습 시 하이퍼파라미터 조정 방향**을 결정한다.

---

## 5. EVA-Giant 학습 파이프라인 (Cell 14~20)

### Cell 14: EVA-Giant 설정

#### EVA-Giant 모델 소개

| 항목 | 값 |
|------|-----|
| 모델명 | `eva_giant_patch14_336.m30m_ft_in22k_in1k` |
| 파라미터 | ~1.0B |
| 입력 크기 | 336 × 336 |
| 패치 크기 | 14 × 14 |
| 사전학습 | ImageNet-22K → ImageNet-1K Fine-tuned |

**EVA의 특징**: 1B 파라미터의 초대형 ViT. ImageNet-22K에서 사전학습 후 ImageNet-1K로 fine-tuned. Fine-grained classification에서 최고 수준의 성능.

#### EVA_CONFIG 상세

```python
EVA_CONFIG = {
    "backbone": "eva_giant",
    "epochs": 20,
    "patience": 15,            # DINOv2(10)보다 여유 — 수렴 느림
    "head_lr": 1e-4,
    "bb_lr": 1e-5,
    "weight_decay": 0.05,
    "drop_rate": 0.3,
    "layer_decay": 0.9,        # ★ DINOv2(0.75)보다 완만
    "warmup_epochs": 2,
    "head_type": "simple",     # ★ DINOv2와 다름
    "scheduler": "cosine",
    "batch_size": 16,
}
```

#### DINOv2 vs EVA 설정 차이 해설

| 설정 | DINOv2 | EVA | 이유 |
|------|--------|-----|------|
| layer_decay | **0.75** | **0.9** | DINOv2는 self-supervised → 하위 레이어 보존 중요. EVA는 supervised → 전 레이어 고르게 조정 |
| head_type | **cross_attn** | **simple** | 304M vs 1.0B — EVA는 자체 표현력 충분 |
| patience | **10** | **15** | EVA는 수렴이 느리므로 더 오래 기다림 |

### Cell 15~19: EVA-Giant 5-Fold 학습

DINOv2와 동일한 구조. 각 fold 학습 완료 시 Drive 동기화.

### Cell 20: EVA-Giant 학습 곡선 분석

Cell 13과 동일한 `plot_learning_curves()` 함수를 EVA에 적용.

---

## 6. 모델 비교 분석 (Cell 21)

### 평가 방법

```python
def evaluate_model(model, fronts, tops, labels, device, batch_size=32):
    # LogLoss + Accuracy + Per-class Accuracy 계산
```

**양쪽 모델의 모든 체크포인트**를 Train set과 Dev set에서 평가하여 비교:

| 메트릭 | 의미 |
|--------|------|
| Train LogLoss | 학습 데이터에서의 손실 (과적합 지표) |
| Dev LogLoss | 검증 데이터에서의 손실 (**핵심 메트릭**) |
| Dev Accuracy | 전체 정확도 |
| Dev Unstable Acc | unstable 클래스 정확도 (Recall) |
| Dev Stable Acc | stable 클래스 정확도 (Specificity) |

### 핵심 분석 포인트

```
DINOv2 → Stable Acc 높음 (구조물 인식 특화)
EVA    → Unstable Acc 높음 (세부 분류 특화)
→ 상보적 결합으로 양쪽 강점 활용 가능
```

이 분석이 **앙상블의 근거**가 된다.

---

## 7. Stacking Meta-Learner (Cell 22)

### Out-Of-Fold (OOF) 예측

```
전체 데이터 (train + dev) = N개
5-Fold CV에서 각 fold의 validation 예측 → N개 전체에 대한 예측 확보
```

**왜 OOF?** 각 모델의 5-fold 체크포인트는 **자신의 학습 데이터로 예측하면 과적합된 결과**가 나온다. 각 fold가 **본 적 없는 데이터에 대한 예측**을 모아야 공정한 메타러너 학습이 가능하다.

### 결합 전략 3가지

#### 1) 가중 평균 (Grid Search)

```python
# α · DINOv2 + (1-α) · EVA
for w in np.arange(0.0, 1.01, 0.05):
    ens = w * dino_oof + (1 - w) * eva_oof
    ll = log_loss(labels, ens)
```

0.00~1.00 범위에서 0.05 단위로 최적 α 탐색. 단순하지만 효과적.

#### 2) Logistic Regression 스태킹

```python
X_stack = [dino_prob_stable, dino_prob_unstable, eva_prob_stable, eva_prob_unstable]
lr_model = LogisticRegression(C=1.0)
lr_model.fit(X_stack, labels)
```

4차원 입력(DINOv2 2-class 확률 + EVA 2-class 확률)으로 **최적 결합 가중치를 자동 학습**. 단순 가중 평균보다 유연하다.

#### 3) XGBoost 스태킹

```python
xgb_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
xgb_model.fit(X_stack, labels)
```

**비선형 결합**이 가능하여, "DINOv2가 0.7이고 EVA가 0.3이면 unstable" 같은 복잡한 패턴을 학습할 수 있다. 다만 추가 과적합 위험이 있으므로 depth를 3으로 제한.

### OOF 결과 해석

```
DINOv2 단독 OOF LogLoss: X.XXXXXX
EVA    단독 OOF LogLoss: X.XXXXXX
가중 평균 (w=?.??):       X.XXXXXX  ← 개선!
LR Stacking:              X.XXXXXX
XGB Stacking:             X.XXXXXX
```

메타러너가 저장되어 추론 시 재사용:
```python
pickle.dump({"optimal_w": w, "lr_model": lr, "xgb_model": xgb}, f)
```

---

## 8. 2-Stage Ensemble 추론 (Cell 23)

### 추론 흐름

```
Test 데이터
    │
    ├──→ DINOv2 (5-fold × TTA 6개) → 평균 확률
    │
    ├──→ EVA-Giant (5-fold × TTA 6개) → 평균 확률
    │
    └──→ 메타러너 결합
              ├── 가중 평균 (α)
              ├── LR Stacking
              └── XGB Stacking
```

### TTA (Test-Time Augmentation)

각 테스트 이미지에 **6가지 변환**을 적용하여 6번 추론, 결과를 평균:

```
하나의 이미지 × 6 TTA × 5 fold = 30번 추론 → 평균
```

이렇게 하면 **개별 추론의 노이즈가 상쇄**되어 더 안정적인 확률을 얻는다.

### `run_model_inference()` 함수

```python
def run_model_inference(backbone):
    ht = HEAD_TYPE_MAP.get(backbone, "simple")  # 백본별 올바른 head type
    for fold in USE_FOLDS:
        model = build_model(backbone, ..., head_type=ht)
        probs, ids = predict_tta(model, test_csv, ...)
        all_preds.append(probs)
    return np.mean(all_preds, axis=0)  # fold 평균
```

### 메타러너 적용

```python
# 저장된 메타러너 로드
meta = pickle.load(f)  # {"optimal_w", "lr_model", "xgb_model"}

# 1) 가중 평균
weighted = OPTIMAL_W * dino_probs + (1 - OPTIMAL_W) * eva_probs

# 2) LR Stacking
X_test = [dino_p0, dino_p1, eva_p0, eva_p1]
lr_probs = META_LR.predict_proba(X_test)

# 3) XGB Stacking
xgb_probs = META_XGB.predict_proba(X_test)
```

---

## 9. 앙상블 가중치 최적화 (Cell 24)

### 세밀한 가중치 탐색

```python
alphas = np.linspace(0, 1, 101)  # 0.01 단위
```

Cell 22의 0.05 단위보다 더 세밀한 0.01 단위 탐색.

### 시각화 2개

#### 1) Alpha vs LogLoss 곡선

α가 0~1로 변할 때 LogLoss의 변화. U자형 곡선의 최저점이 최적 α.

#### 2) 신뢰도 기반 라우팅 분석

```
x축: DINOv2 신뢰도 (|P(unstable) - 0.5| × 2)
y축: EVA 신뢰도
색상: 정답 여부
  초록: 양쪽 정답
  파랑: DINOv2만 정답
  주황: EVA만 정답
  빨강: 양쪽 오답
```

이 산점도로 **두 모델이 어떤 영역에서 상보적인지** 직관적으로 파악할 수 있다.

### 상보성(Complementarity) 분석

```
양쪽 정답:   XXX개 (XX%)  → 동의하는 쉬운 샘플
DINO만 정답: XX개 (X%)    → EVA가 놓치는 패턴
EVA만 정답:  XX개 (X%)    → DINOv2가 놓치는 패턴
양쪽 오답:   XX개 (X%)    → 앙상블로도 해결 불가
```

**"DINO만 정답" + "EVA만 정답"** 의 합이 클수록 앙상블 효과가 크다.

---

## 10. 제출 파일 생성 (Cell 25)

### 출력 형태

```csv
id,unstable_prob,stable_prob
TEST_001,0.12345678,0.87654322
TEST_002,0.98765432,0.01234568
...
```

### 생성되는 제출 파일 목록

| 파일명 | 전략 |
|--------|------|
| `submission_dino_eva_w{α}_tta_{timestamp}.csv` | 가중 평균 |
| `submission_dino_eva_lr_stack_tta_{timestamp}.csv` | LR 스태킹 |
| `submission_dino_eva_xgb_stack_tta_{timestamp}.csv` | XGB 스태킹 |
| `submission_dinov2_5fold_tta_{timestamp}.csv` | DINOv2 단독 (비교용) |
| `submission_eva_giant_5fold_tta_{timestamp}.csv` | EVA 단독 (비교용) |

### 검증

```python
assert not np.isnan(p).any()           # NaN 없음
assert np.allclose(p.sum(axis=1), 1.0) # 확률 합 = 1
```

---

## 11. 시각화 및 분석 (Cell 26~28)

### Cell 26: 성능 비교 시각화

4개 차트:

1. **히스토그램 오버레이**: P(unstable) 확률 분포 비교 (DINOv2 / EVA / 앙상블)
2. **산점도**: DINOv2 vs EVA 예측 — 대각선에서의 이탈이 불일치 영역
3. **메트릭 바 차트**: LogLoss, Accuracy, Unstable Recall, Stable Recall 비교
4. **상보성 바 차트**: 양쪽 정답/DINO만/EVA만/양쪽 오답

### Cell 27: Attention Rollout 비교

**Attention Rollout**이란?

ViT의 각 layer에서 attention weight를 추출하여, CLS 토큰이 **이미지의 어느 부분에 집중하는지** 시각화하는 기법.

```python
def attention_rollout(attns):
    # 모든 레이어의 attention을 곱하여 최종 attention map 생성
    result = I  # 단위 행렬
    for attn in attns:
        a = 0.5 * attn.mean(heads) + 0.5 * I  # residual 반영
        result = a @ result
    return result[0, CLS, patches]  # CLS → 패치 attention
```

**시각화 대상**: Unstable / Borderline / Stable 각 그룹에서 3개씩 선택

**분석 포인트**:
- DINOv2: 건물 구조 영역에 집중 (밝은 부분)
- EVA: 세부 불안정 요소 + 배경까지 확산 (과적합 위험)
- 앙상블: DINOv2의 구조물 포커스 + EVA의 분류 정밀도

### Cell 28: 오류 분석

#### Confusion Matrix

DINOv2와 EVA 각각의 confusion matrix를 나란히 비교.

#### 에러 유형 분석

```
DINO만 틀림: FN이 많음 → unstable을 stable로 오분류 (놓침)
EVA만 틀림:  FP가 많음 → stable을 unstable로 오분류 (과민)
양쪽 틀림:   추가 모델/데이터 필요
```

#### 에러 샘플 이미지 시각화

실제로 틀린 샘플의 front/top 이미지를 보여주며, 각 모델의 예측 확률도 표시.

---

## 12. Drive 동기화 (Cell 29)

### 동기화 대상

1. DINOv2 best 가중치 → Drive
2. EVA-Giant best 가중치 → Drive
3. 메타러너 (`meta_learner.pkl`) → Drive
4. 학습 로그 (`*_log.csv`) → Drive

### 최종 현황 출력

각 백본의 완료 fold 수, 제출 파일 수, 디스크 여유 공간.

---

## 13. 핵심 기법 정리

### 1. Stratified K-Fold with merge_dev

```
train.csv (400개) + dev.csv (100개) = 500개
→ StratifiedKFold(n_splits=5) → 각 fold 400/100
```

dev 데이터도 학습에 활용하여 데이터 효율 극대화. Stratified로 클래스 비율 유지.

### 2. Layer-wise Learning Rate Decay (LLRD)

사전학습 모델의 fine-tuning에서 **거의 필수적인 기법**. 하위 레이어의 일반적 특징을 보존하면서 상위 레이어만 적극 조정.

### 3. Warmup + Cosine Annealing

```
Epoch 1~2: LR 선형 증가 (0 → target)  ← Warmup
Epoch 3~N: Cosine 감소 (target → 0)   ← Annealing
```

Warmup은 초기 학습의 불안정성을 방지. Cosine은 후반부에 갈수록 섬세하게 수렴.

### 4. Gradient Checkpointing

```
일반: 전체 중간 활성화값 저장 → VRAM 많이 사용
Checkpointing: 일부만 저장, 나머지는 역전파 시 재계산 → VRAM 절약 (속도 약간 감소)
```

EVA-Giant 1.0B를 A100 40GB에 올리려면 필수.

### 5. Out-Of-Fold Stacking

일반 앙상블(단순 평균)과 달리, 각 모델의 OOF 예측을 **메타 특징**으로 사용하여 메타러너를 학습. **모델 간 상호작용을 학습**할 수 있다.

### 6. Test-Time Augmentation (TTA)

테스트 시 여러 변환을 적용하여 예측의 분산을 줄인다. 특히 건물 이미지는 crop 위치나 brightness에 따라 예측이 달라질 수 있으므로 효과적.

---

## 14. 하이퍼파라미터 비교표

| 항목 | DINOv2-Large | EVA-Giant | 이유 |
|------|:---:|:---:|------|
| 파라미터 수 | 304M | 1.0B | — |
| Head LR | 1e-4 | 1e-4 | 동일 |
| Backbone LR | 1e-5 | 1e-5 | 동일 |
| Layer Decay | **0.75** | **0.9** | DINOv2: self-sup 보존 / EVA: supervised 전체 조정 |
| Head Type | **cross_attn** | **simple** | DINOv2: 작아서 fusion 필요 / EVA: 충분한 표현력 |
| Weight Decay | 0.05 | 0.05 | 동일 |
| Dropout | 0.3 | 0.3 | 동일 |
| Epochs | 20 | 20 | 동일 |
| Patience | **10** | **15** | EVA 수렴 느림 |
| Warmup | 2 | 2 | 동일 |
| Scheduler | cosine | cosine | 동일 |
| Batch Size | 16 | 16 | A100 기준 |

---

## 15. 개선 포인트 및 회고

### 현재 구조의 강점

1. **상보적 모델 조합**: DINOv2(구조 인식) + EVA(세밀 분류)
2. **메타러너**: 단순 평균보다 지능적인 결합
3. **OOF 기반 최적화**: 과적합 없는 공정한 가중치 학습
4. **종합 시각화**: Attention Rollout 비교로 모델 행동 이해

### 알려진 문제

1. **Colab 구버전 파일 이슈**: Drive에 있는 models.py가 구버전이면 `CrossViewFusion` 관련 에러 발생. Cell 3 패치가 train_v2.py는 잡아주지만, models.py의 클래스 추가는 자동화하기 어려움.
   - **해결**: 최신 models.py를 Drive에 직접 업로드
   
2. **EVA-Giant VRAM**: A100 40GB에서도 batch_size=16이 한계. 80GB라면 32까지 가능.

### 가능한 개선 방향

1. **3모델 앙상블**: ConvNeXt-V2 등 CNN 계열 추가 시 ViT와의 상보성 극대화
2. **Mixup / CutMix**: 현재 비활성 (`no_mixup=True`) — 데이터가 적을 때 효과적일 수 있음
3. **Label Smoothing**: 경계 샘플의 과확신 방지
4. **Pseudo Labeling**: 테스트 데이터의 고확신 예측을 학습에 재활용
5. **ShapeStacks 합성 데이터**: 이미 데이터 디렉토리에 있으나 현재 미활용

---

## 부록: 셀 번호 / 역할 빠른 참조

| Cell | 역할 | 키워드 |
|:----:|------|--------|
| 1 | 환경 설정 (Drive + 패키지) | `drive.mount`, `pip install` |
| 2 | 소스 복사 + 체크포인트 | `shutil.copy2`, `sync_best_to_drive` |
| 3 | 경쟁 모드 패치 | `train_v2.py 검증`, `증강 강화`, `TTA 확장` |
| 4 | 데이터 압축 해제 | `zipfile`, `로컬 SSD` |
| 5 | 환경 검증 | `파일 존재 확인`, `fold 현황` |
| 6 | DINOv2 설정 + 학습 함수 | `DINO_CONFIG`, `cross_attn`, `run_finetune_fold` |
| 7 | DINOv2 Pretrain (선택) | `pretrain`, `기본 스킵` |
| 8~12 | DINOv2 Fold 0~4 | `run_finetune_fold` × 5 |
| 13 | DINOv2 학습 곡선 | `plot_learning_curves`, `과적합 분석` |
| 14 | EVA-Giant 설정 | `EVA_CONFIG`, `simple head`, `layer_decay 0.9` |
| 15~19 | EVA Fold 0~4 | `run_finetune_fold` × 5 |
| 20 | EVA 학습 곡선 | `plot_learning_curves` |
| 21 | 양쪽 모델 비교 | `evaluate_model`, `per-class accuracy` |
| 22 | Stacking Meta-Learner | `OOF`, `LogisticRegression`, `XGBoost` |
| 23 | 2-Stage Ensemble 추론 | `predict_tta`, `메타러너 적용` |
| 24 | 앙상블 가중치 최적화 | `0.01 단위 탐색`, `상보성 분석` |
| 25 | 제출 파일 생성 | `5가지 전략`, `검증` |
| 26 | 성능 비교 시각화 | `히스토그램`, `산점도`, `메트릭 바` |
| 27 | Attention Rollout | `attention_rollout`, `DINOv2 vs EVA` |
| 28 | 오류 분석 | `Confusion Matrix`, `에러 패턴`, `이미지 시각화` |
| 29 | Drive 동기화 | `sync`, `최종 요약` |
