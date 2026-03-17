# dacon-structural-stability

멀티뷰 이미지를 활용한 구조물 안정성 예측 AI 모델

---

## 환경 설정 (Setup)

```powershell
cd c:\Pyg\Projects\dacon\dacon-structural-stability

# 1) 가상환경 생성
python -m venv venv

# 2) 가상환경 활성화
.\venv\Scripts\activate

# 3) pip 업그레이드
python -m pip install --upgrade pip

# 4) PyTorch 설치 (CUDA 12.8, RTX 5060 지원)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 5) 나머지 패키지 설치
pip install -r requirements.txt
```

---

## 외부 데이터 다운로드 (ShapeStacks ~33GB)

```powershell
python download_shapestacks.py
```

---

## 학습 (Train)

### Stage 1: Pretrain (외부 데이터)

```powershell
python train.py --model efficientnet --stage pretrain --pretrain_epochs 15
python train.py --model convnext --stage pretrain --pretrain_epochs 15
python train.py --model swinv2 --stage pretrain --pretrain_epochs 15
```

### Stage 2: Finetune (Dacon 데이터, 5-Fold)

```powershell
python train.py --model efficientnet --stage finetune --finetune_epochs 50 --n_folds 5
python train.py --model convnext --stage finetune --finetune_epochs 50 --n_folds 5
python train.py --model swinv2 --stage finetune --finetune_epochs 50 --n_folds 5
```

### 한 번에 Pretrain + Finetune

```powershell
python train.py --model efficientnet --stage both --pretrain_epochs 15 --finetune_epochs 50
```

---

## 추론 (Inference)

### Dev 성능 검증

```powershell
python inference.py --validate
```

### 최종 제출 파일 생성 (앙상블 + TTA)

```powershell
python inference.py --tta --temperature 1.0
```

### 단일 모델만 추론

```powershell
python inference.py --models efficientnet
```

결과물: `submissions/` 폴더에 CSV 파일 생성

---

## 전체 파이프라인 한 번에 실행

```powershell
# 가상환경 활성화 후 순서대로 실행
.\venv\Scripts\activate

python download_shapestacks.py

python train.py --model efficientnet --stage pretrain --pretrain_epochs 15
python train.py --model convnext --stage pretrain --pretrain_epochs 15
python train.py --model swinv2 --stage pretrain --pretrain_epochs 15

python train.py --model efficientnet --stage finetune --finetune_epochs 50 --n_folds 5
python train.py --model convnext --stage finetune --finetune_epochs 50 --n_folds 5
python train.py --model swinv2 --stage finetune --finetune_epochs 50 --n_folds 5

python inference.py --validate
python inference.py --tta --temperature 1.0
```
