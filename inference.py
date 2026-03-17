"""
추론 및 앙상블 스크립트
- 3개 모델 x 5-Fold = 15개 체크포인트 앙상블
- TTA (Test-Time Augmentation) 적용
- 최종 submission.csv 생성

사용법:
  python inference.py                           # 전체 앙상블
  python inference.py --model efficientnet       # 단일 모델
  python inference.py --tta                      # TTA 활성화
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets import get_val_transforms, get_tta_transforms
from models import build_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "submissions")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class TestDualViewDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["id"]
        sample_dir = os.path.join(self.data_dir, sample_id)

        front = cv2.imread(os.path.join(sample_dir, "front.png"))
        front = cv2.cvtColor(front, cv2.COLOR_BGR2RGB)
        top = cv2.imread(os.path.join(sample_dir, "top.png"))
        top = cv2.cvtColor(top, cv2.COLOR_BGR2RGB)

        if self.transform:
            front = self.transform(image=front)["image"]
            top = self.transform(image=top)["image"]

        return front, top, sample_id


def get_model_img_size(model_type):
    sizes = {"efficientnet": 384, "convnext": 384, "swinv2": 256}
    return sizes[model_type]


@torch.no_grad()
def predict_single_model(model, loader, device):
    model.eval()
    all_probs = []
    all_ids = []

    for front, top, ids in tqdm(loader, desc="Predict", leave=False):
        front = front.to(device)
        top = top.to(device)

        with autocast():
            logits = model(front, top)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_ids.extend(ids)

    return np.concatenate(all_probs, axis=0), all_ids


@torch.no_grad()
def predict_with_tta(model, test_csv, test_dir, model_type, device, batch_size=8):
    """TTA 적용 추론"""
    model.eval()
    img_size = get_model_img_size(model_type)
    tta_transforms = get_tta_transforms(img_size)

    all_probs_list = []
    all_ids = None

    for tta_tf in tta_transforms:
        ds = TestDualViewDataset(test_csv, test_dir, transform=tta_tf)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
        probs, ids = predict_single_model(model, loader, device)
        all_probs_list.append(probs)
        if all_ids is None:
            all_ids = ids

    avg_probs = np.mean(all_probs_list, axis=0)
    return avg_probs, all_ids


def load_model_weights(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    return model


def ensemble_predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_csv = os.path.join(DATA_DIR, "open", "sample_submission.csv")
    test_dir = os.path.join(DATA_DIR, "open", "test")

    models_to_use = args.models if args.models else ["efficientnet", "convnext", "swinv2"]
    all_predictions = []

    for model_type in models_to_use:
        img_size = get_model_img_size(model_type)
        config = {"efficientnet": 8, "convnext": 4, "swinv2": 8}
        batch_size = config.get(model_type, 8)

        # K-Fold 체크포인트 수집
        fold_paths = []
        for fold in range(args.n_folds):
            path = os.path.join(SAVE_DIR, f"{model_type}_fold{fold}.pth")
            if os.path.exists(path):
                fold_paths.append(path)

        if not fold_paths:
            print(f"[WARN] No checkpoints for {model_type}, skipping.")
            continue

        print(f"\n{'=' * 40}")
        print(f" {model_type}: {len(fold_paths)} fold(s)")
        print(f"{'=' * 40}")

        for ckpt_path in fold_paths:
            print(f"  Loading: {os.path.basename(ckpt_path)}")
            model = build_model(model_type, pretrained=False, num_classes=2, drop_rate=0.0)
            model = load_model_weights(model, ckpt_path, device)
            model = model.to(device)

            if args.tta:
                probs, ids = predict_with_tta(model, test_csv, test_dir,
                                               model_type, device, batch_size)
            else:
                val_tf = get_val_transforms(img_size)
                ds = TestDualViewDataset(test_csv, test_dir, transform=val_tf)
                loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                    num_workers=4, pin_memory=True)
                probs, ids = predict_single_model(model, loader, device)

            all_predictions.append(probs)
            del model
            torch.cuda.empty_cache()

    if not all_predictions:
        print("[ERROR] No predictions generated!")
        return

    # 앙상블 (평균)
    ensemble_probs = np.mean(all_predictions, axis=0)

    # Temperature scaling (캘리브레이션)
    if args.temperature != 1.0:
        logits = np.log(ensemble_probs + 1e-10)
        logits /= args.temperature
        exp_logits = np.exp(logits)
        ensemble_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # Submission 생성
    sub_df = pd.read_csv(test_csv)
    sub_df["unstable_prob"] = ensemble_probs[:, 1]  # class 1 = unstable
    sub_df["stable_prob"] = ensemble_probs[:, 0]    # class 0 = stable

    # Probability normalization
    row_sums = sub_df[["unstable_prob", "stable_prob"]].sum(axis=1)
    sub_df["unstable_prob"] /= row_sums
    sub_df["stable_prob"] /= row_sums

    # Clipping for numerical stability
    eps = 1e-7
    sub_df["unstable_prob"] = sub_df["unstable_prob"].clip(eps, 1 - eps)
    sub_df["stable_prob"] = sub_df["stable_prob"].clip(eps, 1 - eps)

    output_name = f"submission_{'_'.join(models_to_use)}"
    if args.tta:
        output_name += "_tta"
    output_path = os.path.join(OUTPUT_DIR, f"{output_name}.csv")
    sub_df[["id", "unstable_prob", "stable_prob"]].to_csv(output_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"  Submission saved: {output_path}")
    print(f"  Models used: {models_to_use}")
    print(f"  Total predictions ensembled: {len(all_predictions)}")
    print(f"  TTA: {args.tta}")
    print(f"{'=' * 60}")

    # 통계 출력
    print(f"\n  unstable_prob: mean={sub_df['unstable_prob'].mean():.4f}, "
          f"std={sub_df['unstable_prob'].std():.4f}")
    print(f"  stable_prob:   mean={sub_df['stable_prob'].mean():.4f}, "
          f"std={sub_df['stable_prob'].std():.4f}")


def validate_on_dev(args):
    """Dev 셋으로 로컬 성능 검증"""
    from sklearn.metrics import log_loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_csv = os.path.join(DATA_DIR, "open", "dev.csv")
    dev_dir = os.path.join(DATA_DIR, "open", "dev")
    dev_df = pd.read_csv(dev_csv)

    models_to_use = args.models if args.models else ["efficientnet", "convnext", "swinv2"]
    all_predictions = []

    for model_type in models_to_use:
        img_size = get_model_img_size(model_type)
        batch_size = {"efficientnet": 8, "convnext": 4, "swinv2": 8}.get(model_type, 8)

        for fold in range(args.n_folds):
            path = os.path.join(SAVE_DIR, f"{model_type}_fold{fold}.pth")
            if not os.path.exists(path):
                continue

            model = build_model(model_type, pretrained=False, num_classes=2, drop_rate=0.0)
            model = load_model_weights(model, path, device)
            model = model.to(device)

            # dev.csv를 사용해서 추론
            val_tf = get_val_transforms(img_size)

            class DevDataset(Dataset):
                def __init__(self, df, data_dir, transform):
                    self.df = df
                    self.data_dir = data_dir
                    self.transform = transform

                def __len__(self):
                    return len(self.df)

                def __getitem__(self, idx):
                    row = self.df.iloc[idx]
                    sid = row["id"]
                    d = os.path.join(self.data_dir, sid)
                    fr = cv2.imread(os.path.join(d, "front.png"))
                    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    tp = cv2.imread(os.path.join(d, "top.png"))
                    tp = cv2.cvtColor(tp, cv2.COLOR_BGR2RGB)
                    if self.transform:
                        fr = self.transform(image=fr)["image"]
                        tp = self.transform(image=tp)["image"]
                    label = 1 if row["label"] == "unstable" else 0
                    return fr, tp, torch.tensor(label, dtype=torch.long)

            ds = DevDataset(dev_df, dev_dir, val_tf)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)

            model.eval()
            probs_list = []
            with torch.no_grad():
                for front, top, _ in loader:
                    with autocast():
                        logits = model(front.to(device), top.to(device))
                    probs_list.append(F.softmax(logits, dim=1).cpu().numpy())

            all_predictions.append(np.concatenate(probs_list, axis=0))
            del model
            torch.cuda.empty_cache()

    if all_predictions:
        ensemble_probs = np.mean(all_predictions, axis=0)
        true_labels = (dev_df["label"] == "unstable").astype(int).values
        ll = log_loss(true_labels, ensemble_probs, labels=[0, 1])
        acc = (ensemble_probs.argmax(axis=1) == true_labels).mean()
        print(f"\n  Dev LogLoss: {ll:.6f}")
        print(f"  Dev Accuracy: {acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="구조물 안정성 예측 추론")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=["efficientnet", "convnext", "swinv2"])
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--tta", action="store_true", help="TTA 활성화")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature scaling")
    parser.add_argument("--validate", action="store_true",
                        help="Dev 셋으로 로컬 성능 검증")
    args = parser.parse_args()

    if args.validate:
        validate_on_dev(args)
    else:
        ensemble_predict(args)


if __name__ == "__main__":
    main()
