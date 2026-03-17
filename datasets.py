"""
데이터셋 모듈: Dacon 대회 데이터 + ShapeStacks + Archive 외부 데이터
"""
import os
import glob
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# Augmentation 설정
# ---------------------------------------------------------------------------
def get_train_transforms(img_size=384):
    """학습용 강력한 augmentation (도메인 shift 대응)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=30, p=1.0),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
        ], p=0.8),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(p=1.0),
        ], p=0.3),
        A.RandomShadow(p=0.2),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=384):
    """검증/추론용 transform"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size=384):
    """TTA (Test-Time Augmentation) transform 리스트"""
    base_norm = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    hflip = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    bright = A.Compose([
        A.Resize(img_size, img_size),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return [base_norm, hflip, bright]


# ---------------------------------------------------------------------------
# 시뮬레이션 비디오에서 프레임 추출
# ---------------------------------------------------------------------------
def extract_video_frames(video_path, num_frames=5):
    """시뮬레이션 mp4에서 균등 간격 프레임 추출"""
    frames = []
    if not os.path.exists(video_path):
        return frames
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return frames
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Dacon 대회 데이터셋 (Dual-View: front + top)
# ---------------------------------------------------------------------------
class DaconDualViewDataset(Dataset):
    """front.png + top.png를 concat하여 입력으로 사용하는 데이터셋"""

    def __init__(self, csv_path, data_dir, transform=None, is_test=False,
                 use_video_frames=False, num_video_frames=3):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.transform = transform
        self.is_test = is_test
        self.use_video_frames = use_video_frames
        self.num_video_frames = num_video_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["id"]
        sample_dir = os.path.join(self.data_dir, sample_id)

        # front + top 이미지 로드
        front = cv2.imread(os.path.join(sample_dir, "front.png"))
        front = cv2.cvtColor(front, cv2.COLOR_BGR2RGB)
        top = cv2.imread(os.path.join(sample_dir, "top.png"))
        top = cv2.cvtColor(top, cv2.COLOR_BGR2RGB)

        if self.transform:
            front = self.transform(image=front)["image"]
            top = self.transform(image=top)["image"]

        # 비디오 프레임 사용 시
        video_frames = None
        if self.use_video_frames:
            video_path = os.path.join(sample_dir, "simulation.mp4")
            raw_frames = extract_video_frames(video_path, self.num_video_frames)
            if raw_frames and self.transform:
                video_frames = torch.stack([
                    self.transform(image=f)["image"] for f in raw_frames
                ])
            elif raw_frames:
                video_frames = torch.stack([
                    torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
                    for f in raw_frames
                ])

        if self.is_test:
            if video_frames is not None:
                return front, top, video_frames, sample_id
            return front, top, sample_id

        label = 1 if row["label"] == "unstable" else 0
        if video_frames is not None:
            return front, top, video_frames, torch.tensor(label, dtype=torch.long)
        return front, top, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Archive 블록 데이터셋 (single-view)
# ---------------------------------------------------------------------------
class ArchiveBlockDataset(Dataset):
    """Kaggle archive 블록 이미지 데이터셋 (pretrain용)"""

    def __init__(self, csv_path, img_dir, transform=None, max_samples=None):
        self.df = pd.read_csv(csv_path)
        if max_samples:
            self.df = self.df.sample(n=min(max_samples, len(self.df)),
                                     random_state=42).reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id"]
        label = row["stable"]  # 1=stable, 0=unstable
        label = 1 - label  # 변환: 1=unstable, 0=stable (대회 기준)

        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((384, 384, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, img, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# ShapeStacks 데이터셋
# ---------------------------------------------------------------------------
class ShapeStacksDataset(Dataset):
    """ShapeStacks 외부 데이터셋 (pretrain & finetune용)"""

    def __init__(self, data_root, split="train", transform=None, max_samples=None):
        self.data_root = data_root
        self.transform = transform
        self.samples = []  # (img_path, label)

        meta_root = os.path.join(data_root, "shapestacks-meta")
        rgb_root = os.path.join(data_root, "shapestacks-rgb")

        if not os.path.exists(meta_root) or not os.path.exists(rgb_root):
            print(f"[WARNING] ShapeStacks data not found at {data_root}")
            return

        # 메타데이터에서 안정/불안정 라벨 로드
        label_dirs = {
            "stable": glob.glob(os.path.join(meta_root, "**", "*stable*"), recursive=True),
            "unstable": glob.glob(os.path.join(meta_root, "**", "*unstable*"), recursive=True),
        }

        # RGB 이미지 검색
        all_imgs = glob.glob(os.path.join(rgb_root, "**", "*.png"), recursive=True)
        if not all_imgs:
            all_imgs = glob.glob(os.path.join(rgb_root, "**", "*.jpg"), recursive=True)

        # 디렉토리명에서 stable/unstable 구분
        for img_path in all_imgs:
            dirname = os.path.dirname(img_path).lower()
            if "unstable" in dirname or "collapse" in dirname:
                self.samples.append((img_path, 1))  # unstable
            elif "stable" in dirname:
                self.samples.append((img_path, 0))  # stable
            else:
                # 파일 구조에 따라 기본 분류
                parent_parts = img_path.replace("\\", "/").split("/")
                for part in parent_parts:
                    if "unstable" in part.lower():
                        self.samples.append((img_path, 1))
                        break
                    elif "stable" in part.lower():
                        self.samples.append((img_path, 0))
                        break

        if max_samples and len(self.samples) > max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        print(f"[ShapeStacks] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((384, 384, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, img, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# 합성 데이터셋 (여러 출처 결합)
# ---------------------------------------------------------------------------
class CombinedDataset(Dataset):
    """여러 데이터셋을 합쳐서 하나로 사용"""

    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumulative = []
        total = 0
        for l in self.lengths:
            self.cumulative.append(total)
            total += l
        self.total_len = total

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        for i, (start, length) in enumerate(zip(self.cumulative, self.lengths)):
            if idx < start + length:
                return self.datasets[i][idx - start]
        raise IndexError(f"Index {idx} out of range")
