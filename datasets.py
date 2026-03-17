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


def build_nested_crops(image, num_crops=1):
    if num_crops <= 1:
        return [image]

    h, w = image.shape[:2]
    crops = [image]
    crop_specs = [0.9, 0.78, 0.66, 0.54]
    for scale in crop_specs[: max(0, num_crops - 1)]:
        crop_h = max(16, int(h * scale))
        crop_w = max(16, int(w * scale))
        start_y = max(0, (h - crop_h) // 2)
        start_x = max(0, (w - crop_w) // 2)
        crops.append(image[start_y:start_y + crop_h, start_x:start_x + crop_w])
    return crops


def apply_transform_to_crops(image, transform, num_crops=1):
    crops = build_nested_crops(image, num_crops=num_crops)
    if transform is None:
        if num_crops <= 1:
            return image

        base_h, base_w = image.shape[:2]
        transformed = []
        for crop in crops:
            resized = cv2.resize(crop, (base_w, base_h), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            transformed.append(tensor)
        return torch.stack(transformed, dim=0)

    transformed = []
    for crop in crops:
        transformed.append(transform(image=crop)["image"])

    if num_crops <= 1:
        return transformed[0]
    return torch.stack(transformed, dim=0)


def load_video_tensor(video_path, transform, num_video_frames=3):
    raw_frames = extract_video_frames(video_path, num_video_frames)
    if not raw_frames:
        return None

    if transform:
        return torch.stack([transform(image=frame)["image"] for frame in raw_frames], dim=0)

    return torch.stack([
        torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        for frame in raw_frames
    ], dim=0)


# ---------------------------------------------------------------------------
# Augmentation 설정
# ---------------------------------------------------------------------------
def get_train_transforms(img_size=384):
    """학습용 강력한 augmentation (도메인 shift 대응)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.RandomResizedCrop((img_size, img_size), scale=(0.9, 1.0), ratio=(0.92, 1.08), p=0.35),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=(-0.04, 0.04), scale=(0.95, 1.05), rotate=(-8, 8), p=0.4),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=1.0),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=12, val_shift_limit=12, p=1.0),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, hue=0.04, p=1.0),
        ], p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.15),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
            A.ISONoise(p=1.0),
        ], p=0.15),
        A.RandomShadow(p=0.1),
        A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.1),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(12, 24), hole_width_range=(12, 24), p=0.15),
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
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
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
                 use_video_frames=False, num_video_frames=3, num_crops=1):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.transform = transform
        self.is_test = is_test
        self.use_video_frames = use_video_frames
        self.num_video_frames = num_video_frames
        self.num_crops = num_crops

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

        front = apply_transform_to_crops(front, self.transform, self.num_crops)
        top = apply_transform_to_crops(top, self.transform, self.num_crops)

        # 비디오 프레임 사용 시
        video_frames = None
        video_mask = None
        if self.use_video_frames:
            video_path = os.path.join(sample_dir, "simulation.mp4")
            video_frames = load_video_tensor(video_path, self.transform, self.num_video_frames)
            if video_frames is None:
                if isinstance(front, torch.Tensor) and front.dim() == 4:
                    c, h, w = front.shape[1:]
                else:
                    c, h, w = front.shape
                video_frames = torch.zeros((self.num_video_frames, c, h, w), dtype=front.dtype)
                video_mask = torch.tensor(0.0, dtype=torch.float32)
            else:
                video_mask = torch.tensor(1.0, dtype=torch.float32)

        if self.is_test:
            if self.use_video_frames:
                return front, top, video_frames, video_mask, sample_id
            return front, top, sample_id

        label = 1 if row["label"] == "unstable" else 0
        if self.use_video_frames:
            return front, top, video_frames, video_mask, torch.tensor(label, dtype=torch.long)
        return front, top, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Archive 블록 데이터셋 (single-view)
# ---------------------------------------------------------------------------
class ArchiveBlockDataset(Dataset):
    """Kaggle archive 블록 이미지 데이터셋 (pretrain용)"""

    def __init__(self, csv_path, img_dir, transform=None, max_samples=None, num_crops=1):
        self.df = pd.read_csv(csv_path)
        if max_samples:
            self.df = self.df.sample(n=min(max_samples, len(self.df)),
                                     random_state=42).reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.num_crops = num_crops

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

        img = apply_transform_to_crops(img, self.transform, self.num_crops)

        return img, img, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# ShapeStacks 데이터셋
# ---------------------------------------------------------------------------
class ShapeStacksDataset(Dataset):
    """ShapeStacks 외부 데이터셋 (pretrain & finetune용)"""

    def __init__(self, data_root, split="train", transform=None, max_samples=None, num_crops=1):
        self.data_root = data_root
        self.transform = transform
        self.num_crops = num_crops
        self.samples = []  # (img_path, label)

        # 실제 구조: data_root/shapestacks/recordings/<scenario>/*.png
        inner_root = os.path.join(data_root, "shapestacks")
        recordings_dir = os.path.join(inner_root, "recordings")
        splits_file = os.path.join(inner_root, "splits", "default", f"{split}.txt")

        if not os.path.exists(recordings_dir):
            print(f"[WARNING] ShapeStacks recordings not found at {recordings_dir}")
            return

        # split 파일이 있으면 해당 시나리오만 사용
        scenario_filter = None
        if os.path.exists(splits_file):
            with open(splits_file, "r") as f:
                scenario_filter = set(line.strip() for line in f if line.strip())

        # 시나리오 폴더 순회, 폴더명에서 라벨 파싱
        # vcom=0 AND vpsf=0 → stable(0), otherwise unstable(1)
        for scenario_name in os.listdir(recordings_dir):
            scenario_path = os.path.join(recordings_dir, scenario_name)
            if not os.path.isdir(scenario_path):
                continue
            if scenario_filter and scenario_name not in scenario_filter:
                continue

            # 라벨 파싱
            label = 0  # default stable
            parts = scenario_name.split("-")
            for part in parts:
                if part.startswith("vcom=") or part.startswith("vpsf="):
                    val = int(part.split("=")[1])
                    if val > 0:
                        label = 1  # unstable
                        break

            # 해당 시나리오의 모든 PNG 이미지 수집 (카메라 앵글별)
            for img_name in os.listdir(scenario_path):
                if img_name.endswith(".png"):
                    self.samples.append(
                        (os.path.join(scenario_path, img_name), label)
                    )

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

        img = apply_transform_to_crops(img, self.transform, self.num_crops)

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
