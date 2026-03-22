"""
데이터셋 모듈 v2
- 경쟁급 최대 증강 (Perspective, Grayscale, CutMix, Mixup 등)
- ShapeStacks h=6 전용 Dual-View 페어링
- 비디오 프레임 추출
- Dev 3× 오버샘플 옵션
"""
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# =========================================================================
# Augmentation
# =========================================================================
def get_train_transforms(img_size=384):
    """경쟁급 최대 증강"""
    return A.Compose([
        A.RandomResizedCrop(
            (img_size, img_size), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=(-0.05, 0.05), scale=(0.93, 1.07),
            rotate=(-10, 10), p=0.5),
        A.Perspective(scale=(0.02, 0.06), p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=15, p=1.0),
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05, p=1.0),
        ], p=0.8),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.2),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.10), p=1.0),
            A.ISONoise(p=1.0),
        ], p=0.2),
        A.RandomShadow(p=0.15),
        A.RandomFog(fog_coef_range=(0.1, 0.35), p=0.1),
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(16, 40), hole_width_range=(16, 40), p=0.25),
        A.ToGray(p=0.05),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_train_transforms_simple(img_size=384):
    """단순 증강 — ColorJitter 강하게, 나머지 최소화"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=15,
            border_mode=0, p=0.6),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=384):
    crop = int(img_size * 0.9)
    return A.Compose([
        A.CenterCrop(crop, crop),
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size=384):
    norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    crop = int(img_size * 0.9)
    return [
        # 0: base (center crop)
        A.Compose([A.CenterCrop(crop, crop),
                    A.Resize(img_size, img_size), A.Normalize(**norm), ToTensorV2()]),
        # 1: hflip
        A.Compose([A.CenterCrop(crop, crop),
                    A.Resize(img_size, img_size), A.HorizontalFlip(p=1.0),
                    A.Normalize(**norm), ToTensorV2()]),
        # 2: brightness
        A.Compose([A.CenterCrop(crop, crop),
                    A.Resize(img_size, img_size),
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
                    A.Normalize(**norm), ToTensorV2()]),
        # 3: tighter crop (90% of 90% = 81%)
        A.Compose([A.CenterCrop(int(img_size * 0.9), int(img_size * 0.9)),
                    A.Resize(img_size, img_size), A.Normalize(**norm), ToTensorV2()]),
    ]


# =========================================================================
# Video
# =========================================================================
def extract_video_frames(video_path, num_frames=5):
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


def load_video_tensor(video_path, transform, num_frames=5):
    frames = extract_video_frames(video_path, num_frames)
    if not frames:
        return None
    if transform:
        return torch.stack([transform(image=f)["image"] for f in frames])
    return torch.stack([torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frames])


# =========================================================================
# CutMix / Mixup
# =========================================================================
def _rand_bbox(W, H, lam):
    cut_rat = np.sqrt(1.0 - lam)
    cw, ch = max(1, int(W * cut_rat)), max(1, int(H * cut_rat))
    cx, cy = np.random.randint(W), np.random.randint(H)
    return (np.clip(cx - cw // 2, 0, W), np.clip(cy - ch // 2, 0, H),
            np.clip(cx + cw // 2, 0, W), np.clip(cy + ch // 2, 0, H))


def cutmix_data(front, top, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(front.size(0), device=front.device)
    H, W = front.size(-2), front.size(-1)
    x1, y1, x2, y2 = _rand_bbox(W, H, lam)
    front[:, :, y1:y2, x1:x2] = front[idx, :, y1:y2, x1:x2]
    top[:, :, y1:y2, x1:x2] = top[idx, :, y1:y2, x1:x2]
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
    return front, top, labels, labels[idx], lam


def mixup_data(front, top, labels, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(front.size(0), device=front.device)
    return (lam * front + (1 - lam) * front[idx],
            lam * top + (1 - lam) * top[idx],
            labels, labels[idx], lam)


# =========================================================================
# Dacon Dual-View Dataset
# =========================================================================
class DaconDualViewDataset(Dataset):
    """Dacon 대회 데이터 (front + top + optional video)"""

    def __init__(self, samples, transform=None, use_video=False, num_video_frames=5):
        self.samples = samples   # [(data_dir, sample_id, label_int), ...]
        self.transform = transform
        self.use_video = use_video
        self.num_video_frames = num_video_frames

    def __len__(self):
        return len(self.samples)

    def _load_img(self, path):
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((384, 384, 3), dtype=np.uint8)

    def __getitem__(self, idx):
        data_dir, sid, label = self.samples[idx]
        d = os.path.join(data_dir, sid)
        front = self._load_img(os.path.join(d, "front.png"))
        top = self._load_img(os.path.join(d, "top.png"))

        if self.transform:
            front = self.transform(image=front)["image"]
            top = self.transform(image=top)["image"]
        else:
            front = torch.from_numpy(front).permute(2, 0, 1).float() / 255.0
            top = torch.from_numpy(top).permute(2, 0, 1).float() / 255.0

        label_t = torch.tensor(label, dtype=torch.long)

        if not self.use_video:
            return front, top, label_t

        vid_path = os.path.join(d, "simulation.mp4")
        video = load_video_tensor(vid_path, self.transform, self.num_video_frames)
        if video is None:
            c, h, w = front.shape
            video = torch.zeros(self.num_video_frames, c, h, w, dtype=front.dtype)
            mask = torch.tensor(0.0)
        else:
            mask = torch.tensor(1.0)
        return front, top, video, mask, label_t


# =========================================================================
# ShapeStacks h=6 Dual-View Dataset (paired camera angles)
# =========================================================================
class ShapeStacksH6Dataset(Dataset):
    """ShapeStacks h=6 전용 — 같은 시나리오의 두 카메라 앵글을 front/top으로 페어링"""

    def __init__(self, data_root, split=None, transform=None,
                 max_samples=None, pairs_per_scenario=4):
        self.transform = transform
        self.scenarios = []       # [(label, [img_path, ...]), ...]
        self.pairs_per = pairs_per_scenario

        inner = os.path.join(data_root, "shapestacks")
        rec_dir = os.path.join(inner, "recordings")

        if not os.path.exists(rec_dir):
            print(f"[WARN] ShapeStacks recordings not found: {rec_dir}")
            return

        scenario_filter = None
        if split:
            sf = os.path.join(inner, "splits", "default", f"{split}.txt")
            if os.path.exists(sf):
                with open(sf) as f:
                    scenario_filter = set(l.strip() for l in f if l.strip())

        for name in sorted(os.listdir(rec_dir)):
            if "h=6" not in name:
                continue
            path = os.path.join(rec_dir, name)
            if not os.path.isdir(path):
                continue
            if scenario_filter and name not in scenario_filter:
                continue

            label = 0
            for p in name.split("-"):
                if p.startswith("vcom=") or p.startswith("vpsf="):
                    if int(p.split("=")[1]) > 0:
                        label = 1
                        break

            imgs = sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".png"))
            if len(imgs) >= 2:
                self.scenarios.append((label, imgs))

        if max_samples and len(self.scenarios) > max_samples:
            random.shuffle(self.scenarios)
            self.scenarios = self.scenarios[:max_samples]

        stable = sum(1 for l, _ in self.scenarios if l == 0)
        unstable = len(self.scenarios) - stable
        print(f"[ShapeStacks h=6] {len(self.scenarios)} scenarios × {self.pairs_per} "
              f"= {len(self)} items  (stable={stable}, unstable={unstable})")

    def __len__(self):
        return len(self.scenarios) * self.pairs_per

    def _load_img(self, path):
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((224, 224, 3), dtype=np.uint8)

    def __getitem__(self, idx):
        sc_idx = idx % len(self.scenarios)
        label, imgs = self.scenarios[sc_idx]
        i, j = random.sample(range(len(imgs)), 2)
        front = self._load_img(imgs[i])
        top = self._load_img(imgs[j])

        if self.transform:
            front = self.transform(image=front)["image"]
            top = self.transform(image=top)["image"]
        else:
            front = torch.from_numpy(front).permute(2, 0, 1).float() / 255.0
            top = torch.from_numpy(top).permute(2, 0, 1).float() / 255.0

        return front, top, torch.tensor(label, dtype=torch.long)


# =========================================================================
# Test Dataset
# =========================================================================
class TestDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sid = self.df.iloc[idx]["id"]
        d = os.path.join(self.data_dir, sid)
        front = cv2.cvtColor(cv2.imread(os.path.join(d, "front.png")), cv2.COLOR_BGR2RGB)
        top = cv2.cvtColor(cv2.imread(os.path.join(d, "top.png")), cv2.COLOR_BGR2RGB)
        if self.transform:
            front = self.transform(image=front)["image"]
            top = self.transform(image=top)["image"]
        else:
            front = torch.from_numpy(front).permute(2, 0, 1).float() / 255.0
            top = torch.from_numpy(top).permute(2, 0, 1).float() / 255.0
        return front, top, sid


# =========================================================================
# Combined Dataset
# =========================================================================
class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumlen = []
        total = 0
        for d in datasets:
            self.cumlen.append(total)
            total += len(d)
        self.total = total

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        for start, ds in zip(self.cumlen, self.datasets):
            if idx < start + len(ds):
                return ds[idx - start]
        raise IndexError(idx)
