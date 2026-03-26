"""
모델 모듈 v2: 대규모 ViT 기반 Dual-View 구조물 안정성 예측

지원 백본:
  [A100 Tier — Colab]
  - eva_giant     : EVA-Giant      (1.0B, 336px) — ImageNet #1급
  - dinov3_huge   : DINOv3 ViT-H+  (840M, 384px) — 자기지도 SOTA, 공간 구조 최강
  - siglip_so400m : SigLIP SO400M  (428M, 384px) — 대비학습 SOTA
  [Local — 12GB GPU]
  - eva02_large   : EVA02-Large    (305M, 448px) — 상위 솔루션 검증
  - dinov2_large  : DINOv2 ViT-L   (304M, 336px) — 공간 구조 특화
"""
import torch
import torch.nn as nn
import timm

BACKBONE_CONFIGS = {
    "eva_giant": {
        "timm_name": "eva_giant_patch14_336.m30m_ft_in22k_in1k",
        "img_size": 336,
        "tier": "a100",
        "pass_img_size": True,
    },
    "dinov3_huge": {
        "timm_name": "vit_huge_plus_patch16_dinov3.lvd1689m",
        "img_size": 384,
        "tier": "a100",
        "pass_img_size": True,
    },
    "siglip_so400m": {
        "timm_name": "vit_so400m_patch14_siglip_384.webli",
        "img_size": 384,
        "tier": "a100",
        "pass_img_size": True,
    },
    "eva02_large": {
        "timm_name": "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        "img_size": 448,
        "tier": "local",
        "pass_img_size": True,
    },
    "dinov2_large": {
        "timm_name": "vit_large_patch14_reg4_dinov2.lvd142m",
        "img_size": 336,
        "tier": "local",
        "pass_img_size": True,
    },
    "convnext_small": {
        "timm_name": "convnext_small.in12k_ft_in1k_384",
        "img_size": 384,
        "tier": "a100",
        "pass_img_size": False,
    },
}

TRAIN_PRESETS = {
    # backbone_key: (batch_size, lr, pretrain_lr, grad_accum)  effective_bs = batch * accum
    "eva_giant":     (6,  3e-5, 1e-5, 4),   # eff 24
    "dinov3_huge":   (4,  3e-5, 1e-5, 8),   # eff 32
    "siglip_so400m": (12, 5e-5, 2e-5, 2),   # eff 24
    "eva02_large":   (2,  5e-5, 1e-5, 16),  # eff 32
    "dinov2_large":  (2,  5e-5, 1e-5, 16),  # eff 32
    "convnext_small": (32, 2e-4, 5e-5, 1),   # eff 32
}


class DualViewModel(nn.Module):
    """공유 백본 → Fusion → MLP Head
    head_type:
      - "attn_gate": Attention Gate Fusion → deep MLP (기존)
      - "simple":    Concat → BN+MLP (단순 헤드)
    """

    def __init__(self, backbone_key="eva02_large", pretrained=True,
                 num_classes=2, drop_rate=0.3,
                 use_video=False, num_video_frames=5,
                 head_type="attn_gate"):
        super().__init__()
        cfg = BACKBONE_CONFIGS[backbone_key]
        self.backbone_key = backbone_key
        self.use_video = use_video
        self.head_type = head_type

        backbone_kwargs = {
            "pretrained": pretrained,
            "num_classes": 0,
            "drop_rate": drop_rate,
        }
        if cfg.get("pass_img_size", True):
            backbone_kwargs["img_size"] = cfg["img_size"]

        self.backbone = timm.create_model(cfg["timm_name"], **backbone_kwargs)
        feat_dim = self.backbone.num_features
        self.feat_dim = feat_dim

        if head_type == "attn_gate":
            self.attn_gate = nn.Sequential(
                nn.Linear(feat_dim * 2, feat_dim),
                nn.GELU(),
                nn.Linear(feat_dim, 2),
                nn.Softmax(dim=1),
            )
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, 512),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(drop_rate * 0.5),
                nn.Linear(256, num_classes),
            )
        else:
            # concat → simple MLP
            self.attn_gate = None
            self.head = nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(feat_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(drop_rate * 0.5),
                nn.Linear(256, num_classes),
            )

    def encode(self, images):
        if images.dim() == 5:
            B, N, C, H, W = images.shape
            feats = self.backbone(images.view(B * N, C, H, W))
            return feats.view(B, N, -1).mean(dim=1)
        return self.backbone(images)

    def forward(self, front, top, video_frames=None, video_mask=None):
        f = self.encode(front)
        t = self.encode(top)

        if self.head_type == "attn_gate":
            gate = self.attn_gate(torch.cat([f, t], dim=1))
            fused = gate[:, 0:1] * f + gate[:, 1:2] * t
        else:
            # Simple concat
            fused = torch.cat([f, t], dim=1)

        if self.use_video and video_frames is not None:
            fused = self._fuse_video(fused, video_frames, video_mask)
        return self.head(fused)

    def _fuse_video(self, fused, video_frames, video_mask):
        valid = video_mask > 0.5 if video_mask is not None else torch.ones(
            video_frames.size(0), dtype=torch.bool, device=fused.device)
        if not valid.any():
            return fused
        vf = video_frames[valid]
        B, T, C, H, W = vf.shape
        vid = self.backbone(vf.view(B * T, C, H, W)).view(B, T, -1).mean(dim=1)
        out = fused.clone()
        out[valid] = 0.7 * fused[valid] + 0.3 * vid
        return out


def build_model(backbone_key, **kwargs):
    return DualViewModel(backbone_key=backbone_key, **kwargs)


def get_backbone_config(key):
    return BACKBONE_CONFIGS[key]


def get_train_preset(key):
    bs, lr, plr, ga = TRAIN_PRESETS[key]
    return {"batch_size": bs, "lr": lr, "pretrain_lr": plr, "grad_accum": ga}


def get_backbone_choices():
    return list(BACKBONE_CONFIGS.keys())


def enable_gradient_checkpointing(model):
    bb = model.backbone
    if hasattr(bb, "set_grad_checkpointing"):
        bb.set_grad_checkpointing(True)
        return True
    return False
