"""
모델 모듈: 3가지 고성능 아키텍처
- Model A: EfficientNetV2-M Dual-View Fusion
- Model B: ConvNeXt-Base Dual-View Fusion
- Model C: SwinV2-Small Dual-View + 선택적 비디오 프레임 활용
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ===================================================================
# Model A: EfficientNetV2-M Dual-View 
# ===================================================================
class EfficientNetDualView(nn.Module):
    """
    EfficientNetV2-M 기반 Dual-View 모델
    - front, top 이미지를 각각 인코딩 후 Attention Fusion
    - 5060 RTX 16GB에서 batch_size=8 가능
    """

    def __init__(self, model_name="tf_efficientnetv2_m.in21k_ft_in1k",
                 pretrained=True, num_classes=2, drop_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained,
                                          num_classes=0, drop_rate=drop_rate)
        feat_dim = self.backbone.num_features

        # Cross-view attention fusion
        self.attn_gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 2),
            nn.Softmax(dim=1),
        )

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, front, top, video_frames=None):
        f_feat = self.backbone(front)
        t_feat = self.backbone(top)

        concat = torch.cat([f_feat, t_feat], dim=1)
        gate = self.attn_gate(concat)  # (B, 2)
        fused = gate[:, 0:1] * f_feat + gate[:, 1:2] * t_feat

        return self.head(fused)


# ===================================================================
# Model B: ConvNeXt-Base Dual-View
# ===================================================================
class ConvNeXtDualView(nn.Module):
    """
    ConvNeXt-Base 기반 Dual-View 모델
    - Bidirectional feature fusion
    - 5060 RTX 16GB에서 batch_size=4~6 가능
    """

    def __init__(self, model_name="convnext_base.fb_in22k_ft_in1k",
                 pretrained=True, num_classes=2, drop_rate=0.3):
        super().__init__()
        self.front_backbone = timm.create_model(model_name, pretrained=pretrained,
                                                 num_classes=0, drop_rate=drop_rate)
        self.top_backbone = timm.create_model(model_name, pretrained=pretrained,
                                               num_classes=0, drop_rate=drop_rate)
        feat_dim = self.front_backbone.num_features

        # Bilinear fusion
        self.bilinear_proj = nn.Bilinear(feat_dim, feat_dim, feat_dim)

        # Multi-head classification
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

    def forward(self, front, top, video_frames=None):
        f_feat = self.front_backbone(front)
        t_feat = self.top_backbone(top)
        fused = self.bilinear_proj(f_feat, t_feat)
        return self.head(fused)


# ===================================================================
# Model C: SwinV2-Small Dual-View + Optional Video Frame Aggregation
# ===================================================================
class SwinV2DualView(nn.Module):
    """
    SwinV2-Small 기반 Dual-View + 시뮬레이션 비디오 프레임 활용
    - front+top: SwinV2로 인코딩
    - video frames: 추가 temporal feature (학습 데이터에만 mp4 존재)
    - 5060 RTX 16GB에서 batch_size=4~8 가능
    """

    def __init__(self, model_name="swinv2_small_window16_256.ms_in1k",
                 pretrained=True, num_classes=2, drop_rate=0.3,
                 use_video=False, num_video_frames=3):
        super().__init__()
        self.use_video = use_video

        self.backbone = timm.create_model(model_name, pretrained=pretrained,
                                          num_classes=0, drop_rate=drop_rate)
        feat_dim = self.backbone.num_features

        # Dual-view concat + projection
        self.view_fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(drop_rate * 0.5),
        )

        # Video temporal aggregation (optional)
        if use_video:
            self.video_backbone = timm.create_model(model_name, pretrained=pretrained,
                                                     num_classes=0, drop_rate=drop_rate)
            self.temporal_attn = nn.MultiheadAttention(feat_dim, num_heads=8, batch_first=True)
            self.video_proj = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.LayerNorm(feat_dim),
                nn.GELU(),
            )
            self.final_fusion = nn.Sequential(
                nn.Linear(feat_dim * 2, feat_dim),
                nn.LayerNorm(feat_dim),
                nn.GELU(),
                nn.Dropout(drop_rate),
            )

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, front, top, video_frames=None):
        f_feat = self.backbone(front)
        t_feat = self.backbone(top)
        fused = self.view_fusion(torch.cat([f_feat, t_feat], dim=1))

        if self.use_video and video_frames is not None:
            B, T, C, H, W = video_frames.shape
            vid = video_frames.view(B * T, C, H, W)
            vid_feat = self.video_backbone(vid)
            vid_feat = vid_feat.view(B, T, -1)
            vid_attn, _ = self.temporal_attn(vid_feat, vid_feat, vid_feat)
            vid_pooled = vid_attn.mean(dim=1)
            vid_proj = self.video_proj(vid_pooled)
            fused = self.final_fusion(torch.cat([fused, vid_proj], dim=1))

        return self.head(fused)


# ===================================================================
# Model Factory
# ===================================================================
MODEL_REGISTRY = {
    "efficientnet": EfficientNetDualView,
    "convnext": ConvNeXtDualView,
    "swinv2": SwinV2DualView,
}


def build_model(model_type="efficientnet", **kwargs):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_type}. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type](**kwargs)
