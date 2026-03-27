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
import torch.utils.checkpoint
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


class CrossViewFusion(nn.Module):
    """Token-level cross-view fusion via Transformer with learnable [CLS] token.

    [CLS] + Front patch tokens + Top patch tokens → Transformer → CLS output.
    CLS token이 양쪽 뷰의 모든 패치를 attend하며 cross-view 관계를 학습.
    """

    def __init__(self, feat_dim, n_layers=4, n_heads=8, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.view_embed = nn.Parameter(torch.zeros(2, 1, feat_dim))
        nn.init.trunc_normal_(self.view_embed, std=0.02)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=feat_dim, nhead=n_heads,
                dim_feedforward=feat_dim * 4,
                dropout=dropout, activation='gelu',
                batch_first=True, norm_first=True,
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(feat_dim)
        self.grad_ckpt = False
        self.n_heads = n_heads

    def _build_sequence(self, front_tokens, top_tokens):
        """[CLS] + front(+view_embed) + top(+view_embed) → (B, 1+Nf+Nt, D)"""
        B = front_tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        front_tokens = front_tokens + self.view_embed[0]
        top_tokens = top_tokens + self.view_embed[1]
        return torch.cat([cls, front_tokens, top_tokens], dim=1)

    def forward(self, front_tokens, top_tokens):
        """(B, N_f, D), (B, N_t, D) → (B, D) CLS output"""
        x = self._build_sequence(front_tokens, top_tokens)

        for layer in self.layers:
            if self.grad_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, use_reentrant=False)
            else:
                x = layer(x)

        return self.norm(x[:, 0])  # CLS token only

    @torch.no_grad()
    def forward_with_attention(self, front_tokens, top_tokens):
        """Forward + 각 layer의 attention weight 추출 (시각화용).

        Returns:
            cls_out: (B, D) — CLS token output
            attentions: list of (B, H, N, N) attention matrices per layer
        """
        x = self._build_sequence(front_tokens, top_tokens)
        attentions = []

        for layer in self.layers:
            # norm_first=True: self_attn(norm1(x))
            x_norm = layer.norm1(x)
            attn_out, attn_w = layer.self_attn(
                x_norm, x_norm, x_norm,
                need_weights=True, average_attn_weights=False,
            )
            # attn_w: (B, H, N, N)
            attentions.append(attn_w)

            # Complete the layer: residual + FFN
            x = x + layer.dropout1(attn_out)
            x = x + layer._ff_block(layer.norm2(x))

        return self.norm(x[:, 0]), attentions


class DualViewModel(nn.Module):
    """공유 백본 → Fusion → MLP Head
    head_type:
      - "attn_gate": Attention Gate Fusion → deep MLP (기존)
      - "simple":    Concat → BN+MLP (단순 헤드)
      - "cross_attn": Token-level Cross-View Transformer Fusion
    """

    def __init__(self, backbone_key="eva02_large", pretrained=True,
                 num_classes=2, drop_rate=0.3,
                 use_video=False, num_video_frames=5,
                 head_type="attn_gate",
                 fusion_layers=4, fusion_heads=8):
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

        if head_type == "cross_attn":
            self.attn_gate = None
            self.fusion = CrossViewFusion(
                feat_dim, n_layers=fusion_layers,
                n_heads=fusion_heads, dropout=drop_rate,
            )
            self.head = nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(feat_dim, 256),
                nn.GELU(),
                nn.Dropout(drop_rate * 0.5),
                nn.Linear(256, num_classes),
            )
        elif head_type == "attn_gate":
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

    def get_patch_tokens(self, images):
        """Extract patch tokens without pooling (for cross_attn)."""
        feat = self.backbone.forward_features(images)
        if feat.dim() == 4:
            # ConvNeXt: (B, C, H, W) → (B, HW, C)
            B, C, H, W = feat.shape
            return feat.reshape(B, C, H * W).permute(0, 2, 1)
        # ViT: strip CLS + register prefix tokens
        n_prefix = getattr(self.backbone, 'num_prefix_tokens',
                           5 if 'dinov2' in self.backbone_key else 1)
        return feat[:, n_prefix:, :]

    def forward(self, front, top, video_frames=None, video_mask=None):
        if self.head_type == "cross_attn":
            tok_f = self.get_patch_tokens(front)
            tok_t = self.get_patch_tokens(top)
            fused = self.fusion(tok_f, tok_t)
            return self.head(fused)

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
    ok = False
    bb = model.backbone
    if hasattr(bb, "set_grad_checkpointing"):
        bb.set_grad_checkpointing(True)
        ok = True
    if hasattr(model, 'fusion') and hasattr(model.fusion, 'grad_ckpt'):
        model.fusion.grad_ckpt = True
        ok = True
    return ok
