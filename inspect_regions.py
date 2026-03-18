import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from datasets import get_val_transforms
from models import build_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "region_checks")


def get_model_img_size(model_type):
    return {"efficientnet": 384, "convnext": 384, "swinv2": 256}[model_type]


def ensure_nchw(feature_map):
    if feature_map.dim() != 4:
        raise ValueError(f"Expected a 4D feature map, got shape={tuple(feature_map.shape)}")
    if feature_map.shape[1] <= 16 and feature_map.shape[-1] > 16:
        return feature_map.permute(0, 3, 1, 2).contiguous()
    return feature_map


def extract_map_and_feat(backbone, images):
    if images.dim() == 5:
        images = images[:, 0]

    feature_map_raw = backbone.forward_features(images)
    feature_map_nchw = ensure_nchw(feature_map_raw)
    feature_map_nchw.retain_grad()

    if hasattr(backbone, "forward_head"):
        feat = backbone.forward_head(feature_map_raw, pre_logits=True)
    else:
        feat = F.adaptive_avg_pool2d(feature_map_nchw, output_size=1).flatten(1)

    return feature_map_nchw, feat


def build_cam(feature_map, gradients, output_size):
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feature_map).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam -= cam.min()
    cam /= max(cam.max(), 1e-8)
    return cam


def overlay_heatmap(image_rgb, cam):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    heat = np.uint8(cam * 255.0)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 0.55, heat, 0.45, 0.0)


def load_state_dict(checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def read_sample(split, sample_id=None, sample_index=None):
    csv_path = os.path.join(DATA_DIR, "open", f"{split}.csv")
    data_dir = os.path.join(DATA_DIR, "open", split)
    df = pd.read_csv(csv_path)

    if sample_id is not None:
        row = df[df["id"] == sample_id].iloc[0]
    else:
        row = df.iloc[sample_index]

    sample_dir = os.path.join(data_dir, row["id"])
    front = cv2.cvtColor(cv2.imread(os.path.join(sample_dir, "front.png")), cv2.COLOR_BGR2RGB)
    top = cv2.cvtColor(cv2.imread(os.path.join(sample_dir, "top.png")), cv2.COLOR_BGR2RGB)
    label = row["label"] if "label" in row.index else None
    return row["id"], front, top, label


def forward_efficientnet(model, front_tensor, top_tensor):
    front_map, front_feat = extract_map_and_feat(model.backbone, front_tensor)
    top_map, top_feat = extract_map_and_feat(model.backbone, top_tensor)

    concat = torch.cat([front_feat, top_feat], dim=1)
    gate = model.attn_gate(concat)
    fused = gate[:, 0:1] * front_feat + gate[:, 1:2] * top_feat
    logits = model.head(fused)

    return logits, front_map, top_map, gate


def run_efficientnet_cam(model, front_tensor, top_tensor, target_class):
    logits, front_map, top_map, gate = forward_efficientnet(model, front_tensor, top_tensor)
    model.zero_grad(set_to_none=True)
    score = logits[:, target_class].sum()
    score.backward()

    return logits, front_map, top_map, gate


def forward_convnext(model, front_tensor, top_tensor):
    front_map, front_feat = extract_map_and_feat(model.front_backbone, front_tensor)
    top_map, top_feat = extract_map_and_feat(model.top_backbone, top_tensor)

    fused = model.bilinear_proj(front_feat, top_feat)
    logits = model.head(fused)

    return logits, front_map, top_map


def run_convnext_cam(model, front_tensor, top_tensor, target_class):
    logits, front_map, top_map = forward_convnext(model, front_tensor, top_tensor)

    model.zero_grad(set_to_none=True)
    score = logits[:, target_class].sum()
    score.backward()

    return logits, front_map, top_map


def save_panel(output_path, sample_id, label_name, pred_name, target_name, probs, gate_text,
               front_orig, top_orig, front_overlay, top_overlay):
    canvas_h = 900
    canvas_w = 900
    panel = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)

    def fit(image_bgr, x0, y0, w, h):
        resized = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        panel[y0:y0 + h, x0:x0 + w] = resized

    fit(cv2.cvtColor(front_orig, cv2.COLOR_RGB2BGR), 30, 120, 400, 300)
    fit(front_overlay, 470, 120, 400, 300)
    fit(cv2.cvtColor(top_orig, cv2.COLOR_RGB2BGR), 30, 500, 400, 300)
    fit(top_overlay, 470, 500, 400, 300)

    cv2.putText(panel, f"ID: {sample_id}", (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    cv2.putText(panel, f"Label: {label_name}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(panel, f"Pred: {pred_name} | unstable={probs[1]:.4f} stable={probs[0]:.4f}",
                (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2)
    cv2.putText(panel, f"Heatmap target: {target_name}", (470, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2)
    if gate_text:
        cv2.putText(panel, gate_text, (470, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2)

    cv2.putText(panel, "Front Original", (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.putText(panel, "Front Heatmap", (470, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.putText(panel, "Top Original", (30, 830), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.putText(panel, "Top Heatmap", (470, 830), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)

    cv2.imwrite(output_path, panel)


def main():
    parser = argparse.ArgumentParser(description="샘플별 영역 집중도 시각화")
    parser.add_argument("--model", type=str, default="efficientnet",
                        choices=["efficientnet", "convnext"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev"])
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--sample_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_class", type=str, default="predicted",
                        choices=["predicted", "stable", "unstable"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint or os.path.join(SAVE_DIR, f"{args.model}_pretrained.pth")
    img_size = get_model_img_size(args.model)
    transform = get_val_transforms(img_size)

    model = build_model(args.model, pretrained=False, num_classes=2, drop_rate=0.0, use_video=False)
    state_dict = load_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    rng = np.random.default_rng(args.seed)
    csv_path = os.path.join(DATA_DIR, "open", f"{args.split}.csv")
    df = pd.read_csv(csv_path)
    if args.sample_id is not None:
        indices = [0]
    else:
        start = max(args.start_index, 0)
        end = min(start + args.num_samples, len(df))
        indices = list(range(start, end))
        if not indices:
            indices = list(rng.choice(len(df), size=min(args.num_samples, len(df)), replace=False))

    output_subdir = os.path.join(OUTPUT_DIR, f"{args.model}_{args.split}")
    os.makedirs(output_subdir, exist_ok=True)

    for offset, idx in enumerate(indices):
        sample_id, front_orig, top_orig, label_name = read_sample(args.split, sample_id=args.sample_id, sample_index=idx)
        front_tensor = transform(image=front_orig)["image"].unsqueeze(0).to(device)
        top_tensor = transform(image=top_orig)["image"].unsqueeze(0).to(device)

        if args.model == "efficientnet":
            with torch.enable_grad():
                logits, _, _, gate = forward_efficientnet(model, front_tensor, top_tensor)
        else:
            with torch.enable_grad():
                logits, _, _ = forward_convnext(model, front_tensor, top_tensor)
                gate = None

        probs = torch.softmax(logits.float(), dim=1)[0].detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_name = "unstable" if pred_idx == 1 else "stable"

        if args.target_class == "predicted":
            target_idx = pred_idx
        elif args.target_class == "unstable":
            target_idx = 1
        else:
            target_idx = 0
        target_name = "unstable" if target_idx == 1 else "stable"

        if args.model == "efficientnet":
            with torch.enable_grad():
                logits, front_map, top_map, gate = run_efficientnet_cam(model, front_tensor, top_tensor, target_class=target_idx)
            gate = gate[0].detach().cpu().numpy()
            gate_text = f"Gate front={gate[0]:.3f} top={gate[1]:.3f}"
        else:
            with torch.enable_grad():
                logits, front_map, top_map = run_convnext_cam(model, front_tensor, top_tensor, target_class=target_idx)
            gate_text = ""

        front_cam = build_cam(front_map, front_map.grad, output_size=front_orig.shape[:2])
        top_cam = build_cam(top_map, top_map.grad, output_size=top_orig.shape[:2])

        front_overlay = overlay_heatmap(front_orig, front_cam)
        top_overlay = overlay_heatmap(top_orig, top_cam)

        output_path = os.path.join(output_subdir, f"{offset:02d}_{sample_id}.png")
        save_panel(output_path, sample_id, label_name or "unknown", pred_name, target_name, probs, gate_text,
                   front_orig, top_orig, front_overlay, top_overlay)
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()