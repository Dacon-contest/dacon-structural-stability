"""Comprehensive comparison: synthetic vs real."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

synth_base = "data/synthetic_preview2"
real_base = "data/open/train"

def analyze_image(path, view):
    """Extract key metrics from an image."""
    img = np.array(Image.open(path).convert("RGB"))
    
    # Background colors - use edges
    bg_top = img[:25, :25, :].mean(axis=(0, 1))
    bg_bot = img[-25:, -25:, :].mean(axis=(0, 1))
    
    # Checker detection: count transitions along center row/column
    center_row = img[192, :, :].astype(float)
    row_diffs = np.abs(np.diff(center_row, axis=0)).mean(axis=1)
    n_transitions = (row_diffs > 20).sum()
    
    # Block detection
    bg_ref = (img[:, :15, :].mean(axis=(0, 1)) + img[:, -15:, :].mean(axis=(0, 1))) / 2
    diff = np.abs(img[50:350, 80:304, :].astype(float) - bg_ref).mean(axis=2)
    block_mask = diff > 25
    block_area = block_mask.sum() / block_mask.size if block_mask.size > 0 else 0
    
    block_mean_rgb = None
    if block_mask.sum() > 100:
        block_pixels = img[50:350, 80:304, :][block_mask]
        block_mean_rgb = block_pixels.mean(axis=0)
    
    # Overall color stats
    overall_mean = img.mean(axis=(0, 1))
    overall_std = img.std(axis=(0, 1))
    
    # Check for checker pattern (two dominant colors)
    # In background areas (outside block region)
    bg_mask = ~np.zeros_like(block_mask)  # full area for now
    
    return {
        "bg_top": bg_top.astype(int),
        "bg_bot": bg_bot.astype(int),
        "n_transitions": n_transitions,
        "block_area": block_area,
        "block_mean_rgb": block_mean_rgb.astype(int) if block_mean_rgb is not None else None,
        "overall_mean": overall_mean.astype(int),
        "overall_std": overall_std.astype(int),
    }

# Analyze real samples
print("=" * 70)
print("COMPREHENSIVE COMPARISON: REAL vs SYNTHETIC")
print("=" * 70)

real_ids = sorted(os.listdir(real_base))[:15]
synth_ids = sorted(os.listdir(synth_base))
synth_ids = [s for s in synth_ids if s.startswith("SYNTH_")]

for label, base, ids in [("REAL", real_base, real_ids), ("SYNTH", synth_base, synth_ids)]:
    print(f"\n--- {label} ---")
    for view in ["front", "top"]:
        areas = []
        means = []
        bg_tops = []
        transitions = []
        for sid in ids:
            path = os.path.join(base, sid, f"{view}.png")
            if not os.path.exists(path):
                continue
            m = analyze_image(path, view)
            areas.append(m["block_area"])
            if m["block_mean_rgb"] is not None:
                means.append(m["block_mean_rgb"])
            bg_tops.append(m["bg_top"])
            transitions.append(m["n_transitions"])
        
        if areas:
            print(f"  {view}: block_area={np.mean(areas):.3f}+-{np.std(areas):.3f}  "
                  f"transitions={np.mean(transitions):.1f}+-{np.std(transitions):.1f}")
            if means:
                mean_arr = np.array(means)
                print(f"         block_RGB_mean={mean_arr.mean(axis=0).astype(int)}  "
                      f"std={mean_arr.std(axis=0).astype(int)}")
            if bg_tops:
                bg_arr = np.array(bg_tops)
                print(f"         bg_top_mean={bg_arr.mean(axis=0).astype(int)}")

# Check SPECIFIC pixels that should match
print("\n--- PIXEL-LEVEL CHECKS ---")
checks = [
    ("wall y=10", 10, 10, "front"),
    ("floor y=375", 375, 10, "front"),
    ("top corner", 10, 10, "top"),
]
for desc, y, x, view in checks:
    real_vals = []
    synth_vals = []
    for sid in real_ids[:5]:
        path = os.path.join(real_base, sid, f"{view}.png")
        if os.path.exists(path):
            img = np.array(Image.open(path).convert("RGB"))
            real_vals.append(img[y, x])
    for sid in synth_ids[:5]:
        path = os.path.join(synth_base, sid, f"{view}.png")
        if os.path.exists(path):
            img = np.array(Image.open(path).convert("RGB"))
            synth_vals.append(img[y, x])
    
    if real_vals and synth_vals:
        r_mean = np.array(real_vals).mean(axis=0).astype(int)
        s_mean = np.array(synth_vals).mean(axis=0).astype(int)
        diff_val = np.abs(r_mean - s_mean).mean()
        status = "OK" if diff_val < 15 else "MISMATCH"
        print(f"  {desc}: REAL={r_mean} SYNTH={s_mean} diff={diff_val:.1f} [{status}]")

# Create side-by-side comparison image (4 columns: real_front, synth_front, real_top, synth_top)
print("\n--- Creating comparison sheet ---")
cell = 384
gap = 6
n_pairs = min(4, len(synth_ids))

# Layout: 4 rows (pairs), 4 columns (real_front, synth_front, real_top, synth_top)
w = 4 * cell + 3 * gap
h = n_pairs * cell + (n_pairs - 1) * gap + 30  # +30 for header

canvas = np.ones((h, w, 3), dtype=np.uint8) * 50

for i in range(n_pairs):
    rid = real_ids[i * 3]  # spread out the real samples
    sid = synth_ids[i]
    
    y0 = 30 + i * (cell + gap)
    
    for j, (sample_id, base) in enumerate([(rid, real_base), (sid, synth_base)]):
        for k, view in enumerate(["front", "top"]):
            path = os.path.join(base, sample_id, f"{view}.png")
            if os.path.exists(path):
                img = np.array(Image.open(path).convert("RGB"))
                x0 = (j * 2 + k) * (cell + gap)
                canvas[y0:y0 + cell, x0:x0 + cell] = img

os.makedirs("previews", exist_ok=True)
Image.fromarray(canvas).save("previews/compare_v3.png")
print("Saved previews/compare_v3.png (columns: real_front, synth_front, real_top, synth_top)")
