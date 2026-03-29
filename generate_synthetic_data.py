"""
generate_synthetic_data.py - structural stability synthetic data generator.

Generates 384x384 front/top images closely matching the Dacon competition data:
- Checkerboard floor/wall background matching real MuJoCo-rendered images
- Dark muted block colors extracted from real data analysis
- Box-only towers (no cylinders/spheres)
- PyBullet physics for 3D rendering + stability checking
- Output labels as 0/1 where 1=unstable, 0=stable

Usage:
    python generate_synthetic_data.py --n 200 --out data/synthetic
    python generate_synthetic_data.py --n 50 --out data/synthetic_preview --seed 7
"""

import argparse
import math
import os
import random

import numpy as np
import pandas as pd

try:
    import pybullet as p
    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

IMG_SIZE = 384
SIM_STEPS = 240

# ---------------------------------------------------------------------------
# Colors extracted from real competition data analysis
# ---------------------------------------------------------------------------
# Checker floor/wall: two alternating colors
CHECKER_C1 = np.array([172, 198, 238], dtype=np.uint8)  # blue-gray
CHECKER_C2 = np.array([254, 254, 254], dtype=np.uint8)  # near-white
WALL_COLOR = np.array([178, 178, 204], dtype=np.uint8)  # grayish-blue wall

# Block colors sampled from actual block pixel values in competition images.
# These are DARK and MUTED - the real MuJoCo textures are quite desaturated.
BLOCK_COLORS = [
    (0.38, 0.33, 0.44),  # dark purple-gray
    (0.43, 0.49, 0.41),  # muted green
    (0.34, 0.45, 0.53),  # muted blue
    (0.44, 0.31, 0.38),  # dark rose
    (0.40, 0.27, 0.52),  # dark purple
    (0.48, 0.45, 0.30),  # olive-brown
    (0.38, 0.36, 0.31),  # dark olive
    (0.55, 0.39, 0.53),  # muted magenta
    (0.30, 0.47, 0.43),  # dark teal
    (0.42, 0.55, 0.36),  # muted green
    (0.54, 0.40, 0.55),  # plum
    (0.36, 0.50, 0.50),  # teal
    (0.50, 0.35, 0.40),  # dusty rose
    (0.32, 0.32, 0.57),  # dark blue
    (0.42, 0.39, 0.42),  # neutral gray
    (0.52, 0.42, 0.35),  # brown
    (0.30, 0.55, 0.37),  # forest green
    (0.65, 0.41, 0.53),  # rose
    (0.37, 0.64, 0.55),  # teal-green
    (0.28, 0.30, 0.57),  # navy blue
]


def _ensure_supported():
    if not HAS_PIL:
        raise ImportError("Pillow is required: pip install Pillow")


# ---------------------------------------------------------------------------
# Procedural background generation matching real competition images
# ---------------------------------------------------------------------------

def _make_top_checker_fast(size=IMG_SIZE, square_px=47, offset_x=5, offset_y=3):
    """Vectorized top-view checker background.
    Real data shows checker squares of ~47px alternation period in both axes."""
    yy, xx = np.mgrid[0:size, 0:size]
    cell = (((xx + offset_x) // square_px) + ((yy + offset_y) // square_px)) % 2
    img = np.where(cell[:, :, None] == 0,
                   CHECKER_C1[None, None, :],
                   CHECKER_C2[None, None, :])
    return img.astype(np.uint8)


def _make_front_checker_fast(size=IMG_SIZE, world_offset_x=0.0, world_offset_y=0.0):
    """Front-view background with wall + perspective checker floor.
    Real data: wall at top (y=0~23) is [178,178,204], then perspective
    checkerboard floor below. world_offset_x/y add variation per sample."""
    img = np.zeros((size, size, 3), dtype=np.uint8)

    wall_end = 23
    img[:wall_end, :] = WALL_COLOR

    # Smooth transition band (wall → floor) matching real data y=25-50 region
    # Real images show gradual blending with intermediate values like [203,219,244]
    trans_start = wall_end
    trans_end = 50
    for y in range(trans_start, trans_end):
        t = (y - trans_start) / max(1, trans_end - trans_start - 1)
        avg_checker = (CHECKER_C1.astype(np.float32) + CHECKER_C2.astype(np.float32)) / 2
        blended = WALL_COLOR.astype(np.float32) * (1 - t) + avg_checker * t
        img[y, :] = blended.astype(np.uint8)

    horizon_y = trans_end
    cam_height = 0.22
    fov_half = math.radians(14)
    checker_world_size = 0.12

    ix_arr = np.arange(size)
    x_frac = (ix_arr - size / 2) / (size / 2)

    for iy in range(horizon_y, size):
        t = (iy - horizon_y) / max(1, size - 1 - horizon_y)
        angle = max(0.001, t * fov_half * 1.8)
        floor_dist = cam_height / math.tan(angle) if angle < math.pi / 2 else 0.01

        world_x = x_frac * floor_dist * math.tan(fov_half) + world_offset_x
        cx = np.floor(world_x / checker_world_size).astype(int)
        cy = int(math.floor((floor_dist + world_offset_y) / checker_world_size))
        cell = (cx + cy) % 2
        row = np.where(cell[:, None] == 0,
                       CHECKER_C1[None, :],
                       CHECKER_C2[None, :])
        img[iy, :] = row.astype(np.uint8)

    return img


class RealisticBlockStackGenerator:
    """PyBullet-based generator producing images matching competition data."""

    def __init__(self, img_size=IMG_SIZE, seed=42):
        _ensure_supported()
        if not HAS_PYBULLET:
            raise ImportError("PyBullet is required for the realistic generator")

        self.img_size = img_size
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client)

        # Front background will be generated per-sample with slight variations
        self._front_bg_cache = None

    def close(self):
        p.disconnect(self.client)

    def _reset_scene(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client)

        # Invisible ground plane (physics only)
        plane_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.client)
        plane_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[3.0, 3.0, 0.001],
                                        rgbaColor=[0, 0, 0, 0], physicsClientId=self.client)
        plane_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane_shape,
                                     baseVisualShapeIndex=plane_vis, basePosition=[0, 0, 0],
                                     physicsClientId=self.client)
        p.changeDynamics(plane_id, -1, lateralFriction=0.9, restitution=0.0,
                         physicsClientId=self.client)
        return plane_id

    def _sample_box_dims(self):
        """Sample block dimensions matching real data.
        Real block heights in front view: 15-80px → world height ~0.008-0.040
        Real block widths in front view:  40-90px → world width ~0.030-0.070"""
        width = self.rng.uniform(0.025, 0.058)
        depth = self.rng.uniform(0.025, 0.058)
        height = self.rng.uniform(0.010, 0.042)

        # Frequent near-square blocks
        if self.rng.random() < 0.60:
            depth = width * self.rng.uniform(0.85, 1.15)
        return width, depth, height

    def _sample_color(self):
        """Sample a block color from the competition-matched palette."""
        base = self.rng.choice(BLOCK_COLORS)
        # Small random variation for natural look
        r = base[0] + self.rng.uniform(-0.04, 0.04)
        g = base[1] + self.rng.uniform(-0.04, 0.04)
        b = base[2] + self.rng.uniform(-0.04, 0.04)
        return (max(0.15, min(0.85, r)),
                max(0.15, min(0.85, g)),
                max(0.15, min(0.85, b)),
                1.0)

    def _sample_stack(self, make_unstable):
        n_blocks = self.rng.randint(5, 10)
        blocks = []
        z = 0.0
        drift_axis = self.rng.choice([0, 1])
        drift_sign = self.rng.choice([-1.0, 1.0])
        cumulative = [0.0, 0.0]

        for idx in range(n_blocks):
            width, depth, height = self._sample_box_dims()
            color = self._sample_color()
            yaw = np.deg2rad(self.rng.choice([0, 0, 0, 90]))

            if make_unstable:
                base_shift = 0.002 + 0.003 * idx
                if idx >= max(2, n_blocks - 3):
                    base_shift += 0.008 + 0.006 * (idx - (n_blocks - 3))
                offset = [
                    self.rng.uniform(-0.003, 0.003),
                    self.rng.uniform(-0.003, 0.003),
                ]
                offset[drift_axis] += drift_sign * base_shift
            else:
                offset = [
                    self.rng.uniform(-0.003, 0.003),
                    self.rng.uniform(-0.003, 0.003),
                ]

            cumulative[0] += offset[0]
            cumulative[1] += offset[1]
            pos = [cumulative[0], cumulative[1], z + height / 2.0]
            z += height

            blocks.append({
                "size": (width, depth, height),
                "color": color,
                "yaw": yaw,
                "position": pos,
            })

        return blocks

    def _create_block(self, block):
        width, depth, height = block["size"]
        half = [width / 2.0, depth / 2.0, height / 2.0]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.client)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=block["color"],
                                  physicsClientId=self.client)
        orn = p.getQuaternionFromEuler([0, 0, block["yaw"]])
        body = p.createMultiBody(baseMass=0.35, baseCollisionShapeIndex=col,
                                 baseVisualShapeIndex=vis, basePosition=block["position"],
                                 baseOrientation=orn, physicsClientId=self.client)
        p.changeDynamics(body, -1, lateralFriction=0.95, spinningFriction=0.02,
                         rollingFriction=0.001, restitution=0.0, linearDamping=0.03,
                         angularDamping=0.04, physicsClientId=self.client)
        return body

    def _camera_spec(self, view_name, tower_height):
        """Camera parameters tuned to match real competition image framing.

        Real data analysis:
        - Front: tower fills ~65% of image height, nearly full width
        - Top: blocks in small central cluster, checker covers full image
        """
        if view_name == "front":
            target_z = tower_height * 0.45
            return {
                "eye": [0.88, -0.04, 0.22],
                "target": [0.0, 0.0, target_z],
                "up": [0.0, 0.0, 1.0],
                "fov": 25,
            }
        else:
            return {
                "eye": [0.0, 0.0, 0.85],
                "target": [0.0, 0.0, 0.0],
                "up": [0.0, 1.0, 0.0],
                "fov": 18,
            }

    def _render_with_mask(self, block_ids, view_name, tower_height):
        """Render blocks and composite onto procedural checker background."""
        spec = self._camera_spec(view_name, tower_height)
        view = p.computeViewMatrix(
            cameraEyePosition=spec["eye"],
            cameraTargetPosition=spec["target"],
            cameraUpVector=spec["up"],
        )
        proj = p.computeProjectionMatrixFOV(
            fov=spec["fov"],
            aspect=1.0,
            nearVal=0.02,
            farVal=4.0,
        )

        _, _, rgba, _, seg = p.getCameraImage(
            width=self.img_size,
            height=self.img_size,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            lightDirection=[1.5, -0.5, 2.5],
            lightColor=[1.0, 1.0, 1.0],
            lightDistance=5.0,
            lightAmbientCoeff=0.66,
            lightDiffuseCoeff=0.46,
            lightSpecularCoeff=0.03,
            physicsClientId=self.client,
        )

        rgb = np.array(rgba, dtype=np.uint8).reshape(self.img_size, self.img_size, 4)[:, :, :3]
        seg = np.array(seg, dtype=np.int64).reshape(self.img_size, self.img_size)
        obj_ids = np.bitwise_and(seg, (1 << 24) - 1)
        block_mask = np.isin(obj_ids, list(block_ids))

        # Get appropriate background
        if view_name == "front":
            # Generate front background with random world-space offset for variety
            owx = self.rng.uniform(0.0, 0.24)
            owy = self.rng.uniform(0.0, 0.24)
            bg = _make_front_checker_fast(self.img_size, owx, owy)
        else:
            # Top checker: real data always starts with C1 at (0,0), no offset
            bg = _make_top_checker_fast(self.img_size, square_px=47,
                                        offset_x=0, offset_y=0)

        # Composite: blocks over checker background
        bg[block_mask] = rgb[block_mask]
        return bg

    def _stability_from_bodies(self, bodies, init_positions):
        max_disp = 0.0
        unstable = False
        for body, init_pos in zip(bodies, init_positions):
            pos, orn = p.getBasePositionAndOrientation(body, physicsClientId=self.client)
            pos = np.array(pos)
            disp = float(np.linalg.norm(pos - init_pos))
            roll, pitch, _ = p.getEulerFromQuaternion(orn)
            max_disp = max(max_disp, disp)
            if pos[2] < 0.01 or disp > 0.06 or abs(roll) > 0.40 or abs(pitch) > 0.40:
                unstable = True
        return (not unstable), max_disp

    def generate_one(self, make_unstable=False):
        self._reset_scene()
        block_defs = self._sample_stack(make_unstable=make_unstable)
        block_ids = [self._create_block(block) for block in block_defs]
        keep_ids = set(block_ids)

        init_positions = [
            np.array(p.getBasePositionAndOrientation(body, physicsClientId=self.client)[0])
            for body in block_ids
        ]

        tower_height = sum(b["size"][2] for b in block_defs)

        front_img = self._render_with_mask(keep_ids, "front", tower_height)
        top_img = self._render_with_mask(keep_ids, "top", tower_height)

        for _ in range(SIM_STEPS):
            p.stepSimulation(physicsClientId=self.client)

        is_stable, max_disp = self._stability_from_bodies(block_ids, init_positions)
        meta = {
            "max_displacement": max_disp,
            "n_blocks": len(block_defs),
            "actual_stable": is_stable,
        }
        return front_img, top_img, is_stable, meta


def _save_sample(out_dir, sample_id, front_img, top_img):
    sample_dir = os.path.join(out_dir, sample_id)
    os.makedirs(sample_dir, exist_ok=True)
    Image.fromarray(front_img).save(os.path.join(sample_dir, "front.png"))
    Image.fromarray(top_img).save(os.path.join(sample_dir, "top.png"))


def generate_dataset(n_samples, out_dir, seed=42, balance=True):
    _ensure_supported()
    os.makedirs(out_dir, exist_ok=True)

    if HAS_PYBULLET:
        print("[INFO] PyBullet 3D generator enabled")
        generator = RealisticBlockStackGenerator(img_size=IMG_SIZE, seed=seed)
    else:
        raise ImportError("PyBullet is required. Install with: pip install pybullet")

    records = []
    stable_count = 0
    unstable_count = 0
    target_each = n_samples // 2
    attempts = 0
    max_attempts = n_samples * 12

    while len(records) < n_samples and attempts < max_attempts:
        attempts += 1
        if balance:
            if stable_count >= target_each:
                make_unstable = True
            elif unstable_count >= target_each:
                make_unstable = False
            else:
                make_unstable = random.random() < 0.5
        else:
            make_unstable = random.random() < 0.5

        front_img, top_img, is_stable, meta = generator.generate_one(make_unstable=make_unstable)

        if balance:
            if make_unstable and is_stable:
                continue
            if (not make_unstable) and (not is_stable):
                continue

        sample_id = f"SYNTH_{len(records) + 1:04d}"
        _save_sample(out_dir, sample_id, front_img, top_img)
        label = 0 if is_stable else 1
        records.append({"id": sample_id, "label": label})
        if is_stable:
            stable_count += 1
        else:
            unstable_count += 1

        if len(records) % 25 == 0 or len(records) == n_samples:
            print(f"  [{len(records)}/{n_samples}] stable={stable_count} unstable={unstable_count}")

    generator.close()

    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "synthetic.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nSynthetic dataset: {len(df)} samples ({stable_count} stable, {unstable_count} unstable)")
    print(f"CSV: {csv_path}")
    if len(df) < n_samples:
        print(f"[WARN] Requested {n_samples}, generated {len(df)}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Structural stability synthetic data generator")
    parser.add_argument("--n", type=int, default=200, help="number of samples to generate")
    parser.add_argument("--out", type=str, default="data/synthetic", help="output directory")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no-balance", action="store_true", help="do not balance stable/unstable")
    args = parser.parse_args()

    generate_dataset(
        n_samples=args.n,
        out_dir=args.out,
        seed=args.seed,
        balance=not args.no_balance,
    )


if __name__ == "__main__":
    main()
