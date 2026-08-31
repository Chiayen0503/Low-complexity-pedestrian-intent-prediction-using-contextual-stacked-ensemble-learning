import os
import io
import sys
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from time import time
from mmpose.apis import MMPoseInferencer

# ── Config ──────────────────────────────────────────────────────────────
IMAGE_DIR   = Path("total_images")
DEVICE      = "cuda:0"
STATS_FILE  = Path("keypoint_timing_stats.json")
# ────────────────────────────────────────────────────────────────────────

def predict_allkeypoints(inferencer, img_path):
    result_generator = inferencer(img_path, show=False)
    result = next(result_generator)
    result = result['predictions'][0]
    people_keypoints = [dic['keypoints'] for dic in result]
    return people_keypoints

def load_stats():
    if STATS_FILE.exists():
        with open(STATS_FILE, "r") as f:
            return json.load(f)
    return {"count": 0, "running_mean": 0.0, "running_m2": 0.0, "failed": []}

def save_stats(stats):
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

def update_welford(stats, new_value):
    """Welford's online algorithm for mean and variance."""
    stats["count"] += 1
    delta = new_value - stats["running_mean"]
    stats["running_mean"] += delta / stats["count"]
    delta2 = new_value - stats["running_mean"]
    stats["running_m2"] += delta * delta2
    return stats

# ── Resume support: skip already-processed images ───────────────────────
stats = load_stats()
already_done = stats["count"] + len(stats["failed"])
print(f"Resuming from {already_done} previously processed images.")

inferencer = MMPoseInferencer('human', device=DEVICE)

img_list = sorted(IMAGE_DIR.glob("*.png"))
img_list = img_list[already_done:]  # skip already processed
print(f"Remaining: {len(img_list)} images")

for img_path in tqdm(img_list):
    try:
        text_trap = io.StringIO()
        sys.stdout = text_trap

        start = time()
        people_keypoints = predict_allkeypoints(inferencer, str(img_path))
        elapsed = time() - start

        sys.stdout = sys.__stdout__

        n_people = max(len(people_keypoints), 1)
        t_per_person = elapsed / n_people

        stats = update_welford(stats, t_per_person)
        save_stats(stats)

    except Exception as e:
        sys.stdout = sys.__stdout__
        stats["failed"].append({"file": img_path.name, "error": str(e)})
        save_stats(stats)

# ── Final results ────────────────────────────────────────────────────────
variance = stats["running_m2"] / stats["count"] if stats["count"] > 1 else 0
std = variance ** 0.5

print(f"\nProcessed : {stats['count']} images")
print(f"Failed    : {len(stats['failed'])} images")
print(f"Mean time : {stats['running_mean']*1000:.3f} ms")
print(f"Std  time : {std*1000:.3f} ms")