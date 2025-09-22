import cv2
import numpy as np
import os
import random
from glob import glob

# Configs

retina_images = sorted(glob(f"data/*.jpg"))

output_dir = f"output"
os.makedirs(output_dir, exist_ok=True)

# Mask generator (feathered circle)
def create_feathered_mask(size, feather_amount=15):
    h, w = size
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w//2, h//2), min(h, w)//2 - feather_amount, 255, -1)
    mask = cv2.GaussianBlur(mask, (feather_amount*2+1, feather_amount*2+1), 0)
    return cv2.merge([mask, mask, mask])

# Process each retina image
for retina_path in retina_images:
    retina = cv2.imread(retina_path)
    h_retina, w_retina = retina.shape[:2]
    print("hi")
    print(h_retina, w_retina)
    arts= random.randint(1,26)
    artifact_images = f"artifact_segment/art{arts}_masked.png"       # Artifact images: art1.png, art2.png, ...
    artifact = cv2.imread(artifact_images)

    # Random scale factor between 0.1 to 0.3 of retina size
    scale = random.uniform(0.1, 0.2)
    target_size = int(min(h_retina, w_retina) * scale)

    # Resize artifact
    artifact_resized = cv2.resize(artifact, (target_size, target_size))

    # Create feathered mask
    feathered_mask = create_feathered_mask(artifact_resized.shape[:2])

    # Random position ensuring artifact fits inside retina
    center_x = random.randint(w_retina//2 - 2*target_size, w_retina//2 + 2*target_size)
    center_y = random.randint(h_retina//2 - 2*target_size, h_retina//2 + 2*target_size)
    center = (center_x, center_y)

    # Blend using Poisson blending
    blended = cv2.seamlessClone(artifact_resized, retina, feathered_mask, center, cv2.NORMAL_CLONE)

    # Save result
    retina_base = os.path.splitext(os.path.basename(retina_path))[0]
    artifact_base = os.path.splitext(os.path.basename(artifact_images))[0]
    out_name = f"{retina_base}_{artifact_base}.png"
    cv2.imwrite(os.path.join(output_dir, out_name), blended)

print("✅ Done. All images saved in 'output' folder.")