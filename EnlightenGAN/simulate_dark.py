import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

input_folder = "C:/Users/Vignesh M/PycharmProjects/Thesis/dataset/lfw_test_clean"
output_folder = "C:/Users/Vignesh M/PycharmProjects/Thesis/dataset/lfw_test_dark"
os.makedirs(output_folder, exist_ok=True)

def darken_image(img, gamma=0.1, add_noise=True):
    # Step 1: Gamma correction (nonlinear darkening)
    img_float = img / 255.0
    dark = np.power(img_float, gamma)

    # Step 2: Optional noise addition (to simulate camera grain)
    if add_noise:
        noise = np.random.normal(0, 0.02, dark.shape)  # small noise
        dark = np.clip(dark + noise, 0, 1)

    # Step 3: Slight desaturation to mimic low-light effect
    hsv = cv2.cvtColor((dark * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv[..., 1] = (hsv[..., 1] * 0.6).astype(np.uint8)  # reduce saturation
    final_dark = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return final_dark

# Loop through all images
for fname in tqdm(os.listdir(input_folder)):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    path = os.path.join(input_folder, fname)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dark_img = darken_image(img, gamma=0.1)

    out_path = os.path.join(output_folder, fname)
    Image.fromarray(dark_img).save(out_path)

print(f"✅ Darkened images saved to: {output_folder}")