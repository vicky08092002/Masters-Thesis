import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

class LFWLowLightDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        if not isinstance(root_dir, str) or not os.path.exists(root_dir):
            raise ValueError(f"Provided path '{root_dir}' is invalid or does not exist.")

        # Only accept .jpg files
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith('.jpg')
        ]

        if not self.image_paths:
            raise FileNotFoundError(f"No .jpg images found in directory: {root_dir}")

        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Create label mapping
        self.name_to_id = {}
        self.labels = []
        self._build_label_mapping()

    def _build_label_mapping(self):
        names = [self._extract_name(p) for p in self.image_paths]
        unique_names = sorted(set(names))
        self.name_to_id = {name: idx for idx, name in enumerate(unique_names)}
        self.labels = [self.name_to_id[self._extract_name(p)] for p in self.image_paths]

    def _extract_name(self, path):
        basename = os.path.basename(path)
        parts = basename.split("_")
        return "_".join(parts[:-1])  # Exclude the image index part

    def simulate_low_light(self, image, gamma=0.3):
        return np.clip(255.0 * (image / 255.0) ** gamma, 0, 255).astype(np.uint8)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"❌ Could not read image at: {img_path}")
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected 3-channel image, but got shape {img.shape} at: {img_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Simulate low-light version
        low_light = self.simulate_low_light(img)

        # Apply transforms
        rgb_tensor = self.transform(img)
        low_light_tensor = self.transform(low_light)

        # Grayscale channel
        gray = TF.rgb_to_grayscale(rgb_tensor)

        # 4-channel input: RGB + grayscale
        combined = torch.cat((rgb_tensor, gray), dim=0)

        return low_light_tensor, rgb_tensor, label, combined, os.path.basename(img_path)

    def __len__(self):
        return len(self.image_paths)
