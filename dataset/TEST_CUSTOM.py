import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from models.enlightengan import EnlightenGAN  # Assuming this model file exists
import matplotlib.pyplot as plt

class FaceLowLightDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])

    def simulate_low_light(self, image, gamma=0.4):
        return np.clip(255.0 * (image / 255.0) ** gamma, 0, 255).astype(np.uint8)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        low_light = self.simulate_low_light(img)
        return self.transform(low_light), os.path.basename(self.image_paths[idx])

    def __len__(self):
        return len(self.image_paths)

# Load pretrained EnlightenGAN model
model = EnlightenGAN()
model.load_state_dict(torch.load("checkpoints/enlightengan.pth"))  # Update with correct path
model.eval()

# Dataset
dataset = FaceLowLightDataset("dataset/train")
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Enhance and Save
os.makedirs("outputs", exist_ok=True)
for i, (low_light, name) in enumerate(loader):
    with torch.no_grad():
        enhanced = model(low_light)
        enhanced_img = transforms.ToPILImage()(enhanced.squeeze(0).cpu())
        enhanced_img.save(f"outputs/enhanced_{name[0]}")