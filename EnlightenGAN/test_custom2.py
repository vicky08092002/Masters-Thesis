import torch
import os
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.datasets import folder

# ✅ Import the actual generator class
from models.networks import Unet_resize_conv
class TestOpt:
    def __init__(self):
        self.input_nc = 4
        self.output_nc = 3
        self.use_norm = 1
        self.syn_norm = 0
        self.use_avgpool = 0
        self.self_attention = True
        self.skip = False
        self.linear_add = False
        self.latent_threshold = False
        self.latent_norm = False
        self.times_residual = False
        self.tanh = True
        self.linear = False

opt = TestOpt()
model = Unet_resize_conv(opt, skip=opt.skip)

class FaceLowLightDataset(Dataset):
    def __init__(self, root_dir):
        valid_ext = ['.jpg', '.jpeg', '.png']
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        ]

        if not self.image_paths:
            raise FileNotFoundError("❌ No valid images found in the directory!")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def simulate_low_light(self, image, gamma=0.4):
        return np.clip(255.0 * (image / 255.0) ** gamma, 0, 255).astype(np.uint8)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)

        if img is None:
            raise FileNotFoundError(f"❌ Could not read image at: {img_path}")
        if img.shape[2] != 3:
            raise ValueError(f"❌ Expected 3 channels but got shape {img.shape} at {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        low_light = self.simulate_low_light(img)

        rgb = self.transform(low_light)
        gray = transforms.functional.rgb_to_grayscale(rgb)

        input_tensor = torch.cat((rgb, gray), dim=0)
        return input_tensor, os.path.basename(img_path)

    def __len__(self):
        return len(self.image_paths)

    print("📁 Found images:")
    for f in os.listdir("C:/Users/Vignesh M/PycharmProjects/Thesis/dataset/train"):
        print(" -", f)


# ✅ Load the Generator
ckpt = torch.load("checkpoints/enlightengan/200_net_G_A.pth", map_location="cpu")
# For checkpoints saved using `nn.DataParallel`
if "module.conv1_1.weight" in ckpt:
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(ckpt)
model.eval()

# Dataset
dataset = FaceLowLightDataset("C:/Users/Vignesh M/PycharmProjects/Thesis/dataset/train")
loader = DataLoader(dataset, batch_size=1)

# Output folder
os.makedirs("outputs", exist_ok=True)

# Enhancement
for input_tensor, filename in loader:
    input_tensor = input_tensor.to(torch.float32)  # ensure type
    rgb = input_tensor[:, :3, :, :]  # [1, 3, 256, 256]
    gray = input_tensor[:, 3:, :, :]  # [1, 1, 256, 256]

    print("💡 conv1_1 expects input channels:", input_tensor.shape[1])  # Should print 4

    with torch.no_grad():
        enhanced = model(input_tensor, gray)  # <- now it should work fine
        enhanced_img = transforms.ToPILImage()(enhanced.squeeze(0).cpu().clamp(0,1))
        enhanced_img.save(f"outputs/enhanced_{filename[0]}")

