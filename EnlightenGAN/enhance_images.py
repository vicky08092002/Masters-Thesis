import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
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

ckpt = torch.load("checkpoints/enlightengan/200_net_G_A.pth", map_location="cpu")
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

class FaceLowLightDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])

    def simulate_low_light(self, image, gamma=0.35):
        image = np.asarray(image) / 255.0
        dark = np.power(image, gamma)
        return (np.clip(dark * 255, 0, 255)).astype(np.uint8)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        low_light = self.simulate_low_light(img)
        low_light_pil = Image.fromarray(low_light)
        rgb_tensor = self.transform(low_light_pil)
        gray_tensor = transforms.functional.rgb_to_grayscale(rgb_tensor)
        input_tensor = torch.cat((rgb_tensor, gray_tensor), dim=0)
        return input_tensor, os.path.basename(self.image_paths[idx])

    def __len__(self):
        return len(self.image_paths)

dataset = FaceLowLightDataset("C:/Users/Vignesh M/PycharmProjects/Thesis/dataset/lfw_test_dark")
loader = DataLoader(dataset, batch_size=1)
output_path = "C:/Users/Vignesh M/PycharmProjects/Thesis/dataset/lfw_test_enhanced"
os.makedirs(output_path, exist_ok=True)

for input_tensor, filename in loader:
    input_tensor = input_tensor.to(torch.float32)
    rgb = input_tensor[:, :3, :, :]
    gray = input_tensor[:, 3:, :, :]
    with torch.no_grad():
        enhanced = model(input_tensor, gray)
        enhanced = torch.clamp(enhanced, 0, 1)
        img_np = (enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        final_img = Image.fromarray(img_eq)
        final_img.save(os.path.join(output_path, filename[0]))

