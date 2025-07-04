import os

import epoch
import torch
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from EnlightenGAN.models.single_model import SingleModel
from models.single_model import SingleModel  # Assuming this model file exists
import matplotlib.pyplot as plt

class TestOpt:
    def __init__(self):
        self.batchSize = 1
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.fineSize = 256
        self.which_epoch = '200'
        self.gpu_ids = []  # Use [] for CPU
        self.isTrain = False
        self.name = 'enlightengan'
        self.model = 'single'
        self.checkpoints_dir = 'checkpoints'
        self.norm = 'batch'
        self.init_type = 'normal'
        self.no_dropout = True
        self.vgg = 0
        self.lambda_A = 10.0
        self.lambda_B = 10.0
        self.lambda_identity = 0.5
        self.pool_size = 50
        self.direction = 'AtoB'
        self.skip = 1
        self.display_id = 0
        self.n_layers_D = 3
        self.niter = 100
        self.niter_decay = 100
        self.lr = 0.0002
        self.beta1 = 0.5
        self.lr_policy = 'lambda'
        self.lr_decay_iters = 50
        self.no_lsgan = False
        self.continue_train = False
        self.load_size = 256
        self.crop_size = 256
        self.which_model_netG = 'unet_256'  # ✅ NEW LINE
        self.use_norm = 1
        self.gpu_ids = []
        self.fcn = 0

opt = TestOpt()

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
# Initialize model
model = SingleModel()
model.initialize(opt)
model.setup(opt)
model.eval()

# Dataset
dataset = FaceLowLightDataset("C:/Users/Vignesh M/PycharmProjects/Thesis/dataset/train")
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Enhance and Save
os.makedirs("outputs", exist_ok=True)
for i, (img_tensor, name) in enumerate(loader):
    data = {'A': img_tensor}
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    enhanced = visuals['fake_B']

    # Save image
    enhanced_img = transforms.ToPILImage()(enhanced.squeeze(0).cpu())
    enhanced_img.save(f"outputs/enhanced_{name[0]}")