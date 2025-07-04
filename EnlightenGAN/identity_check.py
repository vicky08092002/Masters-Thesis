import os
from facenet_pytorch import InceptionResnetV1
import torch
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

model = InceptionResnetV1(pretrained='vggface2').eval()
transform = T.Compose([
    T.Resize((160, 160)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        return model(img).squeeze(0).numpy()

base_path = 'C:/Users/Vignesh M/PycharmProjects/Thesis/dataset/lfw_test_dark'
enhanced_path = 'C:/Users/Vignesh M/PycharmProjects/Thesis/dataset/lfw_test_enhanced'

samples = sorted(os.listdir(base_path))[:10]

for file in samples:
    dark_emb = get_embedding(os.path.join(base_path, file))
    enh_emb = get_embedding(os.path.join(enhanced_path, file))
    sim = cosine_similarity([dark_emb], [enh_emb])[0][0]
    print(f"{file}: Cosine Similarity = {sim:.4f}")