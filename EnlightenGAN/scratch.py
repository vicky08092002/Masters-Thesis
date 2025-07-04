from facenet_pytorch import InceptionResnetV1
import torch
import torchvision.transforms as T
from PIL import Image

# Load model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Define transform
transform = T.Compose([
    T.Resize((160, 160)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

def get_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = model(img_tensor)
    return emb.squeeze(0)


import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

dark_dir = "C:/Users/Vignesh M/PycharmProjects/Thesis/dataset/lfw_test_dark"
classical_dir = "C:/Users/Vignesh M/PycharmProjects/Thesis/dataset/lfw_test_classical"
clean_dir = "C:/Users/Vignesh M/PycharmProjects/Thesis/dataset/lfw_test_clean"

# Pick 10 people by sorting filenames
files = sorted(os.listdir(clean_dir))
unique_names = sorted(set([f.split("_")[0] for f in files]))
selected_names = unique_names[:10]  # You can randomize if you want

for name in selected_names:
    try:
        # Get images for this identity
        clean_file = next(f for f in files if f.startswith(name))
        dark_file = clean_file
        classical_file = clean_file

        clean_path = os.path.join(clean_dir, clean_file)
        dark_path = os.path.join(dark_dir, dark_file)
        classical_path = os.path.join(classical_dir, classical_file)

        # Get embeddings
        emb_clean = get_embedding(clean_path)
        emb_dark = get_embedding(dark_path)
        emb_cla = get_embedding(classical_path)

        # Cosine similarities
        sim_clean_dark = cosine_similarity([emb_clean], [emb_dark])[0][0]
        sim_clean_cla = cosine_similarity([emb_clean], [emb_cla])[0][0]

        # Show images and similarities
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, path, title in zip(axes, [clean_path, dark_path, classical_path],
                                   [f"Clean\nSim=1.0",
                                    f"Dark\nSim={sim_clean_dark:.3f}",
                                    f"Enhanced\nSim={sim_clean_cla:.3f}"]):
            ax.imshow(Image.open(path))
            ax.axis("off")
            ax.set_title(title)

        plt.suptitle(f"Person: {name}", fontsize=14)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"⚠️ Error with {name}: {e}")
