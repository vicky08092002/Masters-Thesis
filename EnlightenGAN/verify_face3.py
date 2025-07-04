from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
import os
from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

model = InceptionResnetV1(pretrained='vggface2').eval()
transform = T.Compose([
    T.Resize((160, 160)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

def get_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = model(img)
    return emb.squeeze(0).numpy()

def evaluate(folder, label):
    print(f"\n🔍 Evaluating on {label} set from: {folder}")
    files = sorted(os.listdir(folder))
    embeddings = {f: get_embedding(os.path.join(folder, f)) for f in files}

    scores, labels = [], []
    for i in range(0, len(files) - 1, 2):  # Pairwise comparison
        f1, f2 = files[i], files[i+1]
        same = f1.split("_")[0] == f2.split("_")[0]
        sim = cosine_similarity([embeddings[f1]], [embeddings[f2]])[0][0]
        scores.append(sim)
        labels.append(1 if same else 0)

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    acc = accuracy_score(labels, [int(s > 0.5) for s in scores])
    f1 = f1_score(labels, [int(s > 0.5) for s in scores])

    print(f"📈 AUC: {roc_auc:.4f}")
    print(f"✅ Accuracy: {acc:.4f}")
    print(f"🎯 F1 Score: {f1:.4f}")

    return {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 'acc': acc, 'f1': f1}

# Evaluate all three sets
res_clean = evaluate('C:/Users/Vignesh M/PycharmProjects/Thesis/dataset/lfw_test_clean', 'Clean')
res_dark = evaluate('C:/Users/Vignesh M/PycharmProjects/Thesis/dataset/lfw_test_dark', 'Dark')
res_enh = evaluate('C:/Users/Vignesh M/PycharmProjects/Thesis/dataset/retinex_output', 'Retinex')

# Optionally: plot ROC curves
plt.figure()
plt.plot(res_clean['fpr'], res_clean['tpr'], label=f'Clean (AUC = {res_clean["auc"]:.2f})')
plt.plot(res_dark['fpr'], res_dark['tpr'], label=f'Dark (AUC = {res_dark["auc"]:.2f})')
plt.plot(res_enh['fpr'], res_enh['tpr'], label=f'Retinex (AUC = {res_enh["auc"]:.2f})')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()

from sklearn.decomposition import PCA

def get_embeddings_for_pca(folder, label, max_images=30):
    files = sorted(os.listdir(folder))[:max_images]
    embeddings = []
    labels = []
    for file in files:
        try:
            emb = get_embedding(os.path.join(folder, file))
            embeddings.append(emb)
            labels.append(label)
        except:
            continue
    return embeddings, labels

print("\n🔍 Generating PCA visualization...")

# Collect up to 30 embeddings from each category to reduce memory
clean_embs, clean_labels = get_embeddings_for_pca('C:/Users/Vignesh M/PycharmProjects/Thesis/dataset/lfw_test_clean', 'Clean')
dark_embs, dark_labels = get_embeddings_for_pca('C:/Users/Vignesh M/PycharmProjects/Thesis/dataset/lfw_test_dark', 'Dark')
enh_embs, enh_labels = get_embeddings_for_pca('C:/Users/Vignesh M/PycharmProjects/Thesis/dataset/retinex_output', 'Retinex')

all_embs = np.array(clean_embs + dark_embs + enh_embs)
all_labels = clean_labels + dark_labels + enh_labels

pca = PCA(n_components=2)
reduced = pca.fit_transform(all_embs)

# Plotting
colors = {'Clean': 'blue', 'Dark': 'red', 'Classical': 'green'}
plt.figure(figsize=(10, 7))
for label in set(all_labels):
    idx = [i for i, l in enumerate(all_labels) if l == label]
    plt.scatter(reduced[idx, 0], reduced[idx, 1], label=label, alpha=0.6, color=colors[label])

plt.legend()
plt.title("🧠 PCA Visualization of Face Embeddings")
plt.xlabel("PCA-1")
plt.ylabel("PCA-2")
plt.grid(True)
plt.tight_layout()
output_path = "C:/Users/Vignesh M/PycharmProjects/Thesis/embedding_pca2.png"
plt.savefig(output_path)
print(f"📊 PCA visualization saved to: {output_path}")
