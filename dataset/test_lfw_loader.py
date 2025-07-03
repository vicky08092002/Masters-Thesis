from dataset.lfw_loader import LFWLowLightDataset
import matplotlib.pyplot as plt

dataset = LFWLowLightDataset("C:/Users/Vignesh M/PycharmProjects/Thesis/dataset/train")

low_light, original, label = dataset[0]

plt.subplot(1, 2, 1)
plt.imshow(low_light.permute(1, 2, 0))
plt.title("Low-Light Image")

plt.subplot(1, 2, 2)
plt.imshow(original.permute(1, 2, 0))
plt.title(f"Original (ID: {label})")
plt.show()
