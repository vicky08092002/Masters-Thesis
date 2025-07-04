import cv2
import os
from PIL import Image

input_dir = "C:/Users/Vignesh M/PycharmProjects/Thesis/dataset/lfw_test_dark"
output_dir = "C:/Users/Vignesh M/PycharmProjects/Thesis/dataset/lfw_test_classical"
os.makedirs(output_dir, exist_ok=True)

for f in os.listdir(input_dir):
    img = cv2.imread(os.path.join(input_dir, f))
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    Image.fromarray(cv2.cvtColor(img_eq, cv2.COLOR_BGR2RGB)).save(os.path.join(output_dir, f))