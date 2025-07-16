
# Low-Light Facial Recognition Enhancement Pipeline

## Overview

This repository presents the complete implementation of a low-light image enhancement pipeline aimed at improving the performance of face verification systems under poor lighting conditions. This is the official codebase for the Master’s thesis titled:

**"Low-Light Enhancement for Robust Face Verification: A Comparative Evaluation of Classical and Deep Learning Methods"**

## 🔍 Problem Statement

Facial recognition systems often struggle with low-light images due to poor visibility and loss of identity-preserving features. The goal of this project is to improve verification accuracy on such images by applying and evaluating several enhancement techniques.

## 🎯 Objectives

- Assess the degradation in face verification performance under dark conditions.
- Compare three image enhancement methods:
  - **Histogram Equalization (HE)**
  - **EnlightenGAN**
  - **RetinexNet**
- Evaluate the impact of enhancement on identity-preserving facial features using **FaceNet** for verification.

## 📁 Folder Structure

```
├── data/                    # Contains raw and enhanced image data
│   ├── test/low/            # Input low-light images
│   ├── test/retinex_output/ # RetinexNet enhanced images
├── model/                   # Pre-trained models for RetinexNet
├── EnlightenGAN/            # Placeholder (requires manual setup due to submodule issues)
├── run_lowlight.py          # Runs RetinexNet on input images
├── facenet_verify.py        # Performs face verification using FaceNet
├── requirements.txt         # Required Python dependencies
├── README.md                # Project overview (this file)
```

## 🧠 Methods Used

1. **FaceNet** for face embedding generation and cosine similarity.
2. **HE, EnlightenGAN, RetinexNet** for image enhancement.
3. **Gamma Correction** to simulate low-light conditions.
4. **Principal Component Analysis (PCA)** for visualizing embeddings.
5. **Verification Metrics:** Cosine similarity, AUC, F1-score, accuracy.

## 📊 Experimental Summary

- Dataset: LFW (Labeled Faces in the Wild)
- Evaluation:
  - Clean-Clean pairs
  - Dark-Dark pairs
  - Enhanced-Dark pairs
- RetinexNet achieved the best identity preservation with:
  - AUC: 0.89
  - F1-Score: 0.84
  - Accuracy: 85.3%

## 💡 Key Findings

- HE introduces artifacts and reduces fidelity.
- EnlightenGAN is visually appealing but inconsistent in preserving identity.
- RetinexNet offers the best trade-off between enhancement and identity retention.
- Real-world low-light images showed similar trends but with slightly reduced performance due to uncontrolled lighting.

## 🛠️ Setup Instructions

```bash
git clone https://github.com/vicky08092002/Masters-Thesis.git
cd Masters-Thesis
pip install -r requirements.txt
```

⚠️ EnlightenGAN is currently an empty folder. Please clone the original repo separately:
```bash
git clone https://github.com/yueruchen/EnlightenGAN.git
mv EnlightenGAN ./EnlightenGAN
```

## 🚀 Run Enhancement

```bash
python run_lowlight.py
```

## 🧪 Run Verification

```bash
python facenet_verify.py
```

## 📌 Citation

If you use this project, please cite the thesis or drop a star ⭐ on this repository!

---

**Author:** Vignesh Muralidharan  
**Programme:** MSc Data Analytics  
**Supervisor:** Dr. Joseph Lemley
