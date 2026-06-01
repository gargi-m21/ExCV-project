# Explainable AI (XAI) Frameworks for Automated Pneumonia and COVID-19 Detection in Chest Radiography

[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-v4.x-yellow)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains a comparative machine learning pipeline evaluating traditional **Convolutional Neural Networks (CNNs)** against modern **Vision Transformers (ViTs)** for multi-class chest X-ray classification (*COVID-19, Normal, Viral Pneumonia*). 

To bridge the gap between black-box deep learning models and clinical trust, this project implements a rigorous **Explainable AI (XAI)** evaluation suite comparing local feature maps (**Grad-CAM**) against global multi-head token correlations (**Attention Rollout**), validated quantitatively using Shannon Entropy and Area Over Perturbation Curves (AOPC).

---

## 📊 Core Performance & Verification Metrics

### 1. Diagnostic Performance Summary
| Model Architecture | Parameter Scaling | Unseen Test Accuracy | Test Precision | Test Recall |
| :--- | :--- | :--- | :--- | :--- |
| **DenseNet-121 (CNN Baseline)** | ~7.04 M (3.1K Trainable) | **93.79%** | 93.79% | 93.79% |
| **Vision Transformer (ViT-B16)** | ~85.8 M (2.3K Trainable) | **86.90%** | 87.11% | 84.50% |

### 2. Quantitative Faithfulness Verification
Visual heatmaps can occasionally be deceptive or highlight random artifacts. To mathematically prove that the models rely on actual lung pathologies, the saliency maps were verified using two rigorous evaluation criteria:
* **Saliency Map Shannon Entropy (DenseNet-121):** `5.0399 bits` (Indicates a highly localized, structurally focused explanation pattern within lung regions rather than diffuse global noise).
* **Global Dataset AOPC Score:** `0.2589` (Area Over Perturbation Curve computed dynamically over 50 random test samples via feature-level activation masking at the terminal `relu` grid layer).

---

## 📸 Visual Comparison Output

Below is a diagnostic comparison generated from the test set pipeline showing the original radiograph alongside its corresponding CNN and Transformer XAI maps:

![XAI Structural Matrix Overlay](assets/xai_comparison.png)

### Key Interpretations for the Report:
* **DenseNet-121 (Grad-CAM):** Demonstrates strong local feature capture, isolating high-frequency boundaries and specific patches of consolidation/ground-glass opacities in the lung fields.
* **Vision Transformer (Attention Rollout):** Demonstrates broad global context tracking, highlighting structural contours across the chest cavity and indicating how distinct anatomical regions correlate to inform final global predictions.

---

## 📂 Repository Structure

```text
├── assets/                  # High-resolution visualization charts and plots
│   └── xai_comparison.png   # Three-panel visual baseline overlay matrix
├── notebooks/               # Interactive prototyping pipelines
│   └── 01_eda_and_training.ipynb  # Main data loader, training, and XAI code
├── .gitignore               # Excludes large checkpoint files and data
├── requirements.txt         # Package dependencies for immediate setup
└── README.md                # Project documentation