# Explainable AI (XAI) Framework for Automated Pneumonia and COVID-19 Diagnostics

[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-v4.x-yellow)](https://huggingface.co)

A comparative medical computer vision and interpretability pipeline evaluating traditional grid-convolutional features (**DenseNet-121**) against global multi-head self-attention networks (**ViT-B16**) for 3-class chest radiograph classification (*COVID-19, Normal, and Viral Pneumonia*).

This framework moves beyond qualitative aesthetics by computing structural localization (**Shannon Entropy**) and operational faithfulness (**Area Over Perturbation Curve - AOPC**) to quantitatively validate visual explanations.

---

## 📂 Repository Structure

```text
ExCV/                            # Main Project Root Workspace
│
├── assets/                      # Visual verification deliverables
│   ├── aopc_evaluation_curve.png
│   ├── densenet_training_curves.png
│   ├── vit_training_curve.png
│   ├── xray_sample.jpg
│   └── xai_comparison.png
│
├── notebooks/                   # Prototyping and pipeline execution
│   └── 01_eda_and_training.ipynb
│
├── .gitignore
├── demo.py
├── README.md
└── requirements.txt
```

---

## 📊 Core Performance & Verification Metrics

### 1. Diagnostic Performance Summary

Both networks were fine-tuned via transfer learning over identical dataset splits at a **224 × 224 × 3** input resolution.

| Model Architecture | Test Loss | Test Accuracy | Test Precision | Test Recall |
|-------------------|-----------|---------------|----------------|-------------|
| **DenseNet-121 (CNN Baseline)** | 0.1866 | **93.79%** | 93.79% | 93.79% |
| **Vision Transformer (ViT-B16)** | 0.3346 | **86.90%** | 87.11% | 84.50% |

### 2. Quantitative Faithfulness Validation

Explanations are rigorously evaluated to verify that highlighted regions correspond to model reasoning rather than diffuse background artifacts.

- **Saliency Map Shannon Entropy (DenseNet-121):** `5.0399 bits`
  - Indicates highly localized attention concentrated within pathological lung regions.

- **Global Dataset AOPC Score:** `0.2589`
  - Computed across 50 test samples using iterative feature-level activation masking at the deepest convolutional representations.

---

## 📸 Visual Explanation Output

The evaluation pipeline generates a side-by-side comparison consisting of:

1. Original Chest X-Ray
2. DenseNet-121 Grad-CAM Heatmap
3. Vision Transformer Attention Rollout Map

This allows qualitative inspection of how convolutional and transformer architectures localize disease-relevant regions.

> Detailed methodology, preprocessing pipelines, training experiments, interpretability analysis, and discussion of Out-of-Distribution (OOD) perturbation artifacts for the bonus research task are documented separately in the accompanying project report.

---

## 🚀 Getting Started

### 1. Clone Repository

```bash
git clone https://github.com/gargi-m21/ExCV-project.git
cd ExCV-project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Training & Evaluation

Launch the complete notebook workflow:

```bash
jupyter notebook notebooks/01_eda_and_training.ipynb
```

The notebook performs:

- Data preprocessing
- Exploratory data analysis (EDA)
- Transfer learning
- DenseNet-121 training
- Vision Transformer training
- Grad-CAM generation
- Attention Rollout generation
- Shannon Entropy computation
- AOPC faithfulness evaluation

---

## 🔍 Demo Inference

Run the standalone diagnostic inference pipeline:

```bash
python demo.py
```

This script:

- Loads saved model checkpoints
- Processes a chest X-ray image
- Generates prediction probabilities
- Produces XAI visualizations
- Saves comparison outputs to the `assets/` directory

---

## 🛠 Tech Stack

- TensorFlow 2.x
- Keras
- Hugging Face Transformers
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Grad-CAM
- Attention Rollout

---

## 📈 Key Findings

- DenseNet-121 achieved the strongest diagnostic performance with **93.79% test accuracy**.
- Vision Transformers produced competitive classification performance while offering global contextual attention.
- Shannon Entropy analysis confirmed concentrated localization behavior in CNN-generated explanations.
- AOPC validation demonstrated measurable faithfulness between highlighted regions and model predictions.
