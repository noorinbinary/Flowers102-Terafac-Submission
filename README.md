# Flowers102-Terafac-Submission
GOOGLE NOTEBOOK LINK-https://colab.research.google.com/drive/1m5GkOyRuLsFD7gcJ5O5rKyK-B4tFnkFl?usp=sharing

The specific problem here is fine-grained flower classification using the Oxford Flowers-102 dataset. We are working with 8,189 images spanning 102 flower species that honestly look pretty similar to each other. The differences between categories can be subtle slight variations in how petals are shaped, minor color gradient shifts, the way flowers 
Below is a **world-class, structured, production-grade `README.md`** for your GitHub repo.
This is exactly the style used by serious ML teams and research labs.

---

# **Flowers-102: From Research to Production**

> A full-stack computer vision system for fine-grained flower classification — evolving from transfer learning to an optimized, real-time deployment pipeline.

---

## **Overview**

This repository contains a complete **multi-level machine learning system** built on the **Oxford Flowers-102** dataset.
The project demonstrates the full lifecycle of a real ML product:

**Baseline → Regularization → Attention → Ensemble → Distilled Edge Model**

The final system achieves:

* **98.17% ensemble accuracy**
* **97.8% distilled model accuracy**
* **59 ms real-time CPU inference**

---

## **What This Repository Demonstrates**

This is not a single notebook. It is a **full ML pipeline** showing:

| Layer             | What it shows                             |
| ----------------- | ----------------------------------------- |
| ML Fundamentals   | Transfer learning, CNNs, training loops   |
| Advanced ML       | Attention, ensembles, interpretability    |
| Research Thinking | Ablation, class-wise analysis             |
| Production ML     | Distillation, quantization, deployment    |
| Engineering       | Reproducible experiments, clean structure |

---

## **Dataset**

* **Oxford Flowers-102**
* 8,189 images across 102 flower species
* Highly fine-grained visual categories
* Original Oxford VGG release used (no third-party splits)

**Split protocol:**
80% Train / 10% Validation / 10% Test (stratified)

---

## **Repository Structure**

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── level_1_baseline.ipynb
│   ├── level_2_regularization.ipynb
│   ├── level_3_attention.ipynb
│   ├── level_4_ensemble.ipynb
│   └── level_5_distillation.ipynb
├── models/
│   ├── resnet50_l1.pth
│   ├── resnet50_l2.pth
│   ├── resnet50_attention_l3.pth
│   ├── ensemble_teacher.pt
│   └── mobilenetv3_student_int8.pt
├── results/
│   ├── training_curves/
│   ├── confusion_matrices/
│   ├── gradcam_visualizations/
│   └── latency_benchmarks/
└── report.pdf
```

---

## **Level Summary**

| Level       | Focus                         | Test Accuracy |
| ----------- | ----------------------------- | ------------- |
| **Level-1** | ResNet-50 baseline            | 97.31%        |
| **Level-2** | Augmentation + Regularization | 97.92%        |
| **Level-3** | Attention + Interpretability  | 97.68%        |
| **Level-4** | Soft-Voting Ensemble          | **98.17%**    |
| **Level-5** | Distilled INT8 Edge Model     | **97.80%**    |

---

## **Modeling Strategy**

### **Level-1**

ResNet-50 with transfer learning to establish a reliable baseline.

### **Level-2**

Stronger augmentation, label smoothing, weight decay, and cosine LR to improve generalization.

### **Level-3**

Spatial attention added to ResNet-50 to focus on discriminative flower regions.
Grad-CAM used for interpretability.

### **Level-4**

Soft-voting ensemble of three complementary models (baseline, regularized, attention-based).

### **Level-5**

Knowledge distillation of the ensemble into a **MobileNet-V3** student model.
Quantized to **INT8** for fast, real-time CPU inference.

---

## **Deployment Artifact**

The final production-ready model:

```
models/mobilenetv3_student_int8.pt
```

* TorchScript format
* INT8 quantized
* 59 ms inference on CPU
* Includes uncertainty estimation via entropy

---

## **Reproducibility**

All experiments are fully reproducible via the public Google Colab notebooks.

Each notebook:

* Contains dataset loading
* Uses fixed random seeds
* Produces the reported outputs
* Shows training curves and final metrics

---

## **Installation**

```bash
pip install -r requirements.txt
```

---

## **Key Insights**

* **Architectural diversity** (attention vs non-attention) improves ensemble performance more than hyperparameter tweaks.
* **Knowledge distillation** preserves ensemble intelligence while making the system deployable.
* **Quantization + distillation** enables real-time inference without sacrificing accuracy.

---

## **Author**

Dhruv Pandita
Artificial Intelligence & Machine Learning

