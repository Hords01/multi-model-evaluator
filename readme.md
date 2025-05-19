# 🧠 Multi-Model Classification for Chest X-Ray Images

This repository provides a modular and extensible training pipeline for comparing multiple deep learning models on a chest X-ray dataset with **four distinct classes**. The primary goal is to evaluate different CNN architectures in terms of accuracy, training efficiency, and diagnostic performance using various classification metrics.

## 📌 Features

- ✅ Supports 27+ CNN architectures from `torchvision.models`:
  - ResNet (18, 34, 50, 101, 152)
  - DenseNet (121, 161, 169, 201)
  - VGG (11, 13, 16, 19 with/without batch norm)
  - MobileNetV2, MobileNetV3 (Large/Small)
  - EfficientNet (B0, B1)
  - AlexNet, SqueezeNet (1.0, 1.1)
  - ShuffleNetV2 (x0.5, x1.0)
  - GoogLeNet, InceptionV3 (with auxiliary heads)
- 🏥 Classifies chest X-ray images into four categories (e.g., Normal, Pneumonia, COVID-19, Tuberculosis)
- 📊 Tracks key evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
- 💾 Saves:
  - Model weights (`.pt`)
  - Training history as CSV
  - Confusion matrix images
  - Final comparison summary across models
- ⚙️ Handles special architectures like Inception and GoogLeNet (auxiliary outputs)
- 📈 Automatically generates plots of training/validation metrics

## 📂 Dataset

The dataset used for this project can be found on Kaggle:

> 🔗 [Chest X-ray Pneumonia, COVID-19, Tuberculosis Dataset – Kaggle](https://www.kaggle.com/)

### Folder Structure

Before training, structure the dataset as follows:
```
data/
├── train/
│ ├── Normal/
│ ├── Pneumonia/
│ ├── COVID-19/
│ └── Tuberculosis/
├── val/
│ ├── Normal/
│ └── ...
└── test/
├── Normal/
└── ...
```
Each subdirectory should contain images belonging to that class. This layout is compatible with PyTorch's `ImageFolder`.

---


### 📊 Outputs

After training, the following files will be saved inside the results/ directory:
```
results/
├── resnet18_history.csv               # Training & validation metrics
├── resnet18_confusion_matrix.png     # Confusion matrix image
├── resnet18.pt                        # Trained model weights
├── ...
summary.csv                            # Final metric comparison across all models
```
---

### 🧪 Evaluation Metrics

All models are evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**
---


### 🧾 Citation / Attribution
If you use this repository, code, or any part of it in your research, publication, or project, please cite it or provide appropriate credit:

Emirkan Beyaz, "Multi-Model Evaluator", GitHub Repository, 2025.  
https://github.com/Hords01/multi-model-evaluator

### ✉️ Contact
For questions, issues, or collaboration opportunities, feel free to open an Issue or start a Discussion on GitHub. mail:beyazemirkan@gmail.com
