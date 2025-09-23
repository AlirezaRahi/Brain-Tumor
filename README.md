:

```markdown
# Stacked CNN Architectures for Robust Brain Tumor MRI Classification

This repository contains the implementation of our research on brain tumor classification using MRI scans with ensemble deep learning approaches, as presented in our publication.

**Publication**: [Stacked CNN Architectures for Robust Brain Tumor MRI Classification](https://www.medrxiv.org/content/10.1101/2025.08.05.25333032v1)  
**DOI**: https://doi.org/10.1101/2025.08.05.25333032  
**Publication Date**: August 5, 2025

## Abstract

Brain tumor classification using MRI scans is crucial for early diagnosis and treatment planning. In this study, we first train a single Convolutional Neural Network (CNN) based on VGG16, achieving a strong standalone test accuracy of 99.24% on a balanced dataset of 7,023 MRI images across four classes: glioma, meningioma, pituitary, and no tumor. To further improve classification performance, we implement three ensemble strategies: stacking, soft voting, and XGBoost-based ensembling, each trained on individually fine-tuned models. These ensemble methods significantly enhance prediction accuracy, with XGBoost achieving a perfect 100% accuracy, and voting reaching 99.54%. Evaluation metrics such as precision, recall, and F1-score confirm the robustness of the approach. This work demonstrates the power of combining fine-tuned deep learning models for highly reliable brain tumor classification.

## Key Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| VGG16 (Single) | 99.24% | 99.26% | 99.24% | 99.24% |
| Voting Ensemble | 99.54% | 99.55% | 99.54% | 99.54% |
| XGBoost Ensemble | **100%** | **100%** | **100%** | **100%** |


## Model Architectures

### 1. Fine-Tuned VGG16
- **Base Model**: VGG16 with ImageNet weights
- **Fine-tuning**: Last 8 layers unfrozen for training
- **Custom Head**: GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.5) → Dense(4, Softmax)

### 2. ResNet50 + DenseNet121 Ensemble
- **Base Models**: ResNet50 and DenseNet121 with ImageNet weights
- **Fine-tuning**: Last 30 layers unfrozen for both models
- **Combination**: Features concatenated from both networks
- **Classifier**: Dense(512) → BatchNorm → Dense(256) → BatchNorm → Dropout(0.5) → Dense(4, Softmax)

## Dataset

The BT-MRI dataset contains 7,023 brain MRI images balanced across four classes:
- **Glioma**: 1,626 images
- **Meningioma**: 1,644 images  
- **Pituitary**: 1,450 images
- **No Tumor**: 1,395 images

**Split**: 80% training (5,618 images), 20% testing (1,405 images)

## Installation & Requirements

```bash
# Create virtual environment
python -m venv brain_tumor_env
source brain_tumor_env/bin/activate  # On Windows: brain_tumor_env\Scripts\activate

# Install requirements
pip install tensorflow keras scikit-learn matplotlib seaborn numpy xgboost
```

## Citation

If you use this work in your research, please cite the paper:

> **Rahi, A.** (2025). *Stacked CNN Architectures for Robust Brain Tumor MRI Classification*. medRxiv. https://doi.org/10.1101/2025.08.05.25333032

If you use the code implementation (software, scripts, etc.), please also cite:

> **Rahi, A.** (2025). *Stacked CNN Architectures for Robust Brain Tumor MRI Classification* [Computer software]. GitHub repository, *AlirezaRahi/Brain-Tumor*. Retrieved from https://github.com/AlirezaRahi/Brain-Tumor

## Author
**Alireza Rahi**

- 📧 Email: alireza.rahi@outlook.com
- 💼 LinkedIn: https://www.linkedin.com/in/alireza-rahi-6938b4154/
- 🔗 GitHub: https://github.com/AlirezaRahi

## License

**All Rights Reserved.**

Copyright (c) 2025 Alireza Rahi

Unauthorized access, use, modification, or distribution of this software is strictly prohibited without explicit written permission from the copyright holder.

**For collaboration or access permissions, please contact:**
- 📧 Email: alireza.rahi@outlook.com
- 💼 LinkedIn: https://www.linkedin.com/in/alireza-rahi-6938b4154/

## Model Availability

Due to the sensitive nature of the trained models and to protect the intellectual property of this research, the actual trained model files are not publicly hosted in this repository. 

However, I understand the importance of reproducibility and academic collaboration. **The complete source code for training and evaluation is provided**, allowing researchers to replicate our results exactly.

If you require access to the pre-trained models for:
- Reproducibility verification
- Academic collaboration
- Research comparison
- Educational purposes

Please feel free to contact me directly. I would be happy to share the model files individually under appropriate academic agreements.

**Contact for model requests:**
- 📧 Email: alireza.rahi@outlook.com  
- 💼 LinkedIn: https://www.linkedin.com/in/alireza-rahi-6938b4154/

I typically respond to academic requests within 24-48 hours.
```
