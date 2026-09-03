# Chest X-Ray Pneumonia Classifier

A transfer learning pipeline that fine-tunes VGG16 (ImageNet-pretrained) to detect pneumonia from chest X-ray images, deployed as an interactive Flask web app.

**Live demo:** [add link after deployment]
**Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Kermany et al.

## Overview

This project replicates the transfer learning methodology used in pneumonia detection research (e.g. Rajpurkar et al.'s CheXNet, Rahaman et al. 2020) — taking an ImageNet-pretrained CNN and fine-tuning it on a small, domain-specific medical imaging dataset.

## Approach

- **Architecture:** VGG16, partial fine-tuning (layers from `block5_conv1` onward unfrozen, earlier layers frozen)
- **Head:** GlobalAveragePooling2D → Dropout → Dense(128, relu) → Dropout → Dense(1, sigmoid)
- **Preprocessing:** aspect-ratio-preserving resize to 224×224, pixel normalization using dataset-specific mean/std
- **Data augmentation:** horizontal flip, small rotation, zoom, contrast jitter
- **Class imbalance handling:** oversampling of the minority (Normal) class on the training split only
- **Validation:** stratified 15% hold-out from the training set (the dataset's provided validation folder was too small — 16 images — to give a reliable signal)

## Results

| Metric | Default threshold (0.5) | Tuned threshold |
|---|---|---|
| Accuracy | **95%** |
| Normal recall | 0.30 | 0.90 |
| Pneumonia recall | 1.00 | 0.92 |
| AUC | 0.99 | 0.99 |

**Key finding:** the model showed strong class separation (AUC 0.99) but a shifted decision boundary — raw sigmoid outputs skewed high for both classes. Diagnosed this via probability distribution analysis and ruled out simple overconfidence (temperature scaling had no effect on the 0.5-threshold result), confirming the issue was boundary placement rather than calibration. Corrected using ROC-curve threshold tuning (Youden's J statistic), which found the optimal cutoff at 0.999.

## Project structure
