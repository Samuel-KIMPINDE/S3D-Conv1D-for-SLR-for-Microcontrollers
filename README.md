# Efficient Word-Level Sign Language Recognition Using Quantized Deep Learning for MCU Deployment

Real-time, privacy-preserving sign language recognition (SLR) running directly on microcontrollers — no cloud, no network, no compromise on accessibility.

![Alt text]("C:\Users\kimpi\Downloads\Real-time sign language recognition system.png")



This repository provides a reproducible pipeline for training S3D-Conv1D, a lightweight spatiotemporal sign language recognition model, quantizing it to INT8, and profiling it on resource-constrained edge device (NUCLEO-H753ZI board). It bridges deep learning research with real-world embedded constraints through methodological insights for TinyML deployment in vision-based tasks.

---

## Why This Matters

SLR systems running on edge devices can deliver accessibility anywhere — affordable, portable, and private. On-device inference brings scalability, energy efficiency, and data sovereignty to wearable, mobile, and smart environments. This work advances both assistive technology and the broader TinyML field by demonstrating that quantized spatiotemporal models can run viably on microcontrollers.

---

## Datasets

| Dataset | Source | Role |
|---|---|---|
| **WLASL Processed** | [Kaggle](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed) | Primary video source |
| **WLASL2000 Resized** | [Kaggle](https://www.kaggle.com/datasets/sttaseen/wlasl2000-resized) | Backup videos + class list + JSON metadata |
| **ASL-Kimpinde Set** | [Zenodo](https://zenodo.org/records/18754451) | Custom domain-aware augmentation set |

### ASL-Kimpinde Set

A custom dataset, collected by Samuel L. Kimpinde under controlled conditions, was collected with two roles: augmenting underrepresented WLASL classes and providing signer-specific calibration data for quantization. This illustrates a broader principle: signer variation is a genuine source of distribution shift, and domain-aware calibration is a necessary step when moving from benchmark evaluation to individual and real-world use on microcontrollers.

- **250 video samples** across **50 isolated ASL word classes** (5 samples/class)
- Recorded on a Lenovo V15 G2 webcam at **1280×720, 30 FPS**
- Metadata includes filename, class label, environment, and recording parameters
- Capture scripts and a README are included for full reproducibility

---

## Model Architecture — S3DConv1D

A lightweight spatiotemporal architecture designed for MCU deployment:

```
Input: (24 frames × 64×64 × 1 channel)
  │
  ├─ Spatial Encoder (per-frame)
  │    ├─ Conv2D(12, 3×3, ReLU) + MaxPool(2×2)
  │    ├─ Conv2D(24, 3×3, ReLU) + MaxPool(2×2)
  │    └─ Flatten
  │
  ├─ Temporal Encoder
  │    ├─ Sequence S
  │    ├─ Conv1D(48, k, ReLU) 
  │    └─ GlobalAveragePooling1D
  │
  └─ Classifier
       └─ Dense(num_classes, Softmax)
```


The spatial backbone extracts per-frame features; Conv1D captures motion patterns across the temporal dimension. The result is a model small enough to quantize to INT8 and deploy on a microcontroller.

---

## Quantization

INT8 quantization uses **quantization aware-training** vs **post-training quantization**.

---

## Citation

If you use this work or the ASL-Kimpinde dataset, please cite:

```bibtex

@Article{a19040248,
AUTHOR = {Kimpinde, Samuel Longwani and Olukanmi, Peter O.},
TITLE = {Efficient Word-Level Sign Language Recognition Using Quantized Spatiotemporal Deep Learning for Low-Power Microcontrollers},
JOURNAL = {Algorithms},
VOLUME = {19},
YEAR = {2026},
NUMBER = {4},
ARTICLE-NUMBER = {248},
URL = {https://www.mdpi.com/1999-4893/19/4/248},
ISSN = {1999-4893},
DOI = {10.3390/a19040248}
}


@dataset{Kimpinde2026ASL,
  author    = {Kimpinde, Samuel L.},
  title     = {ASL-Kimpinde set: Minimal Dataset for Sign Language Recognition},
  year      = {2026},
  version   = {1.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18754451},
  url       = {https://doi.org/10.5281/zenodo.18754451}
}

```
