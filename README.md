# Efficient Word-Level Sign Language Recognition Using Quantized Deep Learning for MCU Deployment

Real-time, privacy-preserving sign language recognition (SLR) running directly on microcontrollers — no cloud, no network, no compromise on accessibility.

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

## Setup & Usage

### 1. Install Dependencies

```bash
pip install tensorflow opencv-python vidaug scikit-learn pandas matplotlib seaborn tqdm
```

### 2. Configure Your Environment

Open `config.py` and verify the active environment, or set it via shell:

```bash
# Kaggle (default — no change needed)
python train_s3dconv1d.py

# Local machine
export PIPELINE_ENV=local
python train_s3dconv1d.py

# Google Colab
export PIPELINE_ENV=colab
python train_s3dconv1d.py
```

For `local` or `colab`, edit the matching path block inside `config.py` to point to your data directories.

### 3. Run Training

```bash
python train_s3dconv1d.py
```

This will:
- Validate and balance videos from both datasets
- Extract grayscale frames at 64×64, apply augmentation and FPS simulation
- Split data into train / val / test (70 / 20 / 10)
- Train the S3DConv1D model for up to 100 epochs with early stopping
- Save the best checkpoint, generate training plots, and produce a full classification report


## Key Hyperparameters

All values live in `config.py` — edit there, not in the individual scripts.

| Parameter | Default | Description |
|---|---|---|
| `IMG_SIZE` | 64 | Frame resolution (px) |
| `NUM_FRAMES` | 24 | Temporal length per sample |
| `FPS_TARGETS` | [10, 15, 30, 50] | FPS simulation variants for augmentation |
| `BATCH_SIZE` | 32 | Training batch size |
| `EPOCHS` | 100 | Max training epochs |
| `MIN_VIDEOS_PER_CLASS` | 18 | Minimum videos required to include a class |
| `MAX_VIDEOS_PER_CLASS` | 25 | Cap per class for dataset balance |

---

## Citation

If you use this work or the ASL-Kimpinde dataset, please cite:

```bibtex

@software{S3DConv1D,
  author    = {Kimpinde, Samuel L. and Olukanmi, Peter O.},
  title     = {Samuel-KIMPINDE/S3D-Conv1D-for-SLR-for-Microcontrollers: Initial release of S3D-Conv1D-ASL pipeline (v1.0)},
  version   = {v1.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18758927},
  url       = {https://doi.org/10.5281/zenodo.18758927}
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
