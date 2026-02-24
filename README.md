# Efficient Word-Level Sign Language Recognition Using Quantized Deep Learning for MCU Deployment

Real-time, privacy-preserving sign language recognition (SLR) running directly on microcontrollers — no cloud, no network, no compromise on accessibility.

This repository provides a complete, reproducible pipeline for training lightweight spatiotemporal sign language recognition models, quantizing them to INT8, and deploying them on resource-constrained edge devices. It bridges deep learning research with real-world embedded constraints through methodological insights for TinyML deployment in vision-based tasks.

---

## Why This Matters

SLR systems running on edge devices can deliver accessibility anywhere — affordable, portable, and private. On-device inference brings scalability, energy efficiency, and data sovereignty to wearable, mobile, and smart environments. This work advances both assistive technology and the broader TinyML field by demonstrating that quantized spatiotemporal models can run viably on microcontrollers.

---

## Project Structure

```
├── config.py               # Central configuration: all paths, hyperparameters, environments
├── utils.py                # Shared utilities: data loading, augmentation, visualization
├── train_s3dconv1d.py      # End-to-end training pipeline
├── export_tflite.py        # TFLite export (INT8 & Float32) + INT8 evaluation
└── README.md
```

---

## Pipeline Overview

```
Raw Videos (WLASL + ASL-Kimpinde)
        │
        ▼
  Validate & Balance          ← utils.py
        │
        ▼
  Frame Extraction            ← utils.py
  + FPS Simulation
  + Augmentation
        │
        ▼
  Train S3DConv1D             ← train_s3dconv1d.py
        │
        ▼
  Evaluate + Report           ← train_s3dconv1d.py
        │
        ▼
  Export to TFLite            ← export_tflite.py
  (INT8 for MCU / Float32)
        │
        ▼
  INT8 Inference Evaluation   ← export_tflite.py
```

---

## Datasets

| Dataset | Source | Role |
|---|---|---|
| **WLASL Processed** | [Kaggle](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed) | Primary video source |
| **WLASL2000 Resized** | [Kaggle](https://www.kaggle.com/datasets/sttaseen/wlasl2000-resized) | Backup videos + class list + JSON metadata |
| **ASL-Kimpinde Set** | [Zenodo](https://zenodo.org/records/18754451) | Custom domain-aware augmentation set |

### ASL-Kimpinde Set

A custom dataset collected by Samuel L. Kimpinde under controlled conditions to augment underrepresented WLASL classes and to enable domain-aware calibration for TinyML quantization.

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
  ├─ TimeDistributed Conv2D(12) + MaxPool   ← per-frame spatial features
  ├─ TimeDistributed Conv2D(24) + MaxPool
  ├─ TimeDistributed Flatten
  │
  ├─ Conv1D(48)                             ← temporal reasoning across frames
  ├─ GlobalAveragePooling1D
  │
  └─ Dense(num_classes, softmax)            ← classification head
```

The spatial backbone extracts per-frame features; Conv1D captures motion patterns across the temporal dimension. The result is a model small enough to quantize to INT8 and deploy on a microcontroller.

---

## Quantization

The trained model is exported in two formats via `export_tflite.py`:

| Format | Purpose | Quantization |
|---|---|---|
| `s3d_conv1d_int8_100.tflite` | MCU deployment | Full INT8 (weights + activations) |
| `s3d_conv1d_float32_100.tflite` | Accuracy baseline comparison | Float32 |

INT8 quantization uses **post-training quantization** with the test split as the representative calibration dataset. Both input and output tensors are quantized to `int8`.

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

### 4. Export to TFLite

```bash
python export_tflite.py
```

This will:
- Load the best Keras checkpoint
- Convert to INT8 TFLite (calibrated) and Float32 TFLite
- Run INT8 inference on the test split and print accuracy, softmax sanity stats, classification report, and confusion matrix

---

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

## Outputs

After a full run, the following files are produced:

```
frames_rgb/
├── <label>/                        # Per-class .npy frame arrays
├── X_train.npy / y_train.npy
├── X_val.npy   / y_val.npy
├── X_test.npy  / y_test.npy
├── label_to_idx.json
├── s3d_conv1d_int8_100.tflite
└── s3d_conv1d_float32_100.tflite

outputs/
├── best_S3D_conv1d.h5
├── s3dConv1d_classification_report.csv
├── s3dConv1d_confusion_matrix.npy
├── s3dConv1d_confusion_matrix.png
├── s3dConv1d_training_plot.png
├── train_distribution.png
├── val_distribution.png
├── test_distribution.png
└── combined_distribution.png
```

---

## Citation

If you use this work or the ASL-Kimpinde dataset, please cite:

```bibtex
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

Kimpinde, S. L. (2026). ASL-Kimpinde set: Minimal Dataset for sign Language Recognition (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.18754451
