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
