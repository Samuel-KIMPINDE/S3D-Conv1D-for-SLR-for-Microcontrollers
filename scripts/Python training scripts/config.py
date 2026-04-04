#!/usr/bin/env python
# coding: utf-8
"""
config.py — Central configuration for the S3DConv1D WLASL pipeline.

Edit the values in this file to match your environment before running
train_s3dconv1d.py or export_tflite.py.

Environments supported out of the box:
  - kaggle  (default)
  - local   (uncomment the LOCAL block below)
  - colab   (uncomment the COLAB block below)

You can also set the environment via shell:
    export PIPELINE_ENV=local
    python train_s3dconv1d.py
"""

import os

# ════════════════════════════════════════════════════════════════
# ENVIRONMENT SELECTOR
# Set to "kaggle", "local", or "colab"
# ════════════════════════════════════════════════════════════════
ENV = os.environ.get("PIPELINE_ENV", "kaggle")


# ════════════════════════════════════════════════════════════════
# PATH PROFILES
# ════════════════════════════════════════════════════════════════

_PROFILES = {

    "kaggle": dict(
        PRIMARY_VIDEO_DIR    = "/kaggle/input/wlasl-processed/videos",
        BACKUP_VIDEO_DIR     = "/kaggle/input/wlasl2000-resized/wlasl-complete/videos",
        JSON_PATH            = "/kaggle/input/wlasl2000-resized/wlasl-complete/nslt_100.json",
        CLASS_LIST_PATH      = "/kaggle/input/wlasl2000-resized/wlasl-complete/wlasl_class_list.txt",
        SAMUEL_VIDEO_DIR     = "/kaggle/input/asl-sam-kimpinde-set2/asl-samuel-kimpinde-dataset-set2/videos",
        SAMUEL_METADATA_PATH = "/kaggle/input/asl-sam-kimpinde-set2/asl-samuel-kimpinde-dataset-set2/metadata.csv",
        RAW_FRAME_DIR        = "/kaggle/working/frames_rgb",
        CHECKPOINT_PATH      = "/kaggle/working/best_S3D_conv1d.h5",
        OUTPUT_DIR           = "/kaggle/working",
    ),

    "local": dict(
        PRIMARY_VIDEO_DIR    = "data/videos/wlasl_processed",
        BACKUP_VIDEO_DIR     = "data/videos/wlasl_backup",
        JSON_PATH            = "data/nslt_100.json",
        CLASS_LIST_PATH      = "data/wlasl_class_list.txt",
        SAMUEL_VIDEO_DIR     = "data/videos/samuel",
        SAMUEL_METADATA_PATH = "data/samuel_metadata.csv",
        RAW_FRAME_DIR        = "data/frames_rgb",
        CHECKPOINT_PATH      = "checkpoints/best_S3D_conv1d.h5",
        OUTPUT_DIR           = "outputs",
    ),

    "colab": dict(
        PRIMARY_VIDEO_DIR    = "/content/drive/MyDrive/wlasl/videos/processed",
        BACKUP_VIDEO_DIR     = "/content/drive/MyDrive/wlasl/videos/backup",
        JSON_PATH            = "/content/drive/MyDrive/wlasl/nslt_100.json",
        CLASS_LIST_PATH      = "/content/drive/MyDrive/wlasl/wlasl_class_list.txt",
        SAMUEL_VIDEO_DIR     = "/content/drive/MyDrive/samuel/videos",
        SAMUEL_METADATA_PATH = "/content/drive/MyDrive/samuel/metadata.csv",
        RAW_FRAME_DIR        = "/content/frames_rgb",
        CHECKPOINT_PATH      = "/content/drive/MyDrive/checkpoints/best_S3D_conv1d.h5",
        OUTPUT_DIR           = "/content/drive/MyDrive/outputs",
    ),
}

# Validate ENV
if ENV not in _PROFILES:
    raise ValueError(f"Unknown ENV '{ENV}'. Choose from: {list(_PROFILES.keys())}")

# Unpack selected profile into module-level variables
_profile = _PROFILES[ENV]
PRIMARY_VIDEO_DIR    = _profile["PRIMARY_VIDEO_DIR"]
BACKUP_VIDEO_DIR     = _profile["BACKUP_VIDEO_DIR"]
JSON_PATH            = _profile["JSON_PATH"]
CLASS_LIST_PATH      = _profile["CLASS_LIST_PATH"]
SAMUEL_VIDEO_DIR     = _profile["SAMUEL_VIDEO_DIR"]
SAMUEL_METADATA_PATH = _profile["SAMUEL_METADATA_PATH"]
RAW_FRAME_DIR        = _profile["RAW_FRAME_DIR"]
CHECKPOINT_PATH      = _profile["CHECKPOINT_PATH"]
OUTPUT_DIR           = _profile["OUTPUT_DIR"]


# ════════════════════════════════════════════════════════════════
# MODEL & TRAINING HYPERPARAMETERS
# ════════════════════════════════════════════════════════════════
IMG_SIZE    = 64          # Height & width each frame is resized to
NUM_FRAMES  = 24          # Fixed temporal length per sample
NUM_CLASSES = 100         # Overridden at runtime after balancing
INPUT_SHAPE = (NUM_FRAMES, IMG_SIZE, IMG_SIZE, 1)

BATCH_SIZE  = 32
EPOCHS      = 100
RANDOM_SEED = 42


# ════════════════════════════════════════════════════════════════
# FPS SIMULATION TARGETS
# ════════════════════════════════════════════════════════════════
FPS_TARGETS = [10, 15, 30, 50]


# ════════════════════════════════════════════════════════════════
# DATASET BALANCING
# ════════════════════════════════════════════════════════════════
MIN_VIDEOS_PER_CLASS = 18
MAX_VIDEOS_PER_CLASS = 25


# ════════════════════════════════════════════════════════════════
# OUTPUT FILE NAMES  (resolved against OUTPUT_DIR / RAW_FRAME_DIR)
# ════════════════════════════════════════════════════════════════
LABEL_MAP_FILENAME        = "label_to_idx.json"
REPORT_FILENAME           = "s3dConv1d_classification_report.csv"
CONFUSION_MATRIX_FILENAME = "s3dConv1d_confusion_matrix.npy"
CONFUSION_MATRIX_PLOT     = "s3dConv1d_confusion_matrix.png"
TRAINING_PLOT             = "s3dConv1d_training_plot.png"
INT8_TFLITE_FILENAME      = "s3d_conv1d_int8_100.tflite"
FLOAT32_TFLITE_FILENAME   = "s3d_conv1d_float32_100.tflite"

# Resolved full paths
LABEL_MAP_PATH             = os.path.join(RAW_FRAME_DIR, LABEL_MAP_FILENAME)
REPORT_PATH                = os.path.join(OUTPUT_DIR,    REPORT_FILENAME)
CONFUSION_MATRIX_PATH      = os.path.join(OUTPUT_DIR,    CONFUSION_MATRIX_FILENAME)
CONFUSION_MATRIX_PLOT_PATH = os.path.join(OUTPUT_DIR,    CONFUSION_MATRIX_PLOT)
TRAINING_PLOT_PATH         = os.path.join(OUTPUT_DIR,    TRAINING_PLOT)
INT8_TFLITE_PATH           = os.path.join(RAW_FRAME_DIR, INT8_TFLITE_FILENAME)
FLOAT32_TFLITE_PATH        = os.path.join(RAW_FRAME_DIR, FLOAT32_TFLITE_FILENAME)


# ════════════════════════════════════════════════════════════════
# AUTO-CREATE DIRECTORIES
# ════════════════════════════════════════════════════════════════
for _dir in [RAW_FRAME_DIR, OUTPUT_DIR, os.path.dirname(CHECKPOINT_PATH)]:
    if _dir:
        os.makedirs(_dir, exist_ok=True)


# ════════════════════════════════════════════════════════════════
# QUICK SANITY PRINT  (runs when you do: python config.py)
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"{'='*52}")
    print(f"  Active environment : {ENV}")
    print(f"{'='*52}")
    fields = [
        ("Primary video dir",    PRIMARY_VIDEO_DIR),
        ("Backup video dir",     BACKUP_VIDEO_DIR),
        ("JSON path",            JSON_PATH),
        ("Class list",           CLASS_LIST_PATH),
        ("Samuel video dir",     SAMUEL_VIDEO_DIR),
        ("Samuel metadata",      SAMUEL_METADATA_PATH),
        ("Raw frame dir",        RAW_FRAME_DIR),
        ("Checkpoint path",      CHECKPOINT_PATH),
        ("Output dir",           OUTPUT_DIR),
        ("",                     ""),
        ("Image size",           IMG_SIZE),
        ("Num frames",           NUM_FRAMES),
        ("Num classes",          NUM_CLASSES),
        ("Batch size",           BATCH_SIZE),
        ("Epochs",               EPOCHS),
        ("FPS targets",          FPS_TARGETS),
        ("Min videos/class",     MIN_VIDEOS_PER_CLASS),
        ("Max videos/class",     MAX_VIDEOS_PER_CLASS),
    ]
    for label, value in fields:
        if label == "":
            print()
        else:
            print(f"  {label:<25} {value}")
    print(f"{'='*52}")
