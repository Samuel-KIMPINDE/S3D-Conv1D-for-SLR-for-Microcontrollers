#!/usr/bin/env python
# coding: utf-8
"""
utils.py — Shared utilities for the S3DConv1D WLASL pipeline.

Covers: label/metadata loading, video validation, frame extraction,
augmentation, dataset splitting, stats, and all visualizations.
All paths and hyperparameters are pulled from config.py.
"""

import os
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
from vidaug import augmentors as va

from config import (
    PRIMARY_VIDEO_DIR, BACKUP_VIDEO_DIR,
    JSON_PATH, CLASS_LIST_PATH,
    SAMUEL_VIDEO_DIR, SAMUEL_METADATA_PATH,
    RAW_FRAME_DIR, OUTPUT_DIR,
    IMG_SIZE, NUM_FRAMES, FPS_TARGETS,
    MIN_VIDEOS_PER_CLASS, MAX_VIDEOS_PER_CLASS,
    RANDOM_SEED,
    LABEL_MAP_PATH,
    TRAINING_PLOT_PATH,
    CONFUSION_MATRIX_PLOT_PATH,
)

# ─────────────────────────────────────────────
# Augmentation Pipeline
# ─────────────────────────────────────────────
augment_seq = va.Sequential([
    va.RandomRotate(degrees=10),
    va.GaussianBlur(1.0),
])


# ─────────────────────────────────────────────
# Label Loading
# ─────────────────────────────────────────────
def load_class_list(path=CLASS_LIST_PATH):
    idx_to_label, label_to_idx = {}, {}
    with open(path, "r") as f:
        for line in f:
            idx, label = line.strip().split("\t")
            idx = int(idx)
            idx_to_label[idx] = label
            label_to_idx[label] = idx
    return idx_to_label, label_to_idx


# ─────────────────────────────────────────────
# Metadata Loading
# ─────────────────────────────────────────────
def load_wlasl_metadata(json_path, idx_to_label,
                         primary_video_dir=PRIMARY_VIDEO_DIR,
                         backup_video_dir=BACKUP_VIDEO_DIR):
    with open(json_path, "r") as f:
        data = json.load(f)
    metadata = {}
    for video_id, info in data.items():
        label_idx = info["action"][0]
        label = idx_to_label.get(label_idx, f"unknown_{label_idx}")
        metadata[video_id] = {
            "label":        label,
            "primary_path": os.path.join(primary_video_dir, f"{video_id}.mp4"),
            "backup_path":  os.path.join(backup_video_dir,  f"{video_id}.mp4"),
        }
    return (pd.DataFrame.from_dict(metadata, orient="index")
              .reset_index()
              .rename(columns={"index": "video_id"}))


def load_samuel_metadata(metadata_path, label_to_idx, idx_to_label,
                          samuel_video_dir=SAMUEL_VIDEO_DIR):
    df = pd.read_csv(metadata_path)
    records, next_idx = [], max(label_to_idx.values()) + 1
    for _, row in df.iterrows():
        label = row["label"]
        if label not in label_to_idx:
            label_to_idx[label] = next_idx
            idx_to_label[next_idx] = label
            next_idx += 1
        records.append({
            "video_id":     row["filename"].replace(".avi", ""),
            "label":        label,
            "primary_path": os.path.join(samuel_video_dir, row["filename"]),
            "backup_path":  os.path.join(samuel_video_dir, row["filename"]),
        })
    return pd.DataFrame(records)


def build_combined_dataframe():
    idx_to_label, label_to_idx = load_class_list()
    df_wlasl  = load_wlasl_metadata(JSON_PATH, idx_to_label)
    df_samuel = load_samuel_metadata(SAMUEL_METADATA_PATH, label_to_idx, idx_to_label)
    df = pd.concat([df_wlasl, df_samuel], ignore_index=True)
    return df, label_to_idx, idx_to_label


# ─────────────────────────────────────────────
# Video Validation & Balancing
# ─────────────────────────────────────────────
def validate_videos(df):
    valid = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating videos"):
        cap = cv2.VideoCapture(row["primary_path"])
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(row["backup_path"])
        path = row["primary_path"] if cap.isOpened() else row["backup_path"]
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            ret, _ = cap.read()
            if ret and fps:
                valid.append({**row, "video_path": path, "fps": fps})
        cap.release()
    return pd.DataFrame(valid)


def balance_dataset(valid_df,
                    min_videos=MIN_VIDEOS_PER_CLASS,
                    max_videos=MAX_VIDEOS_PER_CLASS,
                    seed=RANDOM_SEED):
    balanced = []
    for label, group in valid_df.groupby("label"):
        if len(group) >= min_videos:
            sampled = group.sample(n=min(len(group), max_videos), random_state=seed)
            balanced.append(sampled)
    filtered_df = pd.concat(balanced).reset_index(drop=True)
    print(f"Balanced: {len(filtered_df)} videos across {filtered_df['label'].nunique()} classes.")
    return filtered_df


# ─────────────────────────────────────────────
# Frame Processing
# ─────────────────────────────────────────────
def simulate_fps(frames, fps_orig, fps_target):
    step = max(1, int(round(fps_orig / fps_target)))
    return frames[::step]


def hybrid_subsample_pad(frames, target_len=NUM_FRAMES):
    num = len(frames)
    if num >= target_len:
        indices = np.linspace(0, num - 1, target_len).astype(int)
        return np.array([frames[i] for i in indices], dtype=np.float32)
    pad = [frames[-1]] * (target_len - num)
    return np.array(frames + pad, dtype=np.float32)


# ─────────────────────────────────────────────
# Frame Extraction & Augmentation
# ─────────────────────────────────────────────
def extract_and_save_frames(filtered_df, output_dir=RAW_FRAME_DIR):
    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Extracting frames"):
        label_dir = os.path.join(output_dir, row["label"])
        os.makedirs(label_dir, exist_ok=True)
        video_id = row["video_id"]
        fps_orig  = row["fps"]

        cap = cv2.VideoCapture(row["video_path"])
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.expand_dims(frame, axis=-1).astype(np.float32) / 255.0
            frames.append(frame)
        cap.release()

        if not frames:
            continue

        frames_aug = augment_seq(frames)

        variants = {
            f"{video_id}_orig_{int(fps_orig)}fps.npy": hybrid_subsample_pad(frames),
            f"{video_id}_aug_{int(fps_orig)}fps.npy":  hybrid_subsample_pad(frames_aug),
        }
        for fps_t in FPS_TARGETS:
            variants[f"{video_id}_orig_{fps_t}fps.npy"] = hybrid_subsample_pad(
                simulate_fps(frames, fps_orig, fps_t))
            variants[f"{video_id}_aug_{fps_t}fps.npy"] = hybrid_subsample_pad(
                simulate_fps(frames_aug, fps_orig, fps_t))

        for name, data in variants.items():
            np.save(os.path.join(label_dir, name), data.astype(np.float32))


# ─────────────────────────────────────────────
# Dataset Loading & Splitting
# ─────────────────────────────────────────────
def load_dataset_from_disk(frame_dir=RAW_FRAME_DIR, label_to_idx=None):
    available_labels = set(os.listdir(frame_dir))
    if label_to_idx is None:
        with open(LABEL_MAP_PATH, "r") as f:
            label_to_idx = json.load(f)
    label_to_idx = {k: v for k, v in label_to_idx.items() if k in available_labels}

    all_data, all_labels = [], []
    for label, idx in label_to_idx.items():
        label_path = os.path.join(frame_dir, label)
        for fname in os.listdir(label_path):
            if fname.endswith(".npy"):
                arr = np.load(os.path.join(label_path, fname))
                all_data.append(arr)
                all_labels.append(idx)

    return np.array(all_data, dtype=np.float32), np.array(all_labels), label_to_idx


def split_and_save(all_data, all_labels, frame_dir=RAW_FRAME_DIR, seed=RANDOM_SEED):
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_data, all_labels, test_size=0.3, stratify=all_labels, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=seed)

    for name, arr in [("X_train", X_train), ("y_train", y_train),
                       ("X_val",   X_val),   ("y_val",   y_val),
                       ("X_test",  X_test),  ("y_test",  y_test)]:
        np.save(os.path.join(frame_dir, f"{name}.npy"), arr)

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_splits(frame_dir=RAW_FRAME_DIR):
    splits = {}
    for name in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
        splits[name] = np.load(os.path.join(frame_dir, f"{name}.npy"))
    with open(LABEL_MAP_PATH, "r") as f:
        splits["label_to_idx"] = json.load(f)
    splits["idx_to_label"] = {v: k for k, v in splits["label_to_idx"].items()}
    return splits


# ─────────────────────────────────────────────
# Dataset Statistics
# ─────────────────────────────────────────────
def compute_dataset_stats(frame_dir=RAW_FRAME_DIR):
    class_counts, frame_lengths = defaultdict(int), []
    for label in os.listdir(frame_dir):
        label_path = os.path.join(frame_dir, label)
        if not os.path.isdir(label_path):
            continue
        for fname in os.listdir(label_path):
            if fname.endswith(".npy"):
                arr = np.load(os.path.join(label_path, fname))
                class_counts[label] += 1
                frame_lengths.append(len(arr))

    df_stats = pd.DataFrame({
        "Label":        list(class_counts.keys()),
        "Valid Videos": list(class_counts.values()),
    }).sort_values("Valid Videos", ascending=False)

    global_stats = {
        "Total Classes":              len(class_counts),
        "Total Valid Videos":         sum(class_counts.values()),
        "Mean Frames per Video":      np.mean(frame_lengths),
        "Median Frames per Video":    np.median(frame_lengths),
        "Min Frames":                 np.min(frame_lengths),
        "Max Frames":                 np.max(frame_lengths),
        "Fixed Keyframes per Sample": NUM_FRAMES,
    }

    print("=== Per-Class Statistics ===")
    print(df_stats.to_string(index=False))
    print("\n=== Global Statistics ===")
    for k, v in global_stats.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    return df_stats, global_stats


# ─────────────────────────────────────────────
# Visualization Helpers
# ─────────────────────────────────────────────
def count_labels(y, idx_to_label):
    counts = np.bincount(y, minlength=len(idx_to_label))
    return pd.DataFrame({
        "Label": [idx_to_label[i] for i in range(len(counts))],
        "Count": counts,
    }).sort_values("Label")


def _save_barplot(df, title, filename, color, out_dir=OUTPUT_DIR):
    plt.figure(figsize=(18, 24))
    sns.barplot(data=df, x="Label", y="Count", palette=color)
    plt.title(title)
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()


def plot_split_distributions(y_train, y_val, y_test, idx_to_label):
    df_train = count_labels(y_train, idx_to_label)
    df_val   = count_labels(y_val,   idx_to_label)
    df_test  = count_labels(y_test,  idx_to_label)

    _save_barplot(df_train, "Train Split: Videos per Label",      "train_distribution.png", "Blues_d")
    _save_barplot(df_val,   "Validation Split: Videos per Label", "val_distribution.png",   "Greens_d")
    _save_barplot(df_test,  "Test Split: Videos per Label",       "test_distribution.png",  "Oranges_d")

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    for ax, df, title, palette in zip(
        axes,
        [df_train, df_val, df_test],
        ["Train Split", "Validation Split", "Test Split"],
        ["Blues_d", "Greens_d", "Oranges_d"],
    ):
        sns.barplot(data=df, x="Label", y="Count", ax=ax, palette=palette)
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
    axes[-1].set_xlabel("Label")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "combined_distribution.png"))
    plt.show()


def plot_training_history(history, save_path=TRAINING_PLOT_PATH):
    if history is None or not hasattr(history, "history"):
        print("Invalid training history object.")
        return
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("accuracy", []),     label="Train Acc")
    plt.plot(history.history.get("val_accuracy", []), label="Val Acc")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history.get("loss", []),     label="Train Loss")
    plt.plot(history.history.get("val_loss", []), label="Val Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_confusion_matrix(cm_path, label_to_idx, save_path=CONFUSION_MATRIX_PLOT_PATH):
    labels = [lbl for lbl, _ in sorted(label_to_idx.items(), key=lambda x: x[1])]
    cm = np.load(cm_path)
    cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(12, 10))
    sns.heatmap(pd.DataFrame(cm_norm, index=labels, columns=labels),
                annot=True, fmt=".2f", cmap="Blues", cbar=True,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="gray")
    plt.title("Confusion Matrix — S3DConv1D (Normalized)", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
