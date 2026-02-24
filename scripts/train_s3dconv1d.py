#!/usr/bin/env python
# coding: utf-8
"""
train_s3dconv1d.py — Full training pipeline for S3D-Conv1D on WLASL + Samuel dataset.

Steps:
  1. Build & validate combined metadata
  2. Extract / augment frames → .npy files
  3. Split and save dataset
  4. Visualize distributions & compute stats
  5. Build, compile, and train the S3DConv1D model
  6. Evaluate on the test set
  7. Generate classification report & confusion matrix

All paths and hyperparameters are read from config.py.
To switch environment:
    export PIPELINE_ENV=local   # or "colab"
    python train_s3dconv1d.py
"""

import os
import json
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import (
    Input, TimeDistributed, Conv2D, MaxPooling2D,
    Flatten, Conv1D, GlobalAveragePooling1D, Dense,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

# ── Config ──────────────────────────────────────────────────────
from config import (
    RAW_FRAME_DIR, OUTPUT_DIR, CHECKPOINT_PATH,
    BATCH_SIZE, EPOCHS, INPUT_SHAPE, RANDOM_SEED,
    LABEL_MAP_PATH, REPORT_PATH, CONFUSION_MATRIX_PATH,
)

# ── Utils ────────────────────────────────────────────────────────
from utils import (
    build_combined_dataframe,
    validate_videos,
    balance_dataset,
    extract_and_save_frames,
    load_dataset_from_disk,
    split_and_save,
    compute_dataset_stats,
    plot_split_distributions,
    plot_training_history,
    plot_confusion_matrix,
)


# ════════════════════════════════════════════════════════════════
# Step 1 — Build combined metadata
# ════════════════════════════════════════════════════════════════
print("=== Step 1: Building combined metadata ===")
df, label_to_idx, idx_to_label = build_combined_dataframe()


# ════════════════════════════════════════════════════════════════
# Step 2 — Validate & balance videos
# ════════════════════════════════════════════════════════════════
print("=== Step 2: Validating & balancing videos ===")
valid_df    = validate_videos(df)
filtered_df = balance_dataset(valid_df)


# ════════════════════════════════════════════════════════════════
# Step 3 — Frame extraction & augmentation
# ════════════════════════════════════════════════════════════════
print("=== Step 3: Extracting and augmenting frames ===")
extract_and_save_frames(filtered_df)


# ════════════════════════════════════════════════════════════════
# Step 4 — Load, split & save dataset
# ════════════════════════════════════════════════════════════════
print("=== Step 4: Loading dataset and splitting ===")
all_data, all_labels, label_to_idx = load_dataset_from_disk(label_to_idx=label_to_idx)
idx_to_label = {v: k for k, v in label_to_idx.items()}
NUM_CLASSES  = len(label_to_idx)

with open(LABEL_MAP_PATH, "w") as f:
    json.dump(label_to_idx, f)

X_train, X_val, X_test, y_train, y_val, y_test = split_and_save(all_data, all_labels)
print(f"Classes: {NUM_CLASSES} | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")


# ════════════════════════════════════════════════════════════════
# Step 5 — Visualize distributions & stats
# ════════════════════════════════════════════════════════════════
print("=== Step 5: Visualizing distributions & computing stats ===")
plot_split_distributions(y_train, y_val, y_test, idx_to_label)
compute_dataset_stats()


# ════════════════════════════════════════════════════════════════
# Step 6 — Build TF Datasets
# ════════════════════════════════════════════════════════════════
print("=== Step 6: Building TF Datasets ===")
AUTOTUNE = tf.data.AUTOTUNE

y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
y_val_cat   = to_categorical(y_val,   num_classes=NUM_CLASSES)
y_test_cat  = to_categorical(y_test,  num_classes=NUM_CLASSES)

train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
            .shuffle(len(X_train), seed=RANDOM_SEED)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val_cat))
          .cache().batch(BATCH_SIZE).prefetch(AUTOTUNE))

test_ds = (tf.data.Dataset.from_tensor_slices((X_test, y_test_cat))
           .cache().batch(BATCH_SIZE).prefetch(AUTOTUNE))


# ════════════════════════════════════════════════════════════════
# Step 7 — Model definition
# ════════════════════════════════════════════════════════════════
def build_s3d_conv1d(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    inputs = Input(shape=input_shape)
    x = TimeDistributed(Conv2D(12, (3, 3), activation="relu", padding="same"))(inputs)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(24, (3, 3), activation="relu", padding="same"))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Flatten())(x)
    x = Conv1D(48, 3, padding="same", activation="relu")(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="S3D_Conv1D")


print("=== Step 7: Building model ===")
model = build_s3d_conv1d(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
model.summary()


# ════════════════════════════════════════════════════════════════
# Step 8 — Compile & train
# ════════════════════════════════════════════════════════════════
print("=== Step 8: Compiling & training ===")
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=[
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_accuracy"),
    ],
)

callbacks = [
    ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True,
                    monitor="val_loss", mode="min", verbose=1),
    EarlyStopping(monitor="val_loss", patience=10,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

model.load_weights(CHECKPOINT_PATH)
plot_training_history(history)


# ════════════════════════════════════════════════════════════════
# Step 9 — Test evaluation
# ════════════════════════════════════════════════════════════════
print("=== Step 9: Evaluating on test set ===")
results = model.evaluate(test_ds, verbose=1)
print("\nTest Results:")
for name, val in zip(model.metrics_names, results):
    print(f"  {name}: {val:.4f}")


# ════════════════════════════════════════════════════════════════
# Step 10 — Classification report & confusion matrix
# ════════════════════════════════════════════════════════════════
print("=== Step 10: Classification report & confusion matrix ===")
y_probs = model.predict(test_ds, verbose=1)
y_pred  = np.argmax(y_probs, axis=1)
y_true  = y_test[:len(y_pred)]

target_names = [lbl for lbl, _ in sorted(label_to_idx.items(), key=lambda x: x[1])]

report_dict = classification_report(y_true, y_pred,
                                     target_names=target_names,
                                     output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(REPORT_PATH, index=True)
print(f"Classification report saved: {REPORT_PATH}")
print(classification_report(y_true, y_pred, target_names=target_names))

# Top / bottom 5 by F1
summary = report_df.iloc[:-3].sort_values("f1-score", ascending=False)
print("\n Top 5 by F1:")
print(summary.head(5)[["precision", "recall", "f1-score", "support"]].round(2).to_string())
print("\n Bottom 5 by F1:")
print(summary.tail(5)[["precision", "recall", "f1-score", "support"]].round(2).to_string())

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
np.save(CONFUSION_MATRIX_PATH, cm)
print(f"Confusion matrix saved: {CONFUSION_MATRIX_PATH}")
plot_confusion_matrix(CONFUSION_MATRIX_PATH, label_to_idx)
