#!/usr/bin/env python
# coding: utf-8
"""
export_tflite.py — Export trained S3DConv1D to TFLite (INT8 & Float32)
and evaluate the INT8 model on the test split.

All paths are read from config.py.
To switch environment:
    export PIPELINE_ENV=local   # or "colab"
    python export_tflite.py
"""

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# ── Config ──────────────────────────────────────────────────────
from config import (
    RAW_FRAME_DIR,
    CHECKPOINT_PATH,
    LABEL_MAP_PATH,
    INT8_TFLITE_PATH,
    FLOAT32_TFLITE_PATH,
)


# ════════════════════════════════════════════════════════════════
# Load calibration data & label map
# ════════════════════════════════════════════════════════════════
print("Loading calibration data …")
X_calib = np.load(os.path.join(RAW_FRAME_DIR, "X_test.npy")).astype(np.float32)
y_calib = np.load(os.path.join(RAW_FRAME_DIR, "y_test.npy"))

with open(LABEL_MAP_PATH, "r") as f:
    label_to_idx = json.load(f)

target_names = [lbl for lbl, _ in sorted(label_to_idx.items(), key=lambda x: x[1])]


# ════════════════════════════════════════════════════════════════
# Representative dataset generator  (used for INT8 calibration)
# ════════════════════════════════════════════════════════════════
def representative_data_gen():
    for i in range(len(X_calib)):
        yield [X_calib[i:i + 1]]   # shape: (1, 24, 64, 64, 1)


# ════════════════════════════════════════════════════════════════
# Load Keras model
# ════════════════════════════════════════════════════════════════
print(f"Loading Keras model from: {CHECKPOINT_PATH}")
model = tf.keras.models.load_model(CHECKPOINT_PATH)


# ════════════════════════════════════════════════════════════════
# Export: INT8 (for OpenMV AE3)
# ════════════════════════════════════════════════════════════════
print("Converting to INT8 TFLite …")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

quant_model = converter.convert()
with open(INT8_TFLITE_PATH, "wb") as f:
    f.write(quant_model)
print(f"INT8 model saved: {INT8_TFLITE_PATH} "
      f"({os.path.getsize(INT8_TFLITE_PATH) / 1024:.1f} KB)")


# ════════════════════════════════════════════════════════════════
# Export: Float32
# ════════════════════════════════════════════════════════════════
print("Converting to Float32 TFLite …")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.inference_input_type  = tf.float32
converter.inference_output_type = tf.float32

float_model = converter.convert()
with open(FLOAT32_TFLITE_PATH, "wb") as f:
    f.write(float_model)
print(f"Float32 model saved: {FLOAT32_TFLITE_PATH} "
      f"({os.path.getsize(FLOAT32_TFLITE_PATH) / 1024:.1f} KB)")


# ════════════════════════════════════════════════════════════════
# Evaluate INT8 model
# ════════════════════════════════════════════════════════════════
print("\nEvaluating INT8 model …")

interpreter = tf.lite.Interpreter(model_path=INT8_TFLITE_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def preprocess_input(x):
    """Quantize float32 frame to int8 using the model's scale/zero-point."""
    scale, zero_point = input_details[0]["quantization"]
    return (x / scale + zero_point).astype(np.int8)


def dequantize_output(x):
    """Dequantize int8 logits back to float32."""
    scale, zero_point = output_details[0]["quantization"]
    return (x.astype(np.float32) - zero_point) * scale


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


y_probs_int8 = []
for x in X_calib:
    x_input = preprocess_input(x[np.newaxis, ...])          # (1, 24, 64, 64, 1)
    interpreter.set_tensor(input_details[0]["index"], x_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    y_probs_int8.append(softmax(dequantize_output(output)[0]))

y_probs_int8 = np.array(y_probs_int8)

# ── Softmax sanity check ──────────────────────────────────────
sums = np.sum(y_probs_int8, axis=1)
print("\nSoftmax sum stats:")
print(f"  Mean:    {np.mean(sums):.6f}")
print(f"  Std:     {np.std(sums):.6f}")
print(f"  Min:     {np.min(sums):.6f}")
print(f"  Max:     {np.max(sums):.6f}")

# ── Accuracy ──────────────────────────────────────────────────
y_pred = np.argmax(y_probs_int8, axis=1)
accuracy = np.mean(y_pred == y_calib)
print(f"\nINT8 model accuracy on test set: {accuracy * 100:.2f}%")

# ── Classification report ─────────────────────────────────────
print("\nClassification Report:")
print(classification_report(y_calib, y_pred,
                             target_names=target_names,
                             zero_division=0))

# ── Confusion matrix ──────────────────────────────────────────
print("Confusion Matrix:")
print(confusion_matrix(y_calib, y_pred))
