# Models

This folder contains the trained models produced by the S3D‑Conv1D pipeline:

- **s3d_conv1d_float32.tflite** — full‑precision TensorFlow Lite model for evaluation and comparison.
- **s3d_conv1d_int8.tflite** — quantized TensorFlow Lite model optimized for deployment on microcontrollers and embedded devices.
- **best_S3D_conv1d.h5** — best checkpoint of the trained Keras model, used as the source for exporting to TFLite.

These files complement the training scripts and can be used directly for inference or deployment.
