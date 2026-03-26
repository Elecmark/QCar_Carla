This directory contains a standalone replacement pipeline for the original world-model code.

Files:
- `build_dataset.py`: builds trajectory-aware datasets from `QCarDataSet/*/*.csv`
- `train_model.py`: trains forward/backward models without cross-file windows
- `evaluate_model.py`: exports CSV metrics and plots
- `carla_driver.py`: CARLA runtime driver aligned with the new state definition

Key differences:
- Uses `manual_*_forward` and `manual_*_backward` folders only
- Preserves CSV boundaries and splits segments on time gaps
- Drops action outliers with `|steering| > 1.0` or `|throttle| > 0.3`
- Replaces quaternion regression with `yaw_sin` and `yaw_cos`
- Predicts next absolute state instead of raw quaternion deltas
