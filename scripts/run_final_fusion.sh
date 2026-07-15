#!/usr/bin/env bash
set -euo pipefail

# Edit these paths for your machine.
ROOT=${ROOT:-/path/to/lt3d-lf}
FRAMEWORK_ROOT=${FRAMEWORK_ROOT:-/path/to/mmdet3d-lt3d}
NUSCENES_SDK=${NUSCENES_SDK:-/path/to/nuscenes-lt3d/python-sdk}
DATA_ROOT=${DATA_ROOT:-/path/to/nuscenes}
PYTHON=${PYTHON:-python}

LIDAR_PREDICTIONS=${LIDAR_PREDICTIONS:-/path/to/lidar_predictions.pkl}
CAMERA_RESULTS=${CAMERA_RESULTS:-/path/to/camera_results/s015}
CALIBRATION_JSON=${CALIBRATION_JSON:-/path/to/current_best.json}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/results/codino_swat_stage2_mmlf_final}

cd "${ROOT}"

"${PYTHON}" main.py fuse \
  --framework-root "${FRAMEWORK_ROOT}" \
  --nuscenes-sdk "${NUSCENES_SDK}" \
  --data-root "${DATA_ROOT}" \
  --lidar-predictions "${LIDAR_PREDICTIONS}" \
  --camera-results "${CAMERA_RESULTS}" \
  --output-dir "${OUTPUT_DIR}" \
  --calibration-json "${CALIBRATION_JSON}" \
  --evaluation-config detection_lt3d \
  --score-3d-threshold 0.10 \
  --score-3d-threshold-json configs/final_lidar_threshold_map.json \
  --score-2d-threshold 0.15 \
  --unmatched-lidar-factor 0.20 \
  --match-mode class_relaxed \
  --iou-threshold 0.50 \
  --relaxed-iou-threshold 0.30 \
  --ioa-det-threshold 0.70 \
  --ioa-proj-threshold 1.10 \
  --center-inside \
  --center-inside-score 0.51 \
  --mismatch-policy relabel \
  --rescue-classes bicycle,child,construction_vehicle,construction_worker,debris,emergency_vehicle,personal_mobility,police_officer,pushable_pullable,stroller,trailer \
  --rescue-score-threshold 0.05 \
  --rescue-score-json configs/final_rescue_score_map.json \
  --rescue-min-2d-score 0.12 \
  --rescue-min-match-score 0.0 \
  --rescue-unmatched-factor 0.0 \
  --adult-child-relaxed \
  --adult-child-min-2d-score 0.12
