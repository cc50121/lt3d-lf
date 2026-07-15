# LT3D-LF

LT3D-LF is a lightweight late-fusion toolkit for long-tail 3D object
detection on nuScenes-style data. It combines LiDAR 3D detections with
camera 2D detections, then applies class-aware matching, score calibration,
semantic relabeling, and long-tail rescue rules.

This release contains the final Co-DINO SWAT-Stage2 nuImages fusion recipe.

## Highlights

- Class-relaxed 2D/3D matching with IoU, IoA, and center-inside fallbacks.
- Per-class score calibration for 2D-supported 3D proposals.
- Long-tail rescue for low-score LiDAR proposals with camera evidence.
- Per-class LiDAR score thresholds for rare and frequent classes.
- LT3D evaluation helpers and qualitative comparison artifacts.

## Main Result

| Recipe | Camera detector | mAP | NDS |
|---|---|---:|---:|
| MMLF final | Co-DINO Swin-L SWAT-Stage2 nuImages | 57.04 | 62.81 |

Long-tail LCA0 AP:

| Split | Classes | Mean LCA0 AP |
|---|---|---:|
| Many | car, truck, adult, traffic_cone, barrier | 80.74 |
| Medium | trailer, bus, construction_vehicle, bicycle, motorcycle, construction_worker, pushable_pullable | 63.96 |
| Few | emergency_vehicle, child, police_officer, stroller, personal_mobility, debris | 29.21 |

More tables are in [`docs/results.md`](docs/results.md). Full prediction files
and 2D camera results are intentionally not stored in this code release. They
are available on Google Drive:

- Final 3D predictions: [part 1](https://drive.google.com/file/d/1ZHD2JIQzZpFG9cwTNrm8kUPSVUBW_z_z/view?usp=drivesdk), [part 2](https://drive.google.com/file/d/1_Wxu0Mq3L1Hjp3vd0_wedgJuVTlOrARa/view?usp=drivesdk), [part 3](https://drive.google.com/file/d/19Cv8Bd2XLfeXlWAbx8Pu25FMk3alM9z-/view?usp=drivesdk)
- Camera detection results: [lt3d_lf_camera_results.tar.gz](https://drive.google.com/file/d/1NGMGuB7detRJkiHy8hbXvok6rJ2PfLUU/view?usp=drivesdk)
- Evaluation summary: [lt3d_lf_evaluation_summary.tar.gz](https://drive.google.com/file/d/1Eq27aE-nJlVLYtpO5Mo0bJUip_7O6KRW/view?usp=drivesdk)
- Qualitative comparison video: [lt3d_lf_qualitative_comparison.mp4](https://drive.google.com/file/d/1AMkLc06Lc34tMeUulnpMAUZGz_stlVzD/view?usp=drivesdk)

To reconstruct the final 3D prediction file:

```bash
cat lt3d_lf_final_3d_predictions.json.gz.part* > lt3d_lf_final_3d_predictions.json.gz
```

On Windows:

```bat
copy /b lt3d_lf_final_3d_predictions.json.gz.part001+lt3d_lf_final_3d_predictions.json.gz.part002+lt3d_lf_final_3d_predictions.json.gz.part003 lt3d_lf_final_3d_predictions.json.gz
```

## Repository Layout

```text
lt3d_lf/                 Core fusion, calibration, evaluation, and metrics code
analysis/                Stratified LT3D LCA0 AP analysis
configs/                 Final fusion parameters and threshold maps
scripts/                 Clean final run script
docs/                    Result tables for the final model
```

## Environment

The code is intended to run inside an existing MMDetection3D + nuScenes-LT3D
environment. When multiple `nuscenes-devkit` versions are installed, set
`LT3D_NUSCENES_SDK` or pass `--nuscenes-sdk` so that the LT3D SDK is imported
first.

For long-tail 18-class evaluation, clone both LT3D repositories:

```bash
git clone https://github.com/neeharperi/LT3D
git clone https://github.com/neeharperi/nuscenes-lt3d.git
export LT3D_NUSCENES_SDK=/path/to/nuscenes-lt3d/python-sdk
```

The `nuscenes-lt3d` SDK provides the LT3D detection configs used by
`python main.py evaluate`, while the LT3D repository contains the benchmark
definition and related resources.

Install the lightweight Python dependencies when needed:

```bash
pip install -r requirements.txt
```

MMDetection3D, MMCV, CUDA, and the LT3D evaluation SDK should follow the
versions used by your detector environment.

## Run Final Fusion

Edit the paths in `scripts/run_final_fusion.sh`, then run:

```bash
bash scripts/run_final_fusion.sh
```

The script expands to:

```bash
python main.py fuse \
  --lidar-predictions /path/to/lidar_predictions.pkl \
  --camera-results /path/to/camera_results/s015 \
  --calibration-json /path/to/current_best.json \
  --score-2d-threshold 0.15 \
  --score-3d-threshold 0.10 \
  --score-3d-threshold-json configs/final_lidar_threshold_map.json \
  --match-mode class_relaxed \
  --mismatch-policy relabel \
  --rescue-score-json configs/final_rescue_score_map.json \
  --adult-child-relaxed
```

To evaluate an existing result JSON without running fusion again:

```bash
python main.py evaluate --result-json /path/to/results_nusc.json
```

## Final Fusion Recipe

The full recipe is recorded in
[`configs/final_fusion_codino_swat_stage2.yaml`](configs/final_fusion_codino_swat_stage2.yaml).
The most important choices are:

- `score_2d_thresh = 0.15`
- `default_score_3d_thresh = 0.10`
- `match_mode = class_relaxed`
- `mismatch_policy = relabel`
- `unmatched_lidar_factor = 0.20`
- `rescue_score_thresh = 0.05`
- `rescue_min_2d_score = 0.12`
- per-class LiDAR thresholds in
  [`configs/final_lidar_threshold_map.json`](configs/final_lidar_threshold_map.json)

## Qualitative Video

The qualitative comparison video is provided as a downloadable artifact rather
than as part of the source code. It shows continuous-frame comparisons between
MMLF, CMT, and CCF with highlighted long-tail improvements.

## Citation

If you use this code, please cite the corresponding paper and the upstream
detectors used in your experiments, including Co-DINO and the LiDAR detector.
