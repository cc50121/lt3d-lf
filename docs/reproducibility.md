# Reproducing the IJCV release

[Paper](https://arxiv.org/pdf/2312.10986v6).
The paper is accepted by the International Journal of Computer Vision (IJCV).

## What is released

- Actual CenterPoint/LT3D, Co-DINO/MMDetection, and LT3D evaluator source snapshots.
- Portable detector training/inference configurations in `configs/detectors/`.
- Co-DINO Swin-L nuImages stage-2 epoch-1 weights, plus a historical CenterPoint epoch-19 checkpoint.
- The exact cached LiDAR input used by the final fusion experiment.
- Final Co-DINO score-adjustment coefficients and per-class LiDAR thresholds.
- Training, inference, camera-result conversion, fusion, and evaluation entry points.

The [GitHub Release](https://github.com/cc50121/lt3d-lf/releases/tag/ijcv-reproducibility-v1)
hosts source archives and large files. `configs/release_assets.json` records
sizes and SHA256 checksums. Co-DINO is split into four parts for reliable
transfer; the downloader verifies and reassembles the original checkpoint.

```bash
python scripts/download_release.py --extract
```

For sources alone, use `--sources-only --extract`. Extraction refuses to
overwrite a nonempty source directory. Original third-party license headers
are retained; see `third_party/README.md`.

## Environments

Keep the RGB detector environment separate from the legacy 3D/fusion environment.
The inspected experiment installation contains:

| Component | 2D detector | Legacy 3D/fusion |
|---|---|---|
| Python | 3.8.20 | 3.7.10 |
| PyTorch | 2.4.1, CUDA 12.1 | 1.7.0, CUDA 10.2 |
| Torchvision | 0.19.1 | 0.8.0 |
| MMDetection | 3.3.0 source snapshot | 2.25.3 |
| MMCV | 2.2.0 source-compatible snapshot | mmcv-full 1.6.0 (verified imported version) |
| MMEngine | 0.10.7 | Not used |
| NumPy | 1.24.4 | 1.19.5 |

The legacy installation also has stale `mmcv==1.4.8` distribution metadata;
the imported module was verified as 1.6.0. Install only the appropriate MMCV
distribution in a clean environment. The critical observed versions are recorded
in `environments/`; these constraints are not complete dependency lockfiles.
The 2D snapshot includes the experiment's MMCV-2.2 compatibility check.
Build MMCV and MMDetection3D CUDA extensions against the selected PyTorch/CUDA
toolchain, and install each detector snapshot editable in its own environment:

```bash
# In the 2D environment, after PyTorch/MMCV/MMEngine are installed:
pip install -r third_party/mmdetection/requirements/runtime.txt
pip install -e third_party/mmdetection --no-deps
pip install -r requirements-export.txt

# In the legacy 3D/fusion environment, after its framework dependencies:
pip install -r third_party/centerpoint/requirements/runtime.txt
pip install -e third_party/centerpoint --no-deps
pip install -r requirements.txt
```

The observed CUDA-10.2 PyTorch build supports only up to `sm_75`; it cannot
run detector inference on A6000 (`sm_86`) or A100 (`sm_80`). CPU-side offline
fusion used this legacy environment. For detector inference on newer GPUs,
use a compatible CUDA/PyTorch build and rebuild its extensions; do not assume
the old binaries are portable. A fresh environment build has not been
validated on every GPU/toolchain combination.

## Data and metadata

Download nuScenes v1.0-trainval and nuImages train from their official providers;
this release does not redistribute dataset images, point clouds, or annotations.
Set absolute paths, including the trailing slash for `LT3D_MERGED_ROOT`:

```bash
export ROOT=/absolute/path/to/lt3d-lf
export LT3D_DATA_ROOT=/absolute/path/to/nuscenes
export LT3D_NUIMAGES_ROOT=/absolute/path/to/nuimages
export LT3D_MERGED_ROOT=/absolute/path/to/nuscenes_nuimages/
export LT3D_NUSCENES_SDK="$ROOT/third_party/nuscenes-sdk"
export PYTHONPATH="$LT3D_NUSCENES_SDK:$ROOT/third_party/centerpoint:${PYTHONPATH:-}"
```

In the 3D environment, prepare 18-class infos, projected 2D annotations,
and the database sampler. Existing `test_2D_*.json` files are not overwritten:

```bash
python scripts/prepare_nuscenes.py --data-root "$LT3D_DATA_ROOT" \
  --framework-root "$ROOT/third_party/centerpoint" --create-infos --create-database
python scripts/convert_nuimages_to_coco.py \
  --nuimages-root "$LT3D_NUIMAGES_ROOT" --version v1.0-train \
  --out-nuimages-json "$LT3D_NUIMAGES_ROOT/nuimages_2D_train.json" \
  --nusc-coco "$LT3D_DATA_ROOT/test_2D_train.json" \
  --nusc-samples-root "$LT3D_DATA_ROOT/samples" \
  --merged-json "${LT3D_MERGED_ROOT}train_2D_nuscenes_nuimages.json"
```

The metadata helpers expose the original 18-class projection/conversion logic.
They have not been audited for byte-identical equivalence to the historical
training annotations; retain this distinction when comparing retrained models.
All 18 categories must be present. Camera exports map category **names**, not
an assumed category-ID offset, and require all six views of every val sample.

## RGB training and inference

In the 2D environment, remove the legacy 3D directory from `PYTHONPATH`.
Stage 1 uses the official Co-DINO Swin-L O365-to-COCO initialization from its
base config and trains on nuScenes+nuImages for six epochs. Stage 2 initializes
from stage-1 epoch 5 and fine-tunes on nuImages for two epochs, using resizing
without random cropping. The released final detector is stage-2 epoch 1.

```bash
export PYTHONPATH="$ROOT/third_party/mmdetection"
GPUS=8 bash scripts/run_codino.sh train-stage1
export LT3D_CODINO_STAGE1_CHECKPOINT="$ROOT/work_dirs/codino_stage1/best_coco_bbox_mAP_epoch_5.pth"
GPUS=8 bash scripts/run_codino.sh train-stage2
CHECKPOINT="$ROOT/downloads/codino_swinl_swat_nuimages_epoch1.pth" \
  GPUS=1 bash scripts/run_codino.sh infer
python scripts/export_camera_results.py \
  --coco-annotations "$LT3D_DATA_ROOT/test_2D_val.json" \
  --bbox-results "$ROOT/results/codino_val.bbox.json" \
  --infos "$LT3D_DATA_ROOT/nuscenes_infos_val.pkl" \
  --output-dir "$ROOT/results/camera/s015" --score-threshold 0.15
```

TXT rows are `class_id x1 y1 x2 y2 score`, with zero-based class IDs in the
fusion class order. Filenames are `<sample_token>@<CAM_VIEW>.txt`; empty files
represent views with no surviving detections. No OpenImages/CODA supplement
is part of the released final detector.

## LiDAR training and inference

In a GPU-compatible 3D environment, use:

```bash
GPUS=8 bash scripts/run_centerpoint.sh train
CHECKPOINT="$ROOT/downloads/centerpoint_epoch19.pth" \
  GPUS=1 bash scripts/run_centerpoint.sh infer
```

**Provenance boundary:** the epoch-19 checkpoint is an available historical
model from the experiment directory. Its exact linkage to the final cached
`prediction_filter_by_dis.pkl` has not been independently established. Raw
new inference is therefore not advertised as byte-identical to that cache.
Use the released `centerpoint_fusion_input.pkl` for the final fusion input.
The 22-output CenterPoint head includes 18 fine classes plus four hierarchy
outputs; these extra outputs are intentional, not a 10-class detector.

## Reproduce fusion and evaluate

The README links the released camera-results archive. Extract it and set
`CAMERA_RESULTS` to the actual directory containing the per-camera TXT files,
or generate the files using the detector steps above. Then, in the fusion environment:

```bash
export FRAMEWORK_ROOT="$ROOT/third_party/centerpoint"
export NUSCENES_SDK="$ROOT/third_party/nuscenes-sdk"
export DATA_ROOT="$LT3D_DATA_ROOT"
export LIDAR_PREDICTIONS="$ROOT/downloads/centerpoint_fusion_input.pkl"
export CAMERA_RESULTS=/absolute/path/to/extracted/camera/s015
bash scripts/run_final_fusion.sh
python main.py evaluate --framework-root "$FRAMEWORK_ROOT" \
  --nuscenes-sdk "$NUSCENES_SDK" --data-root "$DATA_ROOT" \
  --config "$ROOT/configs/detectors/centerpoint_lt3d.py" \
  --result-json "$ROOT/results/codino_swat_stage2_mmlf_final/pts_bbox/results_nusc.json" \
  --evaluation-config detection_lt3d
```

Historical final 18-class LCA0 results are **57.0362 mAP / 62.8100 NDS**,
with Many/Medium/Few AP **80.74 / 63.96 / 29.21** under the paper's fixed
frequency groups. `detection_lt3d_hierarchy` is a different diagnostic protocol;
do not compare its aggregate directly to LCA0 or to the standard 10-class benchmark.

The released per-class coefficients were selected for the detection-AP
objective in the experiment's validation search. They are score adjustments,
not independently verified well-calibrated posterior probabilities. Applying
them unchanged to another detector or domain is not validated.

## Verification and limits

```bash
python -m unittest discover -s tests -v
```

Tests cover category mapping, camera coverage, invalid scores, final
coefficients, and safe archive extraction. They do not substitute for full
dataset/GPU reruns. This release does not claim a newly measured full-validation
score or deterministic from-scratch retraining on a fresh machine.
