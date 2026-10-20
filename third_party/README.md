# Detector source snapshots

Use `python scripts/download_release.py --sources-only --extract` to unpack
the source archives from the GitHub Release. They contain actual experiment
source, not links that silently follow an upstream default branch.

- `centerpoint/`: MMDetection3D LT3D fork, inspected base commit
  `23af86a734e5ff1c06cec244b9ca5434ca7e8b03`, including local modifications.
  The preserved `LICENSE` and OpenMMLab headers apply.
- `mmdetection/`: actual MMDetection 3.3.0 source and CO-DETR project from
  the shared training directory. That directory did not have a trustworthy
  independent Git revision; the archive SHA256 identifies the snapshot.
  Hidden duplicate packaging trees, compiled binaries, and checkpoints are excluded.
  The preserved `LICENSE` and project headers apply.
- `nuscenes-sdk/`: the actual LT3D evaluator package used by offline fusion.
  It is based on nuScenes-devkit 1.1.9 with LT3D evaluation changes.
  See `NUSCENES_LICENSE.txt` and the source copyright headers (Apache 2.0).

Only source directories needed for these detector workflows are included.
Data, credentials, build artifacts, and unrelated experiment outputs are not
part of the source snapshots. The datasets and pretrained initialization
weights retain their own provider terms. Do not treat this notice as changing
third-party licenses or the terms of the datasets.
