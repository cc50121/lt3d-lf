#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MMDET_ROOT=${MMDET_ROOT:-${ROOT}/third_party/mmdetection}
PYTHON=${PYTHON:-python}
GPUS=${GPUS:-8}
export PYTHONPATH="${MMDET_ROOT}:${PYTHONPATH:-}"
cd "${MMDET_ROOT}"
case "${1:-}" in
  train-stage1|train-stage2)
    stage=${1#train-}
    "${PYTHON}" -m torch.distributed.run --nproc_per_node="${GPUS}" \
      tools/train.py "${ROOT}/configs/detectors/codino_${stage}.py" \
      --launcher pytorch --work-dir "${ROOT}/work_dirs/codino_${stage}"
    ;;
  infer)
    : "${CHECKPOINT:?Set CHECKPOINT to the released Co-DINO checkpoint}"
    OUTPUT_PREFIX=${OUTPUT_PREFIX:-${ROOT}/results/codino_val}
    mkdir -p "$(dirname "${OUTPUT_PREFIX}")"
    "${PYTHON}" -m torch.distributed.run --nproc_per_node="${GPUS}" \
      tools/test.py "${ROOT}/configs/detectors/codino_stage2.py" "${CHECKPOINT}" \
      --launcher pytorch --cfg-options \
      test_evaluator.outfile_prefix="${OUTPUT_PREFIX}"
    ;;
  *) echo "Usage: $0 {train-stage1|train-stage2|infer}" >&2; exit 2 ;;
esac
