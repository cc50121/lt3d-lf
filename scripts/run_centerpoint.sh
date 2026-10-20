#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FRAMEWORK_ROOT=${FRAMEWORK_ROOT:-${ROOT}/third_party/centerpoint}
PYTHON=${PYTHON:-python}
GPUS=${GPUS:-1}
export PYTHONPATH="${ROOT}/third_party/nuscenes-sdk:${FRAMEWORK_ROOT}"
cd "${FRAMEWORK_ROOT}"
case "${1:-}" in
  train)
    "${PYTHON}" -m torch.distributed.launch --nproc_per_node="${GPUS}" \
      tools/train.py "${ROOT}/configs/detectors/centerpoint_lt3d.py" \
      --launcher pytorch --work-dir "${ROOT}/work_dirs/centerpoint"
    ;;
  infer)
    : "${CHECKPOINT:?Set CHECKPOINT to a CenterPoint checkpoint}"
    PREDICTIONS=${PREDICTIONS:-${ROOT}/results/centerpoint_raw.pkl}
    mkdir -p "$(dirname "${PREDICTIONS}")"
    "${PYTHON}" -m torch.distributed.launch --nproc_per_node="${GPUS}" \
      tools/test.py "${ROOT}/configs/detectors/centerpoint_lt3d.py" "${CHECKPOINT}" \
      --launcher pytorch --out "${PREDICTIONS}"
    ;;
  *) echo "Usage: $0 {train|infer}" >&2; exit 2 ;;
esac
