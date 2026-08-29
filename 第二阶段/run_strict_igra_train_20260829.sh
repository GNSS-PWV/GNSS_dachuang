#!/bin/bash
# Strict phase-2 ProfileTransformer training on reconstructed IGRA profiles.
# Submit a smoke run first, then a full run after the smoke artifacts pass.
#
# Usage:
#   sbatch run_strict_igra_train_20260829.sh smoke
#   sbatch run_strict_igra_train_20260829.sh full
#
# The data directory is intentionally outside the Git working tree.
# The split is implemented in strict_train.py: train=2014-2016, test=2017,
# validation=2018 plus 10% held-out stations, and final validation=2019.
#SBATCH --job-name=p2strictigra
#SBATCH --partition=L40
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=/share/home/u23114/GNSS_dachuang/training_logs/p2strictigra_%j.out
#SBATCH --error=/share/home/u23114/GNSS_dachuang/training_logs/p2strictigra_%j.err

set -euo pipefail

MODE="${1:-full}"
CODE_DIR="/share/home/u23114/GNSS_dachuang/GNSS_dachuang_code/第二阶段"
DATA_DIR="/share/home/u23114/GNSS_dachuang/GNSS_dachuang_data/第二阶段/profile_reconstructed_igra_20260825"
RESULT_ROOT="/share/home/u23114/GNSS_dachuang/strict_training_results_20260829"
PYTHON="/share/home/u23114/.conda/envs/kokolo/bin/python"

mkdir -p /share/home/u23114/GNSS_dachuang/training_logs "${RESULT_ROOT}"
cd "${CODE_DIR}"

echo "host=$(hostname)"
echo "mode=${MODE}"
echo "code_commit=$(git -C /share/home/u23114/GNSS_dachuang/GNSS_dachuang_code rev-parse HEAD)"
echo "started=$(date -Is)"
"${PYTHON}" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable on the allocated node")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

COMMON=(
  --data_dir "${DATA_DIR}"
  --batch_size 128
  --max_len 30
  --lr 1e-4
  --weight_decay 1e-4
  --d_model 128
  --n_heads 8
  --n_layers 4
  --ff_dim 512
  --dropout 0.1
  --seed 42
  --holdout_ratio 0.10
  --num_workers 5
  --require_all_splits
)

case "${MODE}" in
  smoke)
    "${PYTHON}" strict_train.py "${COMMON[@]}" \
      --epochs 1 --patience 1 --max_files 10 \
      --output_dir "${RESULT_ROOT}/smoke_${SLURM_JOB_ID}"
    ;;
  full)
    "${PYTHON}" strict_train.py "${COMMON[@]}" \
      --epochs 100 --patience 20 \
      --output_dir "${RESULT_ROOT}/full_${SLURM_JOB_ID}"
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use smoke or full." >&2
    exit 2
    ;;
esac

echo "finished=$(date -Is)"
