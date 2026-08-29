#!/bin/bash
# Intermediate causal phase-one retraining. Submit smoke before full runs.
#SBATCH --job-name=p1causal
#SBATCH --partition=L40
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/share/home/u23114/GNSS_dachuang/training_logs/p1causal_%j.out
#SBATCH --error=/share/home/u23114/GNSS_dachuang/training_logs/p1causal_%j.err
set -euo pipefail
MODE="${1:-smoke}"
TARGET="${2:-PS}"
PYTHON="/share/home/u23114/.conda/envs/kokolo/bin/python"
CODE_DIR="/share/home/u23114/GNSS_dachuang/GNSS_dachuang_code/第一阶段"
DATA_DIR="/share/home/u23114/GNSS_dachuang/GNSS_dachuang_data/第二阶段/profile_reconstructed_igra_20260825"
OUT_ROOT="/share/home/u23114/GNSS_dachuang/causal_phase1_results"
cd "${CODE_DIR}"
case "${TARGET}" in
  PS|WPS|TS|Tm) ;;
  *) echo "target must be PS, WPS, TS, or Tm" >&2; exit 2 ;;
esac
COMMON=(--data-dir "${DATA_DIR}" --target "${TARGET}" --output-dir "${OUT_ROOT}/${TARGET}_${SLURM_JOB_ID}" --batch-size 256 --seed 42)
if [[ "${MODE}" == "smoke" ]]; then
  "${PYTHON}" train_causal_phase1.py "${COMMON[@]}" --max-files 10 --epochs 1
elif [[ "${MODE}" == "full" ]]; then
  "${PYTHON}" train_causal_phase1.py "${COMMON[@]}" --epochs 30
else
  echo "use smoke or full" >&2
  exit 2
fi
