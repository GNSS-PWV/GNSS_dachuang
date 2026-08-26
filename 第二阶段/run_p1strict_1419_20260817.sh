#!/bin/bash
#SBATCH --job-name=p1strict1419
#SBATCH --partition=L40
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/share/home/u23114/tj23114/packages/dachuang_pwv/phase2/logs/p1strict1419_%j.out
#SBATCH --error=/share/home/u23114/tj23114/packages/dachuang_pwv/phase2/logs/p1strict1419_%j.err

# Strict replay-stacking Phase-1 inference for the 2014--2019 deployment study.
# This intentionally uses a new output root and never overwrites legacy outputs.
set -euo pipefail
source activate kokolo
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export KMP_DUPLICATE_LIB_OK=TRUE

BASE=/share/home/u23114/tj23114/packages/dachuang_pwv
OUT_ROOT="$BASE/phase2/p1_full_predictions_strict_20260817"
cd "$BASE"

if [[ -e "$OUT_ROOT" ]]; then
  echo "Refusing to reuse existing strict output root: $OUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUT_ROOT"

sha256sum phase2/predict_phase1_all_years.py phase2/phase2_p1_deploy.py phase2/analyze_p1deploy.py
python -u phase2/predict_phase1_all_years.py \
  --years 2014 2015 2016 2017 2018 2019 \
  --data-min-year 2014 --data-max-year 2019 \
  --params PS TS WPS Tm \
  --outlier-policy none \
  --device cuda \
  --batch-size 2048 \
  --out-root "$OUT_ROOT"

echo "P1_STRICT_2014_2019_DONE"
