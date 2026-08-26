#!/bin/bash
#SBATCH --job-name=p2strict1419
#SBATCH --partition=L40
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/share/home/u23114/tj23114/packages/dachuang_pwv/phase2/logs/p2strict1419_%j.out
#SBATCH --error=/share/home/u23114/tj23114/packages/dachuang_pwv/phase2/logs/p2strict1419_%j.err

# Runs only after strict Phase-1 prediction has completed successfully.
# The cache remains an all-history climatological cache; this is a replay
# deployment evaluation, not a future-only temporal extrapolation claim.
set -euo pipefail
source activate kokolo
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export KMP_DUPLICATE_LIB_OK=TRUE

BASE=/share/home/u23114/tj23114/packages/dachuang_pwv
P1_ROOT="$BASE/phase2/p1_full_predictions_strict_20260817"
RESULT_ROOT="$BASE/phase2/result_p1deploy_ft_strict_20260817"
cd "$BASE"

if [[ ! -f "$P1_ROOT/manifest.json" ]]; then
  echo "Missing completed Phase-1 manifest: $P1_ROOT/manifest.json" >&2
  exit 2
fi
# Validate the strict Phase-1 contract before launching any Phase-2 work.
P1_ROOT="$P1_ROOT" python - <<'PY'
import csv, json, os, pathlib, sys
root = pathlib.Path(os.environ["P1_ROOT"])
manifest_path = root / "manifest.json"
with manifest_path.open(encoding="utf-8") as handle:
    manifest = json.load(handle)
expected_years = [2014, 2015, 2016, 2017, 2018, 2019]
expected_params = ["PS", "TS", "WPS", "Tm"]
if manifest.get("outlier_policy") != "none":
    raise SystemExit("Strict P1 validation failed: outlier_policy is not none")
if manifest.get("prediction_years") != expected_years:
    raise SystemExit(f"Strict P1 validation failed: prediction_years={manifest.get('prediction_years')}")
if manifest.get("station_count_requested") != 36:
    raise SystemExit(f"Strict P1 validation failed: station_count_requested={manifest.get('station_count_requested')}")
params = manifest.get("params", {})
missing = [name for name in expected_params if name not in params]
if missing:
    raise SystemExit(f"Strict P1 validation failed: missing params {missing}")
for name in expected_params:
    for year in expected_years:
        year_dir = root / name / f"year_{year}"
        files = sorted(year_dir.glob("*.txt"))
        if not files or len(files) > 36:
            raise SystemExit(f"Strict P1 validation failed: {name} {year} has {len(files)} station files; expected 1..36")
        for path in files:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter=" ", skipinitialspace=True)
                if not reader.fieldnames or "TIME" not in reader.fieldnames:
                    raise SystemExit(f"Strict P1 validation failed: missing TIME column in {path}")
                times = [row["TIME"] for row in reader if row.get("TIME")]
            if not times:
                raise SystemExit(f"Strict P1 validation failed: empty prediction file {path}")
            if len(times) != len(set(times)):
                raise SystemExit(f"Strict P1 validation failed: duplicate TIME keys in {path}")
print("STRICT_P1_VALIDATION_OK")
PY
if [[ -e "$RESULT_ROOT" ]]; then
  echo "Refusing to reuse existing strict Phase-2 output root: $RESULT_ROOT" >&2
  exit 2
fi
mkdir -p "$RESULT_ROOT"
sha256sum phase2/predict_phase1_all_years.py phase2/phase2_p1_deploy.py phase2/analyze_p1deploy.py

for YEAR in 2014 2015 2016 2017 2018 2019; do
  python -u phase2/phase2_p1_deploy.py \
    --model_dir phase2/result_p1deploy_ft \
    --year "$YEAR" \
    --cache phase2/result_grid/st_seasonal_cache.pkl \
    --test_stations phase2/test_stations_official_36.txt \
    --k 5 \
    --p1_root "$P1_ROOT" \
    --out "$RESULT_ROOT/year_$YEAR"
done

python -u phase2/analyze_p1deploy.py \
  --csv "$RESULT_ROOT"/year_*/deploy_p1_predictions_*.csv \
  --test_stations phase2/test_stations_official_36.txt \
  --out "$RESULT_ROOT/analysis_2014_2019" \
  --year 2014_2019 \
  --audit "$RESULT_ROOT"/year_*/p1_match_audit_*.json

echo "P2_STRICT_2014_2019_DONE"
