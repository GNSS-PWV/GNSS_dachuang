> **2026-08-17 completion note:** The server rerun described below has now completed and the strict six-year result has been independently recomputed from raw yearly CSVs. The former authentication/VPN blocker is resolved. The current audit-ready status is `strict_run_live_status_20260817.md`; the authoritative research summary is `2014_2019严格回放式堆叠部署评估_成果汇总_20260817.md`.
# Strict Phase-1-to-Phase-2 Deployment Code Status (2026-08-16)

## Scope and current conclusion

This note records code-readiness work for the teacher's latest task: use the
Phase-1 GRU predictions to drive the Phase-2 ProfileTransformer and evaluate
2014--2019.  It does **not** report a new six-year result.  The historical
2017 output must remain labelled as a legacy implementation result until the
corrected pipeline is rerun.

The technically accurate current input statement is:

> Phase-2 is driven by the three surface-state predictions **PS / TS / WPS**.
> The generated Tm prediction is retained only as an auxiliary diagnostic; the
> current ProfileTransformer architecture does not consume it.

The experiment is a replay-style stacking evaluation: Phase-1 feature
construction uses historical observations, and Phase-2 still uses the true
GNSS ZWD as its wet-delay anchor.  It must not be described as a fully
autonomous real-time deployment without a separate recursive-input experiment.

## Corrected implementation

Files changed in this readiness pass:

- `predict_phase1_all_years.py`
  - strict default `--outlier-policy none`; the old global-IQR option is
    explicitly legacy-only because it selects samples using full-period target
    distributions;
  - prediction manifests record policy, years, station request and output
    summaries; empty/duplicate surface prediction outputs fail loudly.
- `phase2_p1_deploy.py`
  - `mean_profile` now uses a per-height, per-variable sum/count equal-weight
    mean, replacing recursive pairwise averaging that overweighted later
    neighbours;
  - a Phase-1 file containing `TIME` is matched only by exact `(station,TIME)`;
    DOY ordered fallback is limited to legacy files that have no `TIME` column;
  - PS/TS/WPS define the eligibility intersection; missing Tm cannot discard an
    otherwise usable sample;
  - writes `p1_match_audit_<year>.json` and
    `p1_match_audit_<year>_per_station.csv`, including matching outcomes and
    station/sample coverage;
  - direct yearly metrics use all six regimes' common valid samples.
- `analyze_p1deploy.py`
  - accepts one or more annual deployment CSVs;
  - writes common-sample micro metrics, annual metrics, per-station metrics,
    station-macro metrics, official-36 coverage, and optional audit summaries;
  - rejects duplicate `(station,time)` rows and zero-common-sample inputs
    instead of silently publishing invalid comparisons.

## Local validation completed

1. Python AST parsing passed for all three scripts.
2. A synthetic three-profile test confirmed the corrected mean is exactly the
   equal-weight average at each height.
3. Synthetic Phase-1 files confirmed that TIME-bearing files expose only exact
   time keys, whereas legacy no-TIME files retain DOY fallback data.
4. A two-year synthetic end-to-end analysis test produced micro, annual,
   station-macro, coverage and audit-summary CSVs.
5. The legacy 2017 prediction CSV was re-analysed without changing it.  This
   verifies the analysis script but does not validate the corrected deployment
   physics/cache result.

The local environment lacks `torch`, so full ProfileTransformer inference was
not run locally.  This is expected for this RTX 4060 workstation workflow; the
formal rerun belongs on the server environment.

## Server execution sequence after authenticated access is restored

Use the dated strict runners as the only formal entrypoint.  They refuse to
reuse historical output directories, so do not substitute the deprecated
manual `--overwrite` example from older notes.

```bash
cd /share/home/[REDACTED_USER]/tj23114/packages/dachuang_pwv
P1_JOB=$(sbatch phase2/run_p1strict_1419_20260817.sh | awk '{print $4}')
sbatch --dependency=afterok:${P1_JOB} phase2/run_p2strict_1419_20260817.sh
```

The Phase-2 runner validates the Phase-1 manifest (`outlier_policy=none`, the
2014--2019 request, four parameters and the requested 36 stations), and checks
that every nonempty strict prediction file has a `TIME` column with no duplicate
TIME keys before any ProfileTransformer evaluation.  Actual usable station
coverage can be below 36 because a station/year may not provide an eligible
24-step Phase-1 sequence; that coverage must be reported rather than silently
filled or claimed as full coverage.

Before any six-year claim, check every yearly prediction manifest,
`p1_match_audit` coverage, strict TIME-match counts (DOY fallback must be zero
for new files), and cache/model/GPT3 provenance.  Aggregate only the six new
deployment CSVs using `analyze_p1deploy.py`; the six-year RMSE must be
calculated from concatenated raw samples (micro average), not by averaging
annual RMSE values.

## External blocker

The previous server-status query failed once with an authentication error.  No
additional login retries were made, and no credential is stored in this note.
The blocker is server authentication/VPN/account state, not a missing small
model or a local GPU/model-loading error.  Once access is safely restored,
first inspect the existing Slurm job/logs and any already-created prediction
outputs before submitting a new strict rerun.
