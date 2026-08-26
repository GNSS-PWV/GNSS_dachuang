# Strict Phase-1 to Phase-2 Deployment Run Status (2026-08-17)

## Status: completed and locally verified

Teacher task: drive the Phase-2 ProfileTransformer with the all-year Phase-1 GRU surface predictions and evaluate 2014-2019 under the strict replay protocol.

Formal server execution completed successfully:

- Phase-1 all-year strict inference finished with `P1_STRICT_2014_2019_DONE`.
- Phase-2 strict deployment evaluation finished with `P2_STRICT_2014_2019_DONE`.
- The verified local result mirror is `result_p1deploy_ft_strict_20260817/`.
- Server/local file sets agree and SHA256 verification reported zero mismatches.

## Protocol boundaries

- Phase-2 is driven by the predicted surface states PS / TS / WPS.
- Tm is an auxiliary diagnostic only. It is neither a ProfileTransformer input nor an eligibility requirement.
- Phase-1 uses historical-observation lag features and Phase-2 retains true GNSS ZWD as its wet-delay anchor. This is a replay-style stacking deployment evaluation, not fully autonomous real-time operation.
- The seasonal climate cache is a spatial leave-station-out, full-history cache. It must not be described as a strict future-year extrapolation test.

## Verified six-year result

Aggregation uses the six raw yearly deployment prediction CSVs on their common samples (micro average), not an average of annual RMSE values.

| Item | Verified value |
|---|---:|
| Years | 2014-2019 |
| Common samples | 110,928 |
| Official stations in six-year union | 36 |
| Annual usable stations | 27, 28, 29, 30, 30, 30 |
| Best deployable regime | clim_surf_p1 |
| clim_surf_p1 RMSE | 0.242729 mm |
| GPT3 RMSE | 0.346177 mm |
| Relative RMSE reduction vs GPT3 | 29.88% |

The oracle regimes `real` and `real_surf_p1` are sensitivity upper bounds and must not be presented as deployment accuracy claims.

## Alignment and coverage audit

- All new Phase-1 prediction files have TIME columns and no duplicate TIME keys.
- PS / TS / WPS / Tm matching uses exact `(station, TIME)` keys for these new files.
- DOY fallback count is zero for every parameter in every year.
- Every yearly deployment CSV has zero duplicate `(station, time)` keys.
- There are exact-match misses in 2014-2018; those timestamps were excluded rather than aligned by fallback. For PS, unmatched counts are 972, 144, 146, 5, 36, and 0 for 2014 through 2019. The final records remain common across all six compared regimes.
- Annual coverage below 36 stations is due to the Phase-1 historical sequence eligibility; it is not missing-value imputation or a Phase-2 station exclusion rule.

## Primary artifacts

- `result_p1deploy_ft_strict_20260817/analysis_2014_2019/metrics_all_stations_2014_2019.csv`
- `result_p1deploy_ft_strict_20260817/analysis_2014_2019/annual_metrics_all_stations_2014_2019.csv`
- `result_p1deploy_ft_strict_20260817/analysis_2014_2019/station_macro_all_stations_2014_2019.csv`
- `result_p1deploy_ft_strict_20260817/analysis_2014_2019/coverage_summary_2014_2019.csv`
- `result_p1deploy_ft_strict_20260817/analysis_2014_2019/p1_match_audit_summary_2014_2019.csv`
- `2014_2019严格回放式堆叠部署评估_成果汇总_20260817.md`

Historical directories `result_p1deploy/`, `result_p1deploy_ft_eval/`, and `result_p1deploy_reanalysis_legacy/` are legacy only and must not be mixed into the strict six-year result.