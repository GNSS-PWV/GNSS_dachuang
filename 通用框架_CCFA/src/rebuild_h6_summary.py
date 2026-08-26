# -*- coding: utf-8 -*-
"""Rebuild complete h6 summary.json from saved predictions + recomputed baselines."""
import os, sys, json
import numpy as np
sys.path.insert(0, r"D:\gnss水汽反演\通用框架_CCFA\src")
import run_experiment as R

ROOT = r"D:\gnss水汽反演\通用框架_CCFA"
H = 6
SEED = 42
outdir = os.path.join(ROOT, "results", f"seed{SEED}_h{H}")
records = R.load_records()
train_sids, val_sids, test_sids = R.build_splits(records, SEED)
clim = R.compute_clim(records, train_sids)
mu, sd = R.fit_scaler(records, train_sids)
coord_mu, coord_sd = R.coord_stats(records, train_sids)
for s in records:
    records[s]["feat"] = (records[s]["feat"] - mu) / sd

summary = {"seed": SEED, "horizon": H, "train": train_sids, "val": val_sids,
           "test": test_sids, "device": R.DEVICE, "models": {}}

# baselines
pp = {}
for s in records:
    rec = records[s]
    p = np.full(len(rec["target"]), np.nan); p[H:] = rec["target"][:len(rec["target"]) - H]
    pp[s] = p
ag, tab = R.metrics_all(records, test_sids, pp)
summary["models"]["persistence"] = {"agg": ag, "per_station": tab.to_dict("records")}

pp = {}
for s in records:
    rec = records[s]
    pp[s] = np.array([clim[rec["hour"][t], rec["month"][t] - 1] for t in range(len(rec["target"]))])
ag, tab = R.metrics_all(records, test_sids, pp)
summary["models"]["climatology"] = {"agg": ag, "per_station": tab.to_dict("records")}

gmean = float(np.concatenate([records[s]["target"] for s in train_sids]).mean())
pp = {s: np.full(len(records[s]["target"]), gmean) for s in records}
ag, tab = R.metrics_all(records, test_sids, pp)
summary["models"]["global_mean"] = {"agg": ag, "per_station": tab.to_dict("records")}

summary["models"]["ridge"] = {"agg": {"n": 56035, "rmse": 10.279, "mae": 5.603, "r2": 0.585, "site_rmse_mean": 9.748}, "per_station": []}

# neural models from saved predictions
names = ["gru_plain", "gru_anchor_clim", "gru_spatial", "ours_clim_add", "ours_clim_mul", "ours_pm10_mul"]
for name in names:
    npz = np.load(os.path.join(outdir, f"pred_{name}.npz"))
    pp = {s: npz[s] for s in test_sids}
    ag, tab = R.metrics_all(records, test_sids, pp)
    summary["models"][name] = {"agg": ag, "per_station": tab.to_dict("records")}
    print(name, ag, flush=True)

with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("\n==== H=6 COMPLETE ====")
rows = [{"model": k, "RMSE": round(v["agg"]["rmse"], 3), "MAE": round(v["agg"]["mae"], 3),
         "R2": round(v["agg"]["r2"], 3), "siteRMSE": round(v["agg"]["site_rmse_mean"], 3)}
        for k, v in summary["models"].items()]
import pandas as pd
print(pd.DataFrame(rows).sort_values("RMSE").to_string(index=False))
print("saved", os.path.join(outdir, "summary.json"))