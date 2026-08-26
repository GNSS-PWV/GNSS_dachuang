# -*- coding: utf-8 -*-
"""Plot aggregate comparison from summary.json."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

ROOT = r"D:\gnss水汽反演\通用框架_CCFA"
outdir = os.path.join(ROOT, "results", "seed42")
with open(os.path.join(outdir, "summary.json"), "r", encoding="utf-8") as f:
    S = json.load(f)

models = []
for k, v in S["models"].items():
    models.append({"model": k, **{kk: v["agg"][kk] for kk in ("rmse", "mae", "r2", "site_rmse_mean")}})
models.sort(key=lambda d: d["rmse"])
names = [m["model"] for m in models]
rmse = [m["rmse"] for m in models]
mae = [m["mae"] for m in models]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = ["#e15759" if n in ("persistence", "ridge") else ("#59a14f" if n.startswith("ours") else "#9aa0a6") for n in names]
axes[0].barh(names, rmse, color=colors)
axes[0].set_title("Unseen-station RMSE (lower better)"); axes[0].set_xlabel("RMSE (ug/m3)")
for i, v in enumerate(rmse):
    axes[0].text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8)
axes[1].barh(names, mae, color=colors)
axes[1].set_title("Unseen-station MAE (lower better)"); axes[1].set_xlabel("MAE (ug/m3)")
for i, v in enumerate(mae):
    axes[1].text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8)
plt.tight_layout()
png = os.path.join(outdir, "comparison.png")
plt.savefig(png, dpi=150)
print("saved", png)