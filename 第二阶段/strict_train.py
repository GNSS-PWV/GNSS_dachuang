# -*- coding: utf-8 -*-
"""Train ProfileTransformer with the teacher's strict station/year protocol.

This entry point deliberately does not reuse ``data.prepare_data`` because that
function implements the historical station-only split.  The strict protocol
has five disjoint outputs:

* train: remaining stations, 2014--2016
* test_2017: remaining stations, 2017
* val_2018: remaining stations, 2018
* val_leave_station: held-out stations, all 2014--2018
* val_2019: all 2019 profiles, independent final validation

The scaler is fitted only on ``train``.  The script is intentionally kept as a
separate entry point so existing historical results and commands remain
reproducible and cannot silently switch to the new protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (  # noqa: E402
    GLOBAL_FEATURE_DIM,
    ProfileDataset,
    collate_profiles,
)
from model import ProfileTransformer  # noqa: E402
from strict_dataset_split import load_profiles_from_dirs, split_profiles  # noqa: E402
from train import compute_metrics, evaluate_model, plot_scatter, set_seed  # noqa: E402


SPLIT_ORDER = ("train", "test_2017", "val_2018", "val_leave_station", "val_2019")


def _make_loader(profiles, scalers, *, fit_scalers=False, batch_size=64,
                 max_len=30, num_workers=0, shuffle=False):
    dataset = ProfileDataset(profiles, fit_scalers=fit_scalers, **({} if fit_scalers else scalers))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_profiles(batch, max_len=max_len),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def build_strict_loaders(data_dirs, args):
    profiles = load_profiles_from_dirs(data_dirs, max_files=args.max_files)
    if not profiles:
        raise ValueError("未加载到完整 profile；需要包含 *_met.txt 的垂直廓线数据")
    splits, manifest = split_profiles(profiles, seed=args.seed, holdout_ratio=args.holdout_ratio)
    required = ["train", "test_2017", "val_2018", "val_leave_station", "val_2019"]
    if args.require_all_splits:
        missing = [name for name in required if not splits[name]]
        if missing:
            raise ValueError(f"严格协议所需集合为空: {missing}; 当前数据可能只是探针或缺少年份")

    train_ds, train_loader = _make_loader(
        splits["train"], None, fit_scalers=True, batch_size=args.batch_size,
        max_len=args.max_len, num_workers=args.num_workers, shuffle=True,
    )
    scalers = train_ds.get_scalers()
    eval_loaders = {}
    eval_datasets = {}
    for name in SPLIT_ORDER[1:]:
        eval_datasets[name], eval_loaders[name] = _make_loader(
            splits[name], scalers, batch_size=args.batch_size, max_len=args.max_len,
            num_workers=args.num_workers,
        )
    info = {
        "protocol": "leave_station_10pct_seed42_then_year_split",
        "manifest": manifest,
        "dataset_counts": {name: len(ds) for name, ds in {"train": train_ds, **eval_datasets}.items()},
        "dataset_station_counts": {
            name: len({p["station_id"] for p in splits[name]}) for name in SPLIT_ORDER
        },
        "scaler_fit_split": "train",
    }
    return train_loader, eval_loaders, scalers, info


def _evaluate_nonempty(model, loader, device):
    if loader is None or len(loader.dataset) == 0:
        return None
    result = evaluate_model(model, loader, device)
    return result


def train_strict(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}", flush=True)
    train_loader, eval_loaders, scalers, info = build_strict_loaders(args.data_dir, args)
    print(json.dumps(info["dataset_counts"], ensure_ascii=False), flush=True)
    print(f"Scaler 拟合集合: {info['scaler_fit_split']}", flush=True)

    model = ProfileTransformer(
        level_feat_dim=4,
        global_feat_dim=GLOBAL_FEATURE_DIM,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.SmoothL1Loss(beta=1.0)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(Path(args.output_dir) / "scalers.pkl", "wb") as fh:
        pickle.dump(scalers, fh)
    (Path(args.output_dir) / "strict_split_manifest.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    best_rmse = float("inf")
    wait = 0
    history = {"train_loss": [], "val_2018_rmse": []}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in train_loader:
            levels = batch["levels"].to(device)
            heights = batch["heights"].to(device)
            global_feat = batch["global_feat"].to(device)
            mask = batch["attention_mask"].to(device)
            zwd = batch["zwd"].to(device)
            target = batch["pwv"].to(device)
            pred = model(levels, heights, global_feat, mask) * zwd
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        scheduler.step()
        avg_loss = total_loss / max(1, batches)
        history["train_loss"].append(avg_loss)
        val_result = _evaluate_nonempty(model, eval_loaders.get("val_2018"), device)
        val_rmse = float(val_result[0]["RMSE"]) if val_result else avg_loss
        history["val_2018_rmse"].append(val_rmse)
        print(f"Epoch {epoch:3d}/{args.epochs} loss={avg_loss:.5f} val2018_RMSE={val_rmse:.5f}", flush=True)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            wait = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "args": vars(args)}, Path(args.output_dir) / "best_model.pth")
        else:
            wait += 1
        if wait >= args.patience:
            print(f"早停: 2018 验证集连续 {args.patience} 轮未改善", flush=True)
            break

    ckpt = Path(args.output_dir) / "best_model.pth"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)["model_state_dict"])

    metrics = {}
    for name, loader in eval_loaders.items():
        result = _evaluate_nonempty(model, loader, device)
        if result is None:
            metrics[name] = {"status": "empty"}
            continue
        pwv_m, pi_m, pwv_pred, pwv_true, pi_pred, pi_true, zwd, station_ids, times, lats, lons, elvs, tms = result
        metrics[name] = {"pwv": pwv_m, "pi": pi_m}
        pd.DataFrame({
            "station_id": station_ids, "time": times, "lat": lats, "lon": lons,
            "elv": elvs, "tm": tms, "zwd": zwd, "pwv_true": pwv_true,
            "pwv_pred": pwv_pred, "pwv_error": pwv_pred - pwv_true,
            "pi_true": pi_true, "pi_pred": pi_pred,
        }).to_csv(Path(args.output_dir) / f"{name}_predictions.csv", index=False, float_format="%.6f")
        plot_scatter(pwv_true, pwv_pred, str(Path(args.output_dir) / f"{name}_scatter.png"), f"Strict {name}: PWV")
    (Path(args.output_dir) / "strict_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (Path(args.output_dir) / "strict_training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", action="append", required=True, help="一个或多个包含 *_met.txt 的目录")
    parser.add_argument("--output_dir", default="result_strict_train")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout_ratio", type=float, default=0.10)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--require_all_splits", action="store_true", help="正式训练时要求五个集合均非空")
    args = parser.parse_args()
    args.output_dir = str(Path(args.output_dir).resolve())
    train_strict(args)


if __name__ == "__main__":
    main()
