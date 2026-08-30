"""Train one phase-one target with the strict past-only feature contract.

This is an intermediate retraining entry point.  It is suitable for checking
the corrected phase-one feature pipeline on the server; it does not claim to
solve the independent phase-two GNSS input problem.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from causal_features import make_causal_sequences


TARGETS = ("PS", "WPS", "TS", "Tm")
TARGET_UNITS = {"PS": "hPa", "WPS": "hPa", "TS": "K", "Tm": "K"}


class CausalGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return self.head(output[:, -1, :])


def read_file(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data[(data["YEAR"] >= 2014) & (data["YEAR"] <= 2018)].copy()
    data["station_id"] = path.name.split("_met")[0]
    return data


def load_sequences(data_dir: Path, target: str, max_files: int | None, time_steps: int):
    files = sorted(data_dir.rglob("*_met.txt"))
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"no *_met.txt files under {data_dir}")
    all_x, all_y, all_meta, feature_names = [], [], [], None
    for index, path in enumerate(files, start=1):
        frame = read_file(path)
        try:
            x, y, meta, feature_names = make_causal_sequences(frame, target=target, time_steps=time_steps)
        except ValueError:
            continue
        all_x.append(x)
        all_y.append(y)
        all_meta.append(meta)
        if index % 10 == 0:
            print(f"loaded {index}/{len(files)} files, sequences={sum(len(v) for v in all_x)}", flush=True)
    if not all_x:
        raise ValueError("no usable causal sequences")
    return np.concatenate(all_x), np.concatenate(all_y), pd.concat(all_meta, ignore_index=True), feature_names


def split_indices(meta: pd.DataFrame, seed: int = 42) -> dict[str, np.ndarray]:
    stations = np.array(sorted(meta["station_id"].dropna().astype(str).unique()))
    rng = np.random.RandomState(seed)
    held_count = max(1, int(len(stations) * 0.10))
    held = set(rng.permutation(stations)[:held_count])
    station_mask = meta["station_id"].astype(str).isin(held).to_numpy()
    years = pd.to_numeric(meta["YEAR"], errors="coerce").to_numpy()
    remaining = ~station_mask
    return {
        "train": np.flatnonzero(remaining & np.isin(years, [2014, 2015, 2016])),
        "test_2017": np.flatnonzero(remaining & (years == 2017)),
        "val_2018": np.flatnonzero(remaining & (years == 2018)),
        "val_leave_station": np.flatnonzero(station_mask),
        "held_out_stations": np.array(sorted(held)),
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    return {
        "N": int(len(y_true)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "Bias": float(np.mean(y_pred - y_true)),
    }


def predict_in_batches(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    """Run evaluation without placing an entire split on GPU at once."""
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start:start + batch_size]).to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--time-steps", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; target={args.target}", flush=True)

    x, y, meta, features = load_sequences(args.data_dir, args.target, args.max_files, args.time_steps)
    split = split_indices(meta, args.seed)
    train_idx = split["train"]
    if not len(train_idx):
        raise ValueError("strict train split is empty; use more files or check years")
    x_scaler = StandardScaler().fit(x[train_idx].reshape(-1, x.shape[-1]))
    y_scaler = StandardScaler().fit(y[train_idx])
    x_scaled = x_scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)
    y_scaled = y_scaler.transform(y).astype(np.float32)

    model = CausalGRU(x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    loss_fn = nn.HuberLoss()
    train_ds = TensorDataset(torch.from_numpy(x_scaled[train_idx]), torch.from_numpy(y_scaled[train_idx]))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        history.append(float(np.mean(losses)))
        print(f"epoch={epoch}/{args.epochs} train_loss={history[-1]:.6f}", flush=True)

    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "target": args.target,
        "target_unit": TARGET_UNITS[args.target],
        "time_steps": args.time_steps, "features": features}, args.output_dir / "model.pth")
    with (args.output_dir / "scalers.pkl").open("wb") as handle:
        pickle.dump({"x_scaler": x_scaler, "y_scaler": y_scaler}, handle)
    result_metrics = {}
    for name in ("test_2017", "val_2018", "val_leave_station"):
        idx = split[name]
        if not len(idx):
            result_metrics[name] = {"status": "empty"}
            continue
        pred_scaled = predict_in_batches(model, x_scaled[idx], device)
        pred = y_scaler.inverse_transform(pred_scaled).ravel()
        result_metrics[name] = metrics(y[idx].ravel(), pred)
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "schema_version": "causal_phase1_metrics/v2",
        "protocol": "causal_past_only_leave_station_10pct_seed42_year_split",
        "target": args.target,
        "target_unit": TARGET_UNITS[args.target],
        "time_steps": args.time_steps,
        "feature_count": len(features),
        "feature_names": features,
        "dataset_counts": {key: int(len(value)) for key, value in split.items() if key != "held_out_stations"},
        "held_out_stations": split["held_out_stations"].tolist(),
        "year_counts": {
            name: {
                str(year): int(np.sum(pd.to_numeric(meta.iloc[idx]["YEAR"], errors="coerce") == year))
                for year in sorted(pd.to_numeric(meta.iloc[idx]["YEAR"], errors="coerce").dropna().unique())
            }
            for name, idx in split.items()
            if name != "held_out_stations"
        },
        "scalers_fit_on": "train",
        "epochs": args.epochs,
        "metrics": result_metrics,
        "history": history,
        "warning": "Intermediate phase-one retraining; not an independent phase-two GNSS result.",
    }, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result_metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
