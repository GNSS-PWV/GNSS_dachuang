# -*- coding: utf-8 -*-
"""Generic spatio-temporal framework feasibility (horizon-aware).

Protocol: SITE-LEVEL GENERALIZATION. 14 train / 3 val / 4 test stations.
Forecast horizon configurable (1 / 6 / 24 h ahead).

Usage: python run_experiment.py --seed 42 --epochs 100 --horizon 24 [--models all|subset]
"""
import os, sys, json, time, argparse, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = r"D:\gnss水汽反演\通用框架_CCFA"
sys.path.insert(0, os.path.join(ROOT, "src"))
from prep_data import FEAT_COLS
from model import GeoIndexGRU

L = 24
BATCH = 512
LR = 1e-3
PATIENCE = 12
TRAIN_STRIDE = 6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PM10_IDX = FEAT_COLS.index("PM10")


def load_records():
    return pd.read_pickle(os.path.join(ROOT, "data", "prepared.pkl"))


def build_splits(records, seed):
    sids = sorted(records.keys())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(sids))
    sids = [sids[i] for i in perm]
    return sids[:14], sids[14:17], sids[17:21]


def compute_clim(records, sids):
    h = np.concatenate([records[s]["hour"] for s in sids])
    m = np.concatenate([records[s]["month"] for s in sids])
    y = np.concatenate([records[s]["target"] for s in sids])
    ok = np.concatenate([records[s]["target_ok"] for s in sids])
    y, h, m = y[ok], h[ok], m[ok]
    clim = np.full((24, 12), np.nan)
    for hh in range(24):
        for mm in range(1, 13):
            sel = (h == hh) & (m == mm)
            if sel.sum() > 0:
                clim[hh, mm - 1] = y[sel].mean()
    return np.where(np.isnan(clim), y.mean(), clim)


def fit_scaler(records, sids):
    X = np.concatenate([records[s]["feat"] for s in sids], axis=0)
    mu = X.mean(0); sd = X.std(0) + 1e-6
    return mu, sd


def coord_stats(records, sids):
    C = np.stack([records[s]["coord"] for s in sids])
    C[:, 2] = np.log1p(C[:, 2])
    mu = C.mean(0); sd = C.std(0) + 1e-6
    return mu, sd


def norm_coord(coord, mu, sd):
    c = coord.copy(); c[2] = np.log1p(c[2])
    return (c - mu) / sd


class Windows(Dataset):
    def __init__(self, rec, coord_mu, coord_sd, clim, anchor_mode, L=L, stride=1, horizon=1):
        self.rec = rec
        self.coord = norm_coord(rec["coord"], coord_mu, coord_sd).astype(np.float32)
        self.clim = clim
        self.anchor_mode = anchor_mode
        self.L = L
        self.horizon = horizon
        self.idx = list(range(L, len(rec["feat"]) - (horizon - 1), stride))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        t = self.idx[i]
        tt = t + self.horizon - 1
        x = self.rec["feat"][t - self.L:t]
        if self.anchor_mode.startswith("clim"):
            a = float(self.clim[self.rec["hour"][tt], self.rec["month"][tt] - 1])
        elif self.anchor_mode.startswith("pm10"):
            a = float(self.rec["feat"][tt, PM10_IDX])
        else:
            a = 0.0
        return x.astype(np.float32), self.coord, np.float32(a), \
               np.float32(self.rec["target"][tt]), bool(self.rec["target_ok"][tt])


def collate(batch):
    xs = torch.from_numpy(np.stack([b[0] for b in batch]))
    cs = torch.from_numpy(np.stack([b[1] for b in batch]))
    a = torch.from_numpy(np.stack([b[2] for b in batch]))
    y = torch.from_numpy(np.stack([b[3] for b in batch]))
    ok = torch.from_numpy(np.stack([np.float32(b[4]) for b in batch]))
    return xs, cs, a, y, ok


def predict_station(model, rec, coord_mu, coord_sd, clim, anchor_mode, device, horizon):
    ds = Windows(rec, coord_mu, coord_sd, clim, anchor_mode, stride=1, horizon=horizon)
    dl = DataLoader(ds, batch_size=2048, shuffle=False, collate_fn=collate, num_workers=0)
    preds = np.full(len(rec["target"]), np.nan, dtype=np.float64)
    model.eval()
    offset = 0
    with torch.no_grad():
        for x, c, a, y, ok in dl:
            p = model(x.to(device), c.to(device), a.to(device)).cpu().numpy()
            for j in range(p.shape[0]):
                preds[ds.idx[offset + j]] = p[j]
            offset += p.shape[0]
    return preds


def metrics_all(records, sids, preds_map):
    rows = []
    agg_y, agg_p = [], []
    for s in sids:
        rec = records[s]; p = preds_map[s]
        ok = rec["target_ok"]
        y = rec["target"][ok]; pp = p[ok]
        m = ~np.isnan(pp)
        if m.sum() == 0:
            continue
        rmse = float(np.sqrt(np.mean((y[m] - pp[m]) ** 2)))
        mae = float(np.mean(np.abs(y[m] - pp[m])))
        ss = np.sum((y[m] - y[m].mean()) ** 2) + 1e-12
        r2 = float(1 - np.sum((y[m] - pp[m]) ** 2) / ss)
        rows.append({"station": s, "n": int(m.sum()), "rmse": rmse, "mae": mae, "r2": r2})
        agg_y.append(y[m]); agg_p.append(pp[m])
    Y = np.concatenate(agg_y); P = np.concatenate(agg_p)
    agg = {
        "n": int(len(Y)),
        "rmse": float(np.sqrt(np.mean((Y - P) ** 2))),
        "mae": float(np.mean(np.abs(Y - P))),
        "r2": float(1 - np.sum((Y - P) ** 2) / (np.sum((Y - Y.mean()) ** 2) + 1e-12)),
        "site_rmse_mean": float(np.mean([r["rmse"] for r in rows])),
    }
    return agg, pd.DataFrame(rows)


def train_model(records, train_sids, val_sids, coord_mu, coord_sd, clim,
                anchor_mode, use_spatial, in_dim, device, epochs, seed, horizon):
    torch.manual_seed(seed); np.random.seed(seed)
    ds_train = torch.utils.data.ConcatDataset(
        [Windows(records[s], coord_mu, coord_sd, clim, anchor_mode, stride=TRAIN_STRIDE, horizon=horizon)
         for s in train_sids])
    dl_train = DataLoader(ds_train, batch_size=BATCH, shuffle=True, collate_fn=collate,
                          num_workers=0, drop_last=True)
    model = GeoIndexGRU(in_dim=in_dim, hidden=64, layers=2, anchor=anchor_mode,
                        use_spatial=use_spatial).to(device)
    torch.cuda.empty_cache()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
    lossf = nn.MSELoss()
    best_val = float("inf"); best_state = None; bad = 0
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        for x, c, a, y, ok in dl_train:
            x, c, a, y = x.to(device), c.to(device), a.to(device), y.to(device)
            opt.zero_grad()
            p = model(x, c, a)
            loss = lossf(p, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1
        vpred = {s: predict_station(model, records[s], coord_mu, coord_sd, clim,
                                    anchor_mode, device, horizon) for s in val_sids}
        vag, _ = metrics_all(records, val_sids, vpred)
        val_rmse = vag["rmse"]
        sched.step(val_rmse)
        if val_rmse < best_val - 1e-4:
            best_val = val_rmse; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 10 == 0 or bad == 1:
            print(f"  ep {ep:3d} loss {tot/nb:.4f} val_rmse {val_rmse:.3f} best {best_val:.3f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        if bad >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model


def ridge_baseline(records, train_sids, val_sids, mu, sd, horizon):
    from sklearn.linear_model import Ridge

    def build_matrix(rec, stride):
        Xs, ys = [], []
        for t in range(L, len(rec["feat"]) - (horizon - 1), stride):
            w = rec["feat"][t - L:t]
            Xs.append(w.reshape(-1)); ys.append(rec["target"][t + horizon - 1])
        return np.stack(Xs), np.array(ys)

    Xa, ya = [], []
    for s in train_sids:
        X, y = build_matrix(records[s], TRAIN_STRIDE); Xa.append(X); ya.append(y)
    Xtr = np.concatenate(Xa); ytr = np.concatenate(ya)
    mu_w = np.concatenate([mu] * L); sd_w = np.concatenate([sd] * L)
    Xtr = (Xtr - mu_w) / sd_w
    best_a, best_v = None, float("inf")
    for al in [0.1, 1.0, 10.0, 100.0]:
        m = Ridge(alpha=al).fit(Xtr, ytr)
        vpred = {}
        for s in val_sids:
            rec = records[s]
            X, _ = build_matrix(rec, 1)
            X = (X - mu_w) / sd_w
            p = np.full(len(rec["target"]), np.nan)
            p[L + horizon - 1:] = m.predict(X)
            vpred[s] = p
        vag, _ = metrics_all(records, val_sids, vpred)
        if vag["rmse"] < best_v:
            best_v, best_a = vag["rmse"], al
    print(f"  ridge best alpha={best_a} val_rmse={best_v:.3f}")
    m = Ridge(alpha=best_a).fit(Xtr, ytr)
    preds = {}
    for s in records:
        rec = records[s]
        X, _ = build_matrix(rec, 1)
        X = (X - mu_w) / sd_w
        p = np.full(len(rec["target"]), np.nan)
        p[L + horizon - 1:] = m.predict(X)
        preds[s] = p
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--models", type=str, default="all")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    records = load_records()
    train_sids, val_sids, test_sids = build_splits(records, args.seed)
    print(f"seed={args.seed} horizon={args.horizon}h", flush=True)
    print("train:", train_sids); print("val  :", val_sids); print("test :", test_sids)
    clim = compute_clim(records, train_sids)
    mu, sd = fit_scaler(records, train_sids)
    coord_mu, coord_sd = coord_stats(records, train_sids)
    for s in records:
        records[s]["feat"] = (records[s]["feat"] - mu) / sd
    in_dim = records[train_sids[0]]["feat"].shape[1]
    out = os.path.join(ROOT, "results", f"seed{args.seed}_h{args.horizon}")
    os.makedirs(out, exist_ok=True)
    summary = {"seed": args.seed, "horizon": args.horizon, "train": train_sids,
               "val": val_sids, "test": test_sids, "device": DEVICE, "models": {}}
    t_all = time.time()

    if args.models in ("all", "baselines"):
        print("\n[baselines]", flush=True)
        # persistence (repeat last observed)
        pp = {}
        for s in records:
            rec = records[s]; h = args.horizon
            p = np.full(len(rec["target"]), np.nan)
            p[h:] = rec["target"][:len(rec["target"]) - h]
            pp[s] = p
        ag, tab = metrics_all(records, test_sids, pp)
        summary["models"]["persistence"] = {"agg": ag, "per_station": tab.to_dict("records")}
        print("  persistence  ", ag)
        # climatology at target time
        pp = {}
        for s in records:
            rec = records[s]
            pp[s] = np.array([clim[rec["hour"][t], rec["month"][t] - 1] for t in range(len(rec["target"]))])
        ag, tab = metrics_all(records, test_sids, pp)
        summary["models"]["climatology"] = {"agg": ag, "per_station": tab.to_dict("records")}
        print("  climatology  ", ag)
        gmean = float(np.concatenate([records[s]["target"] for s in train_sids]).mean())
        pp = {s: np.full(len(records[s]["target"]), gmean) for s in records}
        ag, tab = metrics_all(records, test_sids, pp)
        summary["models"]["global_mean"] = {"agg": ag, "per_station": tab.to_dict("records")}
        print("  global_mean  ", ag)
        print("  ridge (tuning alpha)...", flush=True)
        pp = ridge_baseline(records, train_sids, val_sids, mu, sd, args.horizon)
        ag, tab = metrics_all(records, test_sids, pp)
        summary["models"]["ridge"] = {"agg": ag, "per_station": tab.to_dict("records")}
        print("  ridge        ", ag)

    grid = {
        "gru_plain":        dict(anchor_mode="none", use_spatial=False),
        "gru_anchor_clim":  dict(anchor_mode="clim_mul", use_spatial=False),
        "gru_spatial":      dict(anchor_mode="none", use_spatial=True),
        "ours_clim_add":    dict(anchor_mode="clim_add", use_spatial=True),
        "ours_clim_mul":    dict(anchor_mode="clim_mul", use_spatial=True),
        "ours_pm10_mul":    dict(anchor_mode="pm10_mul", use_spatial=True),
    }
    for name, cfg in grid.items():
        if args.models != "all" and name not in args.models.split(","):
            continue
        print(f"\n[{name}] {cfg}", flush=True)
        t0 = time.time()
        try:
            model = train_model(records, train_sids, val_sids, coord_mu, coord_sd, clim,
                                cfg["anchor_mode"], cfg["use_spatial"], in_dim, DEVICE,
                                args.epochs, args.seed, args.horizon)
            preds = {s: predict_station(model, records[s], coord_mu, coord_sd, clim,
                                        cfg["anchor_mode"], DEVICE, args.horizon) for s in test_sids}
            ag, tab = metrics_all(records, test_sids, preds)
            summary["models"][name] = {"agg": ag, "per_station": tab.to_dict("records")}
            print(f"  TEST {name}: {ag}  ({time.time()-t0:.0f}s)", flush=True)
            np.savez_compressed(os.path.join(out, f"pred_{name}.npz"),
                                **{s: preds[s] for s in test_sids})
            del model, preds; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [{name}] FAILED: {e}", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n==== AGGREGATE TEST (unseen stations) ====")
    rows = []
    for k, v in summary["models"].items():
        a = v["agg"]
        rows.append({"model": k, "RMSE": round(a["rmse"], 3), "MAE": round(a["mae"], 3),
                     "R2": round(a["r2"], 3), "siteRMSE": round(a["site_rmse_mean"], 3)})
    print(pd.DataFrame(rows).sort_values("RMSE").to_string(index=False))
    print(f"\ntotal time {time.time()-t_all:.0f}s, device={DEVICE}")
    print("saved:", os.path.join(out, "summary.json"))


if __name__ == "__main__":
    main()