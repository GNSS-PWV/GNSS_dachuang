# -*- coding: utf-8 -*-
"""COLD-START protocol: predict PM2.5 at UNSEEN stations using ONLY
continuous spatial coordinates + time features (+ optional climatology anchor).
No local history, no local meteorology. This isolates the "continuous spatial
index embedding" pillar (zero-shot at brand-new locations).

Models: mlp_plain / mlp_clim / mlp_spatial / mlp_spatial_clim  (L=1 -> MLP-ish)
Baselines: global_mean, climatology, ridge(+coords).
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

TIME_COLS = [FEAT_COLS.index(c) for c in
             ("hour_sin", "hour_cos", "doy_sin", "doy_cos", "doy_sin2", "doy_cos2")]
L = 1
BATCH = 1024
LR = 1e-3
PATIENCE = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_records():
    return pd.read_pickle(os.path.join(ROOT, "data", "prepared.pkl"))


def build_splits(records, seed):
    sids = sorted(records.keys())
    rng = np.random.default_rng(seed)
    sids = [sids[i] for i in rng.permutation(len(sids))]
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


def coord_stats(records, sids):
    C = np.stack([records[s]["coord"] for s in sids])
    C[:, 2] = np.log1p(C[:, 2])
    mu = C.mean(0); sd = C.std(0) + 1e-6
    return mu, sd


def norm_coord(coord, mu, sd):
    c = coord.copy(); c[2] = np.log1p(c[2])
    return (c - mu) / sd


class ColdStart(Dataset):
    def __init__(self, rec, coord_mu, coord_sd, clim, anchor_mode, stride=1):
        self.rec = rec
        self.coord = norm_coord(rec["coord"], coord_mu, coord_sd).astype(np.float32)
        self.clim = clim
        self.anchor_mode = anchor_mode
        self.idx = list(range(0, len(rec["feat"]), stride))
        # time-only features (1 timestep)
        self.x = rec["feat"][:, TIME_COLS].astype(np.float32)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        t = self.idx[i]
        x = self.x[t:t + 1]  # (1, 6)
        if self.anchor_mode.startswith("clim"):
            a = float(self.clim[self.rec["hour"][t], self.rec["month"][t] - 1])
        else:
            a = 0.0
        return x, self.coord, np.float32(a), np.float32(self.rec["target"][t]), \
               bool(self.rec["target_ok"][t])


def collate(batch):
    xs = torch.from_numpy(np.stack([b[0] for b in batch]))
    cs = torch.from_numpy(np.stack([b[1] for b in batch]))
    a = torch.from_numpy(np.stack([b[2] for b in batch]))
    y = torch.from_numpy(np.stack([b[3] for b in batch]))
    ok = torch.from_numpy(np.stack([np.float32(b[4]) for b in batch]))
    return xs, cs, a, y, ok


def predict(model, rec, coord_mu, coord_sd, clim, anchor_mode, device):
    ds = ColdStart(rec, coord_mu, coord_sd, clim, anchor_mode, stride=1)
    dl = DataLoader(ds, batch_size=4096, shuffle=False, collate_fn=collate, num_workers=0)
    preds = np.full(len(rec["target"]), np.nan, dtype=np.float64)
    model.eval(); off = 0
    with torch.no_grad():
        for x, c, a, y, ok in dl:
            p = model(x.to(device), c.to(device), a.to(device)).cpu().numpy()
            for j in range(p.shape[0]):
                preds[ds.idx[off + j]] = p[j]
            off += p.shape[0]
    return preds


def metrics_all(records, sids, preds_map):
    rows = []; Ys = []; Ps = []
    for s in sids:
        rec = records[s]; p = preds_map[s]
        ok = rec["target_ok"]; y = rec["target"][ok]; pp = p[ok]
        m = ~np.isnan(pp)
        rmse = float(np.sqrt(np.mean((y[m] - pp[m]) ** 2)))
        mae = float(np.mean(np.abs(y[m] - pp[m])))
        rows.append({"station": s, "rmse": rmse, "mae": mae})
        Ys.append(y[m]); Ps.append(pp[m])
    Y = np.concatenate(Ys); P = np.concatenate(Ps)
    return {"n": int(len(Y)), "rmse": float(np.sqrt(np.mean((Y - P) ** 2))),
            "mae": float(np.mean(np.abs(Y - P))),
            "r2": float(1 - np.sum((Y - P) ** 2) / (np.sum((Y - Y.mean()) ** 2) + 1e-12)),
            "site_rmse_mean": float(np.mean([r["rmse"] for r in rows]))}, pd.DataFrame(rows)


def train_model(records, train_sids, val_sids, coord_mu, coord_sd, clim,
                anchor_mode, use_spatial, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    ds = torch.utils.data.ConcatDataset(
        [ColdStart(records[s], coord_mu, coord_sd, clim, anchor_mode, stride=2) for s in train_sids])
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=0, drop_last=True)
    model = GeoIndexGRU(in_dim=6, hidden=64, layers=2, anchor=anchor_mode,
                        use_spatial=use_spatial).to(DEVICE)
    torch.cuda.empty_cache()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
    lossf = nn.MSELoss()
    best_val = float("inf"); best_state = None; bad = 0
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0; nb = 0
        for x, c, a, y, ok in dl:
            x, c, a, y = x.to(DEVICE), c.to(DEVICE), a.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); p = model(x, c, a); loss = lossf(p, y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += loss.item(); nb += 1
        vpred = {s: predict(model, records[s], coord_mu, coord_sd, clim, anchor_mode, DEVICE) for s in val_sids}
        vag, _ = metrics_all(records, val_sids, vpred)
        vr = vag["rmse"]; sched.step(vr)
        if vr < best_val - 1e-4:
            best_val = vr; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 10 == 0 or bad == 1:
            print(f"  ep {ep:3d} loss {tot/nb:.4f} val_rmse {vr:.3f} best {best_val:.3f}", flush=True)
        if bad >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model


def ridge_coords(records, train_sids, val_sids, coord_mu, coord_sd, clim):
    from sklearn.linear_model import Ridge
    def feats(rec):
        X = rec["feat"][:, TIME_COLS].copy()
        C = norm_coord(rec["coord"], coord_mu, coord_sd)
        Xc = np.concatenate([X, np.tile(C, (len(X), 1))], axis=1)
        return np.ascontiguousarray(Xc, dtype=np.float32)
    Xtr = np.ascontiguousarray(np.concatenate([feats(records[s]) for s in train_sids]), dtype=np.float32)
    ytr = np.ascontiguousarray(np.concatenate([records[s]["target"] for s in train_sids]), dtype=np.float32)
    best_a, best_v = None, float("inf")
    for al in [0.1, 1.0, 10.0, 100.0]:
        m = Ridge(alpha=al, solver="lsqr", copy_X=False).fit(Xtr, ytr)
        vpred = {s: m.predict(feats(records[s])) for s in val_sids}
        vag, _ = metrics_all(records, val_sids, vpred)
        if vag["rmse"] < best_v:
            best_v, best_a = vag["rmse"], al
    print(f"  ridge+coords best alpha={best_a} val_rmse={best_v:.3f}")
    m = Ridge(alpha=best_a, solver="lsqr", copy_X=False).fit(Xtr, ytr)
    return {s: m.predict(feats(records[s])) for s in records}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()
    records = load_records()
    train_sids, val_sids, test_sids = build_splits(records, args.seed)
    print("cold-start seed", args.seed); print("train:", train_sids); print("val:", val_sids); print("test:", test_sids)
    clim = compute_clim(records, train_sids)
    coord_mu, coord_sd = coord_stats(records, train_sids)
    out = os.path.join(ROOT, "results", f"coldstart_seed{args.seed}")
    os.makedirs(out, exist_ok=True)
    summary = {"seed": args.seed, "models": {}}
    # baselines
    gmean = float(np.concatenate([records[s]["target"] for s in train_sids]).mean())
    pp = {s: np.full(len(records[s]["target"]), gmean) for s in records}
    ag, _ = metrics_all(records, test_sids, pp); summary["models"]["global_mean"] = ag; print("global_mean", ag)
    pp = {}
    for s in records:
        rec = records[s]
        pp[s] = np.array([clim[rec["hour"][t], rec["month"][t] - 1] for t in range(len(rec["target"]))])
    ag, _ = metrics_all(records, test_sids, pp); summary["models"]["climatology"] = ag; print("climatology", ag)
    print("ridge+coords...", flush=True)
    pp = ridge_coords(records, train_sids, val_sids, coord_mu, coord_sd, clim)
    ag, _ = metrics_all(records, test_sids, pp); summary["models"]["ridge_coords"] = ag; print("ridge_coords", ag)
    grid = {
        "mlp_plain":         dict(anchor_mode="none", use_spatial=False),
        "mlp_clim":          dict(anchor_mode="clim_mul", use_spatial=False),
        "mlp_spatial":       dict(anchor_mode="none", use_spatial=True),
        "mlp_spatial_clim":  dict(anchor_mode="clim_mul", use_spatial=True),
    }
    for name, cfg in grid.items():
        print(f"\n[{name}]", flush=True)
        model = train_model(records, train_sids, val_sids, coord_mu, coord_sd, clim,
                            cfg["anchor_mode"], cfg["use_spatial"], args.epochs, args.seed)
        pp = {s: predict(model, records[s], coord_mu, coord_sd, clim, cfg["anchor_mode"], DEVICE) for s in test_sids}
        ag, tab = metrics_all(records, test_sids, pp)
        summary["models"][name] = ag
        print(f"  TEST {name}: {ag}", flush=True)
        np.savez_compressed(os.path.join(out, f"pred_{name}.npz"), **{s: pp[s] for s in test_sids})
        del model, pp; gc.collect()
        torch.cuda.empty_cache()
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    rows = [{"model": k, "RMSE": round(v["rmse"], 3), "MAE": round(v["mae"], 3),
             "R2": round(v["r2"], 3), "siteRMSE": round(v["site_rmse_mean"], 3)}
            for k, v in summary["models"].items()]
    print("\n==== COLD-START TEST (unseen stations) ====")
    print(pd.DataFrame(rows).sort_values("RMSE").to_string(index=False))
    print("saved:", os.path.join(out, "summary.json"))


if __name__ == "__main__":
    main()