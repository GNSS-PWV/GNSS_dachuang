# -*- coding: utf-8 -*-
"""
部署鲁棒性微调: 让第二阶段模型适应"没有真实温度气压"的部署输入
================================================================
输入分布 = 真实探空廓线(概率 1-p) 或 部署式廓线(概率 p):
  部署式廓线 = 气候态平均廓线(最近K训练站, 排除本站) + 地表层 TS/PS/WPS
              替换为"真实地表值 + 一阶段误差噪声(N(0, [3.92,4.37,2.19]))"
微调自 result_aligned 官方模型, 低学习率, SmoothL1 on PWV, 早停按"部署式验证集"RMSE.
评估: 用 phase2_p1_deploy.py --model_dir result_p1deploy_ft 复算 6 种口径.

用法:
  python p1deploy_finetune.py --data_dir <xg_data> --init result_aligned \
      --cache result_grid/st_seasonal_cache.pkl --test_stations test_stations_official_36.txt \
      --max_files 200 --epochs 15 --p_deploy 0.5 --out result_p1deploy_ft
"""
import os, sys, glob, pickle, argparse
import numpy as np
import pandas as pd
import datetime as dt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_all_profiles, GLOBAL_FEATURE_DIM, _encode_global
from model import ProfileTransformer

# Phase1 官方误差 (RMSE)
P1_STD = {'TS': 3.9177, 'PS': 4.3744, 'WPS': 2.1926}
SEASONS = {'DJF': 15, 'MAM': 105, 'JJA': 198, 'SON': 288}
MAX_HEIGHT = 30000.0; BIN_WIDTH = 1000.0


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = np.deg2rad(lat1); p2 = np.deg2rad(lat2)
    dp = np.deg2rad(lat2 - lat1); dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(a))


def season_of(month):
    if month in (12,1,2): return 'DJF'
    if month in (3,4,5): return 'MAM'
    if month in (6,7,8): return 'JJA'
    return 'SON'


def mean_profile(levels_list):
    if not levels_list: return None
    all_elv = np.unique(np.concatenate([lv[:,0] for lv in levels_list]))
    ts = np.full(len(all_elv), np.nan); ps = np.full(len(all_elv), np.nan); wps = np.full(len(all_elv), np.nan)
    for lv in levels_list:
        pos = np.searchsorted(all_elv, lv[:,0])
        ts[pos] = np.nanmean([ts[pos], lv[:,1]], axis=0)
        ps[pos] = np.nanmean([ps[pos], lv[:,2]], axis=0)
        wps[pos] = np.nanmean([wps[pos], lv[:,3]], axis=0)
    ok = ~np.isnan(ts)
    return np.stack([all_elv[ok], ts[ok], ps[ok], wps[ok]], axis=1).astype(np.float32)


class DeployProfileDataset(torch.utils.data.Dataset):
    """训练: 真实/部署式廓线混合; 验证: 部署式(p_deploy=1)."""
    def __init__(self, profiles, scalers, st, train_ids, tl, tn, k, p_deploy,
                 noise_std, rng=None):
        self.profiles = profiles
        self.ls = scalers['level_scaler']; self.gs = scalers['global_scaler']
        self.h_mean = scalers['height_mean']; self.h_std = scalers['height_std']
        self.st = st; self.train_ids = train_ids; self.tl = tl; self.tn = tn; self.k = k
        self.p_deploy = p_deploy; self.noise_std = noise_std
        self.rng = rng if rng is not None else np.random.default_rng(42)
        # 预计算每站每季气候态(排除本站) —— 延迟到首次访问缓存
        self._clim_cache = {}

    def _clim_for(self, station_id, lat, lon, season):
        key = (station_id, season)
        if key in self._clim_cache:
            return self._clim_cache[key]
        d = haversine(self.tl, self.tn, lat, lon)
        idx = np.argsort(d)
        levels_list = []
        for kk in idx:
            s = self.train_ids[kk]
            if s == station_id:
                continue
            if season in self.st[s]['season']:
                lv, _ = self.st[s]['season'][season]
                if len(lv) > 0:
                    levels_list.append(lv)
            if len(levels_list) >= self.k:
                break
        prof = mean_profile(levels_list)
        self._clim_cache[key] = prof
        return prof

    def __len__(self):
        return len(self.profiles)

    def __getitem__(self, i):
        p = self.profiles[i]
        levels = p['levels'].astype(np.float32)
        heights = p['heights'].astype(np.float32)
        real_surf = levels[0, 1:4].copy()  # TS, PS, WPS
        if self.rng.random() < self.p_deploy:
            season = season_of(dt.datetime.fromisoformat(p['time_str'].replace(' ', 'T')[:10]).month)
            clim = self._clim_for(p['station_id'], p['global_raw']['lat'], p['global_raw']['lon'], season)
            if clim is not None:
                levels = clim.copy()
                heights = clim[:, 0].astype(np.float32)
                # 地表替换为 真实地表 + 一阶段误差噪声
                noise = np.array([self.noise_std['TS'], self.noise_std['PS'], self.noise_std['WPS']],
                                 dtype=np.float32) * self.rng.standard_normal(3)
                levels[0, 1:4] = real_surf + noise
        levels_n = self.ls.transform(levels).astype(np.float32)
        heights_n = ((heights - self.h_mean) / self.h_std).astype(np.float32)
        gf = self.gs.transform(_encode_global(p['global_raw']).reshape(1, -1))[0].astype(np.float32)
        zwd = float(p['zwd_surface']); pwv = float(p['pwv_surface'])
        return {'levels': torch.from_numpy(levels_n),
                'heights': torch.from_numpy(heights_n),
                'global': torch.from_numpy(gf),
                'pwv': torch.tensor(pwv, dtype=torch.float32),
                'zwd': torch.tensor(zwd, dtype=torch.float32)}


def collate(batch):
    out = {}
    out['levels'] = torch.nn.utils.rnn.pad_sequence([b['levels'] for b in batch], batch_first=True)
    out['heights'] = torch.nn.utils.rnn.pad_sequence([b['heights'] for b in batch], batch_first=True)
    L = out['levels'].shape[1]
    out['mask'] = torch.zeros(len(batch), L, dtype=torch.bool)
    for j, b in enumerate(batch):
        out['mask'][j, :b['levels'].shape[0]] = True
    out['global'] = torch.stack([b['global'] for b in batch])
    out['pwv'] = torch.stack([b['pwv'] for b in batch])
    out['zwd'] = torch.stack([b['zwd'] for b in batch])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--init', default='result_aligned')
    ap.add_argument('--cache', required=True)
    ap.add_argument('--test_stations', required=True)
    ap.add_argument('--max_files', type=int, default=200)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--p_deploy', type=float, default=0.5)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='result_p1deploy_ft')
    args = ap.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}', flush=True)
    os.makedirs(args.out, exist_ok=True)

    # scalers from init model
    with open(os.path.join(args.init, 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)

    # profiles
    all_profiles = load_all_profiles(args.data_dir, max_files=args.max_files)
    test_ids = set()
    with open(args.test_stations, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith('#'):
                test_ids.add(ln.split()[0])
    all_ids = sorted({p['station_id'] for p in all_profiles})
    train_ids_full = [s for s in all_ids if s not in test_ids]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(train_ids_full))
    n_val = max(1, int(0.15 * len(train_ids_full)))
    val_ids = set([train_ids_full[i] for i in perm[:n_val]])
    train_ids = [train_ids_full[i] for i in perm[n_val:]]
    train_profs = [p for p in all_profiles if p['station_id'] in train_ids]
    val_profs = [p for p in all_profiles if p['station_id'] in val_ids]
    print(f'训练廓线: {len(train_profs)} 站数={len(train_ids)} | 验证廓线: {len(val_profs)} 站数={len(val_ids)}', flush=True)

    # cache for clim profiles (all stations), keep only train stations for nearest-K
    with open(args.cache, 'rb') as f:
        st = pickle.load(f)
    st_train = {s: st[s] for s in st if s not in test_ids}
    train_id_list = sorted(st_train.keys())
    tl = np.array([st_train[s]['lat'] for s in train_id_list])
    tn = np.array([st_train[s]['lon'] for s in train_id_list])
    print(f'气候态缓存站数: {len(train_id_list)}', flush=True)

    ds_tr = DeployProfileDataset(train_profs, scalers, st_train, train_id_list, tl, tn,
                                 args.k, args.p_deploy, P1_STD)
    ds_va = DeployProfileDataset(val_profs, scalers, st_train, train_id_list, tl, tn,
                                 1, 1.0, P1_STD)  # 验证 = 纯部署式
    tr_loader = torch.utils.data.DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                                            collate_fn=collate, num_workers=2)
    va_loader = torch.utils.data.DataLoader(ds_va, batch_size=256, shuffle=False,
                                            collate_fn=collate, num_workers=2)

    # init model
    ckpt = torch.load(os.path.join(args.init, 'best_model.pth'), map_location=device, weights_only=True)
    model = ProfileTransformer(
        d_model=ckpt.get('args', {}).get('d_model', 128),
        n_heads=ckpt.get('args', {}).get('n_heads', 8),
        n_layers=ckpt.get('args', {}).get('n_layers', 4),
        ff_dim=ckpt.get('args', {}).get('ff_dim', 512),
        dropout=ckpt.get('args', {}).get('dropout', 0.1),
        global_feat_dim=GLOBAL_FEATURE_DIM).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'已加载初始模型: {args.init}', flush=True)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr*0.1)
    crit = nn.SmoothL1Loss(beta=1.0)
    best_val = float('inf'); bad = 0; best_state = None

    def eval_loader(loader):
        model.eval(); preds = []; trues = []
        with torch.no_grad():
            for b in loader:
                levels = b['levels'].to(device); heights = b['heights'].to(device)
                gf = b['global'].to(device); mask = b['mask'].to(device)
                pi = model(levels, heights, gf, mask)
                pwv = (pi * b['zwd'].to(device)).cpu().numpy()
                preds.append(pwv); trues.append(b['pwv'].numpy())
        p = np.concatenate(preds); t = np.concatenate(trues)
        return float(np.sqrt(mean_squared_error(t, p)))

    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0; nb = 0
        for b in tr_loader:
            levels = b['levels'].to(device); heights = b['heights'].to(device)
            gf = b['global'].to(device); mask = b['mask'].to(device)
            pwv = b['pwv'].to(device); zwd = b['zwd'].to(device)
            opt.zero_grad()
            pi = model(levels, heights, gf, mask)
            loss = crit(pi * zwd, pwv)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        vr = eval_loader(va_loader)
        print(f'Epoch {ep:3d}/{args.epochs} loss={tot/max(nb,1):.4f} deploy_val_RMSE={vr:.4f}', flush=True)
        if vr < best_val - 1e-5:
            best_val = vr; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= args.patience:
            print('早停', flush=True); break

    model.load_state_dict(best_state)
    torch.save({'model_state_dict': model.state_dict(),
                'args': ckpt.get('args', {}),
                'best_val_rmse': best_val}, os.path.join(args.out, 'best_model.pth'))
    with open(os.path.join(args.out, 'scalers.pkl'), 'wb') as f:
        pickle.dump(scalers, f)
    with open(os.path.join(args.out, 'metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f'p_deploy={args.p_deploy} k={args.k} epochs={ep}\nbest deploy_val_RMSE={best_val:.5f}\n')
    print(f'完成, best deploy_val_RMSE={best_val:.5f} -> {args.out}', flush=True)


if __name__ == '__main__':
    main()