# -*- coding: utf-8 -*-
"""Generic non-uniform spatio-temporal forecaster (CCF-A feasibility prototype).

Transfers the three methodological pillars of the PWV ProfileTransformer to a
general multi-station time-series setting:
  1) Continuous spatial-index embedding  (PWV: continuous height/pressure index PE
     -> here: continuous (lon, lat, elev) coordinate PE). Enables zero-shot
     deployment at UNSEEN stations (no discrete station-ID table).
  2) Global/context conditioning        (PWV: site meteorological params condition
     profile -> here: temporal context + spatial embedding condition each step).
  3) Physical-anchor reparameterization  (PWV: PWV = Pi x ZWD with observed ZWD
     -> here: y = anchor * exp(g) or anchor + g, anchor = diurnal-seasonal
     climatology or observed co-pollutant PM10).
"""
import math
import torch
import torch.nn as nn


def sinusoidal_pe(coords, n_freq=8, base=2.0):
    """Continuous coordinate PE. coords: (..., C) in normalized space."""
    freqs = base ** torch.arange(n_freq, device=coords.device, dtype=coords.dtype)  # (n_freq,)
    ang = coords.unsqueeze(-1) * freqs                     # (..., C, n_freq)
    s = torch.sin(ang).flatten(-2)
    c = torch.cos(ang).flatten(-2)
    return torch.cat([s, c], dim=-1)                       # (..., C*n_freq*2)


class GeoIndexGRU(nn.Module):
    """GRU encoder + spatial-index embedding + anchor reparameterization.

    anchor in {"none","clim_add","clim_mul","pm10_mul","pm10_add"}.
    If use_spatial=False, the model degenerates to a plain GRU (baseline).
    """

    def __init__(self, in_dim, hidden=64, layers=2, coord_dim=3, n_freq=8,
                 anchor="clim_mul", use_spatial=True, dropout=0.1):
        super().__init__()
        self.anchor = anchor
        self.use_spatial = use_spatial
        self.n_freq = n_freq
        self.coord_dim = coord_dim
        sp_dim = coord_dim * n_freq * 2
        enc_in = in_dim + (hidden if use_spatial else 0)
        if use_spatial:
            self.spatial_proj = nn.Sequential(
                nn.Linear(sp_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.encoder = nn.GRU(enc_in, hidden, num_layers=layers, batch_first=True,
                              dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, coords, anchor_val):
        """x: (B, L, D_in); coords: (B, C) normalized; anchor_val: (B,) or None."""
        if self.use_spatial:
            pe = sinusoidal_pe(coords, self.n_freq)
            sp = self.spatial_proj(pe).unsqueeze(1).expand(-1, x.size(1), -1)
            xin = torch.cat([x, sp], dim=-1)
        else:
            xin = x
        out, _ = self.encoder(xin)
        g = self.head(out[:, -1]).squeeze(-1)
        if self.anchor in ("clim_mul", "pm10_mul"):
            # Clamp only the exponent for numerical stability.  Clamping the
            # direct-regression path would cap anchor-free predictions at 12.
            pred = anchor_val * torch.exp(torch.clamp(g, -12.0, 12.0))
        elif self.anchor in ("clim_add", "pm10_add"):
            pred = anchor_val + g
        elif self.anchor == "none":
            pred = g
        else:
            raise ValueError(self.anchor)
        return pred