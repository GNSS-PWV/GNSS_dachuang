# -*- coding: utf-8 -*-
"""
第二阶段模型: ProfileTransformer

核心架构:
  1. 每个高度层的探空特征 (ELV, TS, PS, WPS) -> 线性投影到 d_model 维
  2. 连续高度位置编码 (sinusoidal on normalized height), 而非离散位置索引
  3. 可学习 CLS token 前置, 聚合整条廓线信息
  4. Transformer Encoder 多头自注意力处理变长序列
  5. CLS 输出 + 全局特征 (ZWD, lat/lon, DOY/hour) 融合 -> MLP -> 转换系数 Pi
  6. PWV = Pi * ZWD, 直接映射, 摆脱 Tm 线性假设

创新点:
  - 用大气垂直结构 (温度/气压/水汽压的垂直分布) 替代单一 Tm 标量
  - Transformer 能捕捉温度倒挂、对流层顶变化等复杂垂直结构
  - 连续高度编码使模型对不同探空层分辨率具有泛化性
"""
import math
import torch
import torch.nn as nn


class HeightPositionalEncoding(nn.Module):
    """
    连续高度位置编码.

    与标准 Transformer 的离散位置编码不同, 这里对归一化后的高度值
    施加多频率 sinusoidal 编码, 使模型能感知绝对高度信息.
    """

    def __init__(self, d_model, num_freqs=16, max_freq=1000.0):
        super().__init__()
        self.d_model = d_model
        # 频率从 1 到 max_freq 等比数列
        freqs = torch.pow(
            max_freq, torch.linspace(0, 1, num_freqs)
        )
        self.register_buffer('freqs', freqs)  # (num_freqs,)

    def forward(self, heights):
        """
        参数:
          heights: (B, L) 归一化后的高度值
        返回:
          (B, L, d_model)  位置编码
        """
        # (B, L, 1) * (num_freqs,) -> (B, L, num_freqs)
        h = heights.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0)
        # sin + cos 交替 -> (B, L, 2*num_freqs)
        pe = torch.cat([torch.sin(h), torch.cos(h)], dim=-1)
        # 截断或补齐到 d_model
        if pe.shape[-1] >= self.d_model:
            pe = pe[..., :self.d_model]
        else:
            pad = torch.zeros(*pe.shape[:-1], self.d_model - pe.shape[-1],
                              device=pe.device, dtype=pe.dtype)
            pe = torch.cat([pe, pad], dim=-1)
        return pe


class ProfileTransformer(nn.Module):
    """
    廓线序列 Transformer 模型: 从垂直廓线 + 地表 ZWD 预测转换系数 Pi.

    Pi = PWV / ZWD, 最终 PWV = Pi * ZWD.
    """

    def __init__(self,
                 level_feat_dim=4,
                 global_feat_dim=9,
                 d_model=128,
                 n_heads=8,
                 n_layers=4,
                 ff_dim=512,
                 dropout=0.1,
                 num_height_freqs=16):
        """
        参数:
          level_feat_dim : 每层输入特征维度 (ELV, TS, PS, WPS = 4)
          global_feat_dim: 全局特征维度 (ZWD, lat/lon, DOY/hour = 9)
          d_model        : Transformer 隐藏维度
          n_heads        : 多头注意力头数
          n_layers       : Transformer 编码器层数
          ff_dim         : 前馈网络维度
          dropout        : dropout 比率
          num_height_freqs: 高度位置编码的频率数
        """
        super().__init__()
        self.d_model = d_model

        # 层特征投影
        self.level_proj = nn.Linear(level_feat_dim, d_model)
        self.level_norm = nn.LayerNorm(d_model)

        # 高度位置编码
        self.height_pe = HeightPositionalEncoding(d_model, num_freqs=num_height_freqs)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,  # Pre-LN, 训练更稳定
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        # 全局特征投影
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # 融合 + 输出头: CLS 输出 + 全局投影 -> Pi
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Pi 的合理范围约 0.1~0.2, 用 sigmoid 限制输出
        self.pi_min = 0.05
        self.pi_max = 0.35
        self.sigmoid_scale = self.pi_max - self.pi_min

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, levels, heights, global_feat, attention_mask):
        """
        参数:
          levels          : (B, L, F)   归一化后的层特征
          heights         : (B, L)       归一化后的高度
          global_feat     : (B, G)       归一化后的全局特征
          attention_mask  : (B, L)       True=有效层, False=填充

        返回:
          pi  : (B,)  转换系数
          pwv : (B,)  PWV = pi * zwd (需在外部传入 zwd)
        """
        B, L, F = levels.shape

        # 层特征投影 + 位置编码
        x = self.level_proj(levels)  # (B, L, d)
        x = self.level_norm(x)
        pe = self.height_pe(heights)  # (B, L, d)
        x = x + pe

        # 前置 CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d)
        x = torch.cat([cls, x], dim=1)  # (B, L+1, d)

        # 扩展 attention mask: CLS token 始终有效
        cls_mask = torch.ones(B, 1, dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([cls_mask, attention_mask], dim=1)  # (B, L+1)

        # Transformer 需要: True 表示被 mask 掉 (不参与注意力)
        # PyTorch nn.MultiheadAttention 的 key_padding_mask: True = padding
        key_padding_mask = ~full_mask  # (B, L+1), True=需要mask

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.encoder_norm(x)

        # CLS token 输出
        cls_out = x[:, 0, :]  # (B, d)

        # 全局特征投影
        g = self.global_proj(global_feat)  # (B, d)

        # 融合
        combined = torch.cat([cls_out, g], dim=-1)  # (B, 2d)
        pi_raw = self.head(combined).squeeze(-1)  # (B,)

        # sigmoid 限制 Pi 范围
        pi = self.pi_min + torch.sigmoid(pi_raw) * self.sigmoid_scale

        return pi

    def predict_pwv(self, levels, heights, global_feat, attention_mask, zwd):
        """便捷方法: 直接返回 PWV = Pi * ZWD."""
        pi = self.forward(levels, heights, global_feat, attention_mask)
        return pi * zwd