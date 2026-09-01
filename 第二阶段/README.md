# 第二阶段: ZWD -> PWV 直接映射深度学习模型

> **Status as of 2026-09-01:** The numerical results below are historical
> radiosonde/reconstructed-profile pipeline checks, not independent GNSS-PWV
> deployment accuracy. Do not report the `0.0653 mm` value or GPT3 improvement
> percentages as final GNSS-PWV performance. Formal retraining is gated by
> `preflight_independent_phase2_inputs.py` and requires independent GNSS
> `ZTD-ZHD`, causal phase-one profiles, and independent PWV labels.

## 概述

本阶段实现项目的核心创新 —— 基于 Transformer 的廓线序列深度学习框架，直接从湿延迟(ZWD)映射到可降水量(PWV)，摆脱传统 Saastamoinen 模型对加权平均温度 Tm 的线性假设。

**核心思路**: 利用全球探空气球全廓线（温度、气压、水汽压的垂直分布）作为监督数据，让 Transformer 学习大气垂直结构到水汽转换系数 Pi 的非线性映射，最终 `PWV = Pi x ZWD`。

## 最终结果（官方 36 测试站, N=113,294）

| 方法 | RMSE (mm) | MAE (mm) | R2 | Bias (mm) |
|------|----------:|---------:|-----:|----------:|
| Saastamoinen（全流程） | 7.127 | 4.944 | 0.813 | -1.026 |
| Saastamoinen（Pi-only, 真值Tm） | 0.255 | 0.155 | 0.9998 | +0.098 |
| GPT3（经验Tm） | 0.345 | 0.249 | 0.9996 | -0.183 |
| **ProfileTransformer（不用Tm）** | **0.0653** | **0.0370** | **1.0000** | **+0.0077** |

- 相对 GPT3 提升 **81.1%**；相对真值 Tm 线性公式提升 74.4%；相对传统全流程提升约 99%
- 极端场景（特湿/强变化）相对 GPT3 提升 75-82%；全球格网化 Pi 产品已生成（0.136-0.171）
- 完整结果与过程见 `服务器全量训练结果与任务总结.md`

## 文件结构

```
第二阶段/
├── data.py             # 数据管线: 读探空->变长廓线->归一化->站点切分(支持 --test_stations 官方名单)
├── model.py            # ProfileTransformer 模型
├── train.py            # 训练 (Huber损失+余弦退火+早停+测试评估+出图)
├── evaluate.py         # 三方法对比评估 (输出含 time/lat/lon/elv/tm 的预测明细)
├── loso_evaluate.py    # 留一站交叉验证 (小数据稳健评估)
├── gpt3_baseline.py    # GPT3 基线对比 (GPT3 经验 Tm vs Transformer)
├── phase3_analysis.py  # 第三阶段分层分析 (季节/纬度带/ZWD)
├── extreme_analysis.py # 极端场景验证 (特湿/强变化等子集, 与 GPT3 对比)
├── grid_product.py     # 全球格网化转换系数 Pi 产品 (创新点三)
├── deploy_simulation.py# 端到端部署模拟 (无探空时用气候态廓线)
├── bias_analysis.py    # 误差订正分析 (全局/逐站留一订正)
├── optuna_search.py    # Optuna 超参搜索 (模型优化)
├── time_series_plots.py# 典型站 PWV 时间序列图 (论文材料)
├── run_train.sh        # 服务器 SLURM 脚本 (随机90/10切分)
├── run_train_aligned.sh# 官方36测试站对齐 + GPT3基线 (正式流程)
├── run_gpt3_baseline.sh# 独立 GPT3 基线脚本
├── run_grid_product.sh # 格网化产品脚本
├── test_stations_official_36.txt  # 第一阶段官方36测试站名单
├── requirements.txt
└── README.md
```

## 运行方法

### 本地
```bash
python data.py                          # 数据管线自检
python train.py --data_dir D:/.../xg_test
python evaluate.py --data_dir D:/.../xg_test --model_path result/best_model.pth
python loso_evaluate.py --data_dir D:/.../xg_test
```

### 服务器（全量）
```bash
# 官方36站对齐训练 + 评估 + GPT3基线 (正式流程)
sbatch run_train_aligned.sh
# 或随机90/10切分
sbatch run_train.sh
```

### 分析脚本（需先有 evaluate.py 产出的 test_predictions.csv）
```bash
python gpt3_baseline.py --csv result/test_predictions.csv --grd ../gpt3_1/gpt3_1.grd --out result_gpt3
python phase3_analysis.py --csv result/test_predictions.csv --out result_phase3
python extreme_analysis.py --csv result/test_predictions.csv --grd ../gpt3_1/gpt3_1.grd --out result_extreme
python bias_analysis.py --csv result/test_predictions.csv --out result_bias
python grid_product.py --model_dir result --data_dir <xg_data> --out result_grid
python deploy_simulation.py --csv result/test_predictions.csv --model_dir result --data_dir <xg_data> --cache result_grid/st_seasonal_cache.pkl --test_stations test_stations_official_36.txt --out result_deploy
```

## 模型架构

- 输入: 廓线层特征 [ELV, TS, PS, WPS] + 连续高度位置编码 + 全局特征 [ZWD, lat/lon, DOY/hour]
- Transformer Encoder (Pre-LN, 4层8头, d_model=128) + CLS token 聚合 + 全局特征融合 -> MLP -> Pi
- 输出: Pi (sigmoid 限制 0.05-0.35), PWV = Pi x ZWD
- 参数量 85.3 万, 训练: AdamW(lr=1e-4) + 余弦退火, SmoothL1 loss, batch=128, 100 epochs

## 关键发现

1. 线性 Tm 公式存在系统性、随站点变化的残差(~0.002 Pi), 正是模型的靶子
2. 模型完全不用 Tm, 精度显著优于 GPT3 经验 Tm(81%) 与真值 Tm 线性公式(74%)
3. 极端高湿/强变化场景优势更大(相对 GPT3 提升 75-82%)
4. 全球格网化 Pi 产品: 极地 0.147 -> 热带 0.166, 夏季>冬季, 与 Tm 气候学一致
5. 部署模拟: 真实廓线 0.065mm vs 气候态廓线 0.965mm -> 部署需实时/预报廓线(ERA5/NWP)
6. 误差订正: 整体偏差仅 +0.008mm, 逐站订正可再降 18%
