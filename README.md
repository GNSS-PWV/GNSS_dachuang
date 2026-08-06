# GNSS 水汽反演（基于深度学习的两步法实时 GNSS 水汽反演）

> 国家级大学生创新创业训练计划项目（同济大学）
> 团队成员：王乐萱、彭怡焱（Pengyiyan0411）、钱顾冲、程禹卜、李乾凯而　|　指导老师：吴志露

## 项目简介

突破传统 GNSS 水汽反演对加权平均温度 Tm 和表面气压 Ps 的强依赖，用**全球探空廓线数据驱动**的方式实现 ZWD → PWV 直接转换，目标 PWV RMSE < 1.5 mm、优于 GPT3 约 25–40%。

**两步法技术路线**
- **第一步（第一阶段）**：GRU + XGBoost 预测 4 个气象参数（Ps / WPS / Ts / Tm）
- **第二步（第二阶段）**：ProfileTransformer 直接从探空廓线预测转换系数 Pi，`PWV = Pi × ZWD`，**完全不用 Tm**

## 仓库结构

```
GNSS_dachuang/
├── phase1/                     # 第一阶段: 4 参数 GRU+XGB
│   ├── PS/ WPS/ Ts_Tm/         #   各参数训练代码与 SLURM 脚本
│   ├── gru_cnn.py / test2.py   #   早期 CNN 实验 (LSTM/CNN-BiLSTM-Attention)
│   ├── analysis/               #   官方指标汇总 / GRU vs CNN 对比 / GRU+XGB 消融
│   └── result/                 #   训练/测试站点信息等
├── phase2/                     # 第二阶段: ProfileTransformer ZWD→PWV
│   ├── data.py / model.py / train.py / evaluate.py / loso_evaluate.py
│   ├── analysis/               #   GPT3基线 / 分层 / 极端 / 格网化 / 部署模拟 / 误差订正 / Optuna / 时序图 / 端到端
│   ├── run_train.sh / run_train_aligned.sh
│   ├── results/                #   关键结果指标表 (CSV)
│   └── README.md               #   第二阶段详细说明
├── docs/                       # 项目文档 (交接/总结/模型对比分析)
├── legacy/test1/               # 早期实验存档 (2014 数据 + CNN 模型权重)
├── 模型解释.ppt
└── README.md
```

## 关键结果

### 第一阶段（GRU+XGB 预测 4 参数，2017 测试年官方精度）

| 参数 | RMSE | 单位 |
|------|-----:|------|
| Ps | 4.37 | hPa |
| WPS | 2.19 | hPa |
| Ts | 3.92 | K |
| Tm | 3.72 | K |

**加 CNN 为什么变差**：同条件对比 GRU (3.87 hPa) vs CNN1D (4.15, +7%) vs CNN-BiLSTM-Attention (7.78, +101%) —— 过参数化 + MaxPool 丢时间分辨率 + 卷积对平滑时序错配。详见 `docs/第一阶段_模型对比分析.md`。

### 第二阶段（ProfileTransformer，官方 36 测试站）

| 方法 | RMSE (mm) |
|------|----------:|
| Saastamoinen 全流程 | 7.13 |
| Saastamoinen (Pi-only, 真值Tm) | 0.255 |
| GPT3 (经验 Tm) | 0.345 |
| **ProfileTransformer（不用 Tm）** | **0.065**（Optuna 优化后 0.061） |

- 相对 GPT3 提升 **~82%**；极端场景（特湿/强变化）相对 GPT3 提升 75–82%
- **Optuna 最优超参重训后 0.061 mm**（官方 36 站，较基线 0.065 提升 6.7%）
- 全球格网化转换系数 Pi 产品：极地 0.147 → 热带 0.166（5°×5°×月尺度）
- **端到端两步法**（2017，N=18.4万）：传统两步法 GRU-Tm 0.224 / GPT3 0.332 / Transformer 0.055——完整梯度见 `docs/服务器全量训练结果与任务总结.md`
- **GNSS 泛化验证**（IGS 2019 ZTD，9 中国站）：GNSS 反演 PWV 物理合理，武汉 2019 与探空真值对比 RMSE=5.78mm、R2=0.897

## 运行方式

数据在服务器（`/share/home/u23114/tj23114/packages/dachuang_pwv/PS/xg_data`），用 SLURM 提交：

```bash
# 第二阶段: 官方36站对齐训练 + 评估 + GPT3基线
cd phase2 && sbatch run_train_aligned.sh
# 或随机 90/10 切分
sbatch run_train.sh
```

本地分析（需先有 evaluate.py 产出的 `test_predictions.csv`）：见 `phase2/README.md`。

## 文档

- `docs/项目当前情况与下一阶段交接.md` —— 项目全貌与交接
- `docs/服务器全量训练结果与任务总结.md` —— 第二阶段完整结果
- `docs/第一阶段_模型对比分析.md` —— CNN 为何变差 + GRU+XGB 分析

## 数据说明

探空数据（IGRA，2014–2019）在服务器上，未随仓库分发。`legacy/test1/2014_sdata/` 含少量样本数据用于早期实验。
