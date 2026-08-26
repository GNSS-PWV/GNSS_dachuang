# 服务器数据上传清单（不进入 GitHub）

以下数据已经在本机生成/审计，但刻意不提交到 GitHub。将代码仓库克隆到服务器后，按原目录布局上传。

| 本地目录 | 用途 | 规模 | 是否训练必需 |
|---|---|---:|---|
| `第二阶段/profile_reconstructed_igra_20260825/` | 363 个站点的 2014--2019 profile 输入 | 约 5.306 GiB | 是 |
| `第二阶段/result_strict_igra_reconstructed_20260825/` | 严格切分 manifest、profile assignment、站点清单 | 约 0.057 GiB | 是（审计/复现） |

## 不必首次上传

- `igra_batch_20260824/` 及 `手动下载文件/`：原始 IGRA ZIP；profile 已生成，除非需要原始重建复现；
- `第一阶段/result/`、`第二阶段/result*/`：历史模型输出；
- PPT、DOC、XLSX、CSV、缓存和本地 Conda 环境。

## 上传后检查

1. 确认 profile 目录为 363 个 `*_met.txt` 文件；
2. 保留 `split_manifest.json` 与 `profile_assignments.csv`；
3. 在服务器端运行 `第二阶段/strict_profile_preflight.py`；
4. 先以少量文件/少量 epoch 执行 `strict_train.py` dry-run，再通过 SLURM 提交正式训练。

数据的最终校验清单保留在本机 `tongji_upload_manifest_20260826.json`，因其中包含具体数据文件清单而不随代码公开发布。
