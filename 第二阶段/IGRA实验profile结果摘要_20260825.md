# IGRA 实验 profile 结果摘要（2026-08-25）

## 结果

- 固定版本：2026-08-23 IGRA 目录。
- CRC 有效 ZIP：363 个；无目录匹配：`ICM00004018`、`POM00008579`。
- 实验性 profile 输出：
  `profile_reconstructed_igra_20260825`。
- 全量 profile：1,310,588；字段预检：PASS；字段错误：0。
- 严格切分输出：`result_strict_igra_reconstructed_20260825`。

| 集合 | Profile 数 | 站点数 | 年份 |
|---|---:|---:|---|
| train | 584,904 | 313 | 2014–2016 |
| test_2017 | 198,654 | 310 | 2017 |
| val_2018 | 198,262 | 310 | 2018 |
| val_leave_station | 110,230 | 35 | 2014–2018 |
| val_2019 | 218,538 | 342 | 2019 |

## 重要限制

1. 这是 IGRA 水汽积分 + 物理 Tm-to-ZWD 的实验性重建标签，不是老师提供的正式标签。
2. 实际 2014–2018 有可接受 profile 的站点为 350，故留站点数为 `floor(350*0.1)=35`；不能按原始目标清单强行补到 36。
3. 13 个 CRC 有效站点转换后没有可接受 profile（仅有表头）；2 个站点在固定 IGRA 目录中不存在。
4. EC 数据、ERA5 真实 NetCDF 积分和同济服务器连接仍未完成。
5. 当前不应据此宣称正式模型训练成果；需老师确认标签口径或提供正式 profile。

## 复核文件

- `igra_batch_20260824\final_audit_20260823.csv`
- `第二阶段\profile_reconstructed_igra_20260825.json`
- `第二阶段\profile_reconstructed_igra_20260825_preflight.json`
- `第二阶段\result_strict_igra_reconstructed_20260825\split_manifest.json`
