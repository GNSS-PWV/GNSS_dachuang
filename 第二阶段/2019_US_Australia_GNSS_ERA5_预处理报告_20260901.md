# 2019 美国/澳大利亚 GNSS–ERA5 独立输入预处理报告

**生成日期：** 2026-09-01  \n+**数据根目录：** `D:\gnss水汽反演\2019_US_Australia_IGS_ERA5`  \n+**预处理脚本：** `prepare_2019_gnss_era5_independent.py`  \n+**派生产物：** `independent_2019_us_australia_era5_v1/`

## 1. 目的与科学边界

本次工作仅完成独立输入链中的 GNSS 侧预处理：

```text
IGS 5 分钟 ZTD + ERA5 单层气压/温度/位势高度
    -> Saastamoinen ZHD
    -> ZWD = ZTD - ZHD
```

输出的 `era5_tcwv_reference_mm` 仅是 ERA5 模式的总柱水汽参考量，**不是独立 PWV 真值标签**。因此本产物不能直接作为二阶段正式训练集，也不能由此报告二阶段 PWV RMSE。

## 2. 数据审计

- 文件级 SHA-256 清单校验通过。
- 时间覆盖为 2019 年 1、4、7、10 月各 1–10 日，共 **40 日**；并非全年数据。
- GNSS 共 80 个日 CSV、609,984 条 5 分钟记录：
  - 澳大利亚：22 站，238,464 条；
  - 美国 CONUS：43 站，371,520 条。
- ERA5 为 0.25°、逐小时单层产品：`t2m (K)`、`sp (Pa)`、`tcwv (kg m^-2)`；静态 `z (m^2 s^-2)` 以 `z / 9.80665` 转为 geopotential height。
- 当前包不含气压层温度、比湿和位势高度，因此不具备完整垂直 profile。

## 3. 匹配与计算口径

1. **时间：** 每个 GNSS 5 分钟时刻与同一 ERA5 十日文件中的最近整点匹配；只接受时间差不超过 1800 秒。
2. **空间：** 匹配最近的 0.25° ERA5 经纬网格点。
3. **气压高程改正：** 以 IGS 椭球高减去 ERA5 geopotential height 得到临时高度差，按标准温度递减率 0.0065 K/m 及静力关系修正 ERA5 `sp` 到站高。
4. **ZHD：**

   `ZHD(mm) = 2.2768 × P(hPa) / [1 − 0.00266 cos(2φ) − 0.00028 H(km)]`

5. **ZWD：** `ZWD(mm) = ZTD_IGS(mm) − ZHD(mm)`。
6. **质控：** `1500≤ZTD≤3500 mm`、`0≤ZTD sigma≤20 mm`、`1500≤ZHD≤3200 mm`、`0<ZWD≤700 mm`、`0≤TCWV≤150 mm`，及上述时间容差。

> 高程基准当前为临时且显式记录的 `ellipsoidal height − geopotential height` 处理。正式试验前须由老师确认是否需要 EGM96/EGM2008 正高转换及统一的 ZHD 口径。

## 4. 全量结果

| 区域 | 记录数 | 站点数 | 质控通过 | 通过率 |
|---|---:|---:|---:|---:|
| 澳大利亚 | 238,464 | 22 | 238,030 | 99.82% |
| 美国 CONUS | 371,520 | 43 | 368,809 | 99.27% |
| **总计** | **609,984** | **65** | **606,839** | **99.48%** |

质控失败原因（失败行不被静默删除，而是保留失败标记）：

- `zwd_range`：2,088 条；
- `era5_time`：1,060 条；
- 两项同时失败：3 条。

`era5_time` 主要落在每个十日 ERA5 文件第 10 天的 23:35–23:55 UTC：由于包内没有下一日 00:00 场，最近可用 ERA5 时刻为 23:00，时间差超过 30 分钟。后续可选择明确剔除这些边界时刻，或补齐第 11 天 00:00 ERA5；不得无记录地外推。

## 5. 产物与复现

```text
第二阶段/independent_2019_us_australia_era5_v1/
├── gnss_era5_zwd_2019_us_australia.csv.gz  # 21.5 MB，逐行匹配与 QC 结果
├── station_summary.csv                      # 65 站汇总
└── preparation_report.json                  # 机器可读报告
```

复现命令：

```powershell
python .\第二阶段\prepare_2019_gnss_era5_independent.py `
  --data-root .\2019_US_Australia_IGS_ERA5 `
  --out-dir .\第二阶段\independent_2019_us_australia_era5_v1
```

代码与计划已推送至 GitHub `GNSS-PWV/GNSS_dachuang` 的提交 `d451060`。派生数据未上传 Git；若服务器需要复算，只需上传源数据目录（ERA5 约 0.215 GiB、GNSS 约 0.104 GiB）或直接上传该 21.5 MB 压缩产物。

## 6. 二阶段正式训练前仍缺少的条件

1. 合规的因果一阶段 profile（动态列仅 `ELV/TS/PS/WPS`，且最大观测时刻严格早于预测时刻）；
2. 独立 PWV 标签及其来源、时空匹配和血缘信息；`ERA5 tcwv` 暂不能自动代替；
3. 老师确认 ZHD、站高与质量控制口径；
4. 将样本装配为 `phase2_independent_contract_schema.json` 要求的 contract，并让 `preflight_independent_phase2_inputs.py` 全量通过；
5. 之后才可提交 GPU 正式重训。
