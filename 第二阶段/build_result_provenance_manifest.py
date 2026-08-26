"""Build a SHA256 provenance manifest for the frozen 2014-2019 strict-result bundle.

This utility intentionally enumerates only public project scripts/results required to
reproduce the current tables and figures. It never searches for or includes .env,
credentials, tokens, account files, or download keys.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
PHASE2 = ROOT / "第二阶段"
ANALYSIS = PHASE2 / "result_p1deploy_ft_strict_20260817" / "analysis_2014_2019"
FIGURES = ANALYSIS / "paper_ppt_figures_20260817"
DEFAULT_JSON = FIGURES / "result_provenance_manifest_20260817.json"
DEFAULT_MD = FIGURES / "结果复现与溯源清单_20260817.md"

# Deliberately explicit allowlist: do not glob the workspace, because that could
# inadvertently include credentials or unrelated private files.
ARTIFACTS = [
    PHASE2 / "audit_strict_result_consistency.py",
    PHASE2 / "analyze_weak_station_leave_one_year_out.py",
    PHASE2 / "generate_paper_ppt_figures.py",
    PHASE2 / "build_result_provenance_manifest.py",
    PHASE2 / "2014_2019严格回放式堆叠部署评估_成果汇总_20260817.md",
    ROOT / "论文PPT_当前权威结果口径卡_20260817.md",
    ANALYSIS / "metrics_official36_2014_2019.csv",
    ANALYSIS / "annual_metrics_official36_2014_2019.csv",
    ANALYSIS / "strict_result_consistency_audit_reproducible_20260817.json",
    ANALYSIS / "严格结果一致性复核_20260817.md",
    ANALYSIS / "station_robustness_20260817" / "station_vs_gpt3_summary.csv",
    ANALYSIS / "station_robustness_20260817" / "annual_station_coverage_strict.csv",
    ANALYSIS / "station_robustness_20260817" / "weak_station_leave_one_year_out_summary_20260817.csv",
    ANALYSIS / "station_robustness_20260817" / "弱站逐年留一稳健性复核_20260817.md",
    FIGURES / "figure_generation_report_20260817.json",
    FIGURES / "论文PPT图表与图注_20260817.md",
    FIGURES / "01_global_rmse_comparison_2014_2019.png",
    FIGURES / "01_global_rmse_comparison_2014_2019.pdf",
    FIGURES / "01_global_rmse_comparison_2014_2019.svg",
    FIGURES / "02_annual_rmse_2014_2019.png",
    FIGURES / "02_annual_rmse_2014_2019.pdf",
    FIGURES / "02_annual_rmse_2014_2019.svg",
    FIGURES / "03_station_rmse_reduction_vs_gpt3_2014_2019.png",
    FIGURES / "03_station_rmse_reduction_vs_gpt3_2014_2019.pdf",
    FIGURES / "03_station_rmse_reduction_vs_gpt3_2014_2019.svg",
    FIGURES / "04_yearly_coverage_2014_2019.png",
    FIGURES / "04_yearly_coverage_2014_2019.pdf",
    FIGURES / "04_yearly_coverage_2014_2019.svg",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def artifact_record(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required provenance artifact is missing: {path}")
    stat = path.stat()
    return {
        "path": relative(path),
        "sha256": sha256(path),
        "bytes": stat.st_size,
        "modified_local": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
    }


def verify_no_sensitive_paths(records: Iterable[dict[str, object]]) -> None:
    prohibited = (".env", "credential", "credentials", "secret", "token", "password", "key")
    hits = [str(record["path"]) for record in records if any(word in str(record["path"]).lower() for word in prohibited)]
    if hits:
        raise ValueError(f"Sensitive-looking artifact path was unexpectedly included: {hits}")


def markdown(manifest: dict[str, object]) -> str:
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, list)
    rows = "\n".join(
        f"| `{item['path']}` | `{item['sha256']}` | {item['bytes']:,} | {item['modified_local']} |"
        for item in artifacts
    )
    return f"""# 结果复现与文件溯源清单（2026-08-17）

## 适用范围

本清单仅覆盖当前已经冻结的 **2014–2019 严格历史回放式堆叠部署评估** 与由其 CSV 自动生成的论文/PPT 图表。当前正式结果为 110,928 个共同样本、36 个站跨年度联合覆盖（单年度实际有样本站点为 27–30）。它不需要下载新的 GNSS 或 ERA5 数据。

## 正确使用边界

- `clim_surf_p1` 是当前部署候选；`real` 与 `real_surf_p1` 是 oracle 分析，不能被当作部署精度。
- 该清单支持核对现有脚本、冻结结果表、图表和说明文件是否变更；它不能证明上游训练数据、外部原始观测、未来数据或外部下载源的完整性。
- 本清单不枚举、不读取也不包含账号、密码、Token、认证文件、`.env` 或下载密钥。
- 结果边界以 `论文PPT_当前权威结果口径卡_20260817.md` 为准：历史回放式评估不等同于完全自治实时部署或训练截止年后的未来独立验证。

## 最小复现步骤

```powershell
python -m py_compile 第二阶段\\generate_paper_ppt_figures.py 第二阶段\\build_result_provenance_manifest.py
python 第二阶段\\generate_paper_ppt_figures.py
python 第二阶段\\build_result_provenance_manifest.py
```

执行后，将本清单 JSON 的 SHA256 与下表逐项比较；若脚本、CSV、图表或说明有合理更新，应重新生成清单，而不是继续沿用旧哈希。

## 文件哈希

| 相对路径 | SHA256 | 字节数 | 本地修改时间 |
|---|---|---:|---|
{rows}

## 清单元数据

```json
{json.dumps({key: value for key, value in manifest.items() if key != 'artifacts'}, ensure_ascii=False, indent=2)}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="Output JSON manifest path")
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD, help="Output Markdown checklist path")
    args = parser.parse_args()
    json_path = args.json.resolve()
    markdown_path = args.markdown.resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    records = [artifact_record(path) for path in ARTIFACTS]
    verify_no_sensitive_paths(records)
    manifest: dict[str, object] = {
        "status": "PASS",
        "manifest_date": "2026-08-17",
        "scope": "Frozen 2014-2019 strict historical replay result and derived paper/PPT figures",
        "workspace_root": str(ROOT),
        "artifact_count": len(records),
        "sensitive_file_policy": "Explicit artifact allowlist; authentication, credential, token, key, password, and .env paths are excluded.",
        "limitations": [
            "Does not prove integrity or availability of upstream training data, external raw observations, future data, or external download sources.",
            "Does not turn historical replay evidence into autonomous real-time deployment or future independent validation evidence.",
            "Oracle analyses remain non-deployable reference analyses.",
        ],
        "artifacts": records,
    }
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown(manifest), encoding="utf-8")

    # Self-validation after write: prove JSON parseability and output non-emptiness.
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    if parsed.get("status") != "PASS" or len(parsed.get("artifacts", [])) != len(ARTIFACTS):
        raise RuntimeError("Written provenance JSON failed self-validation")
    if markdown_path.stat().st_size < 1024:
        raise RuntimeError("Written provenance Markdown is unexpectedly small")
    print(json.dumps({"status": "PASS", "artifacts": len(records), "json": str(json_path), "markdown": str(markdown_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
