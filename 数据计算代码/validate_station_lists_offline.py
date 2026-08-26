#!/usr/bin/env python3
"""离线审计 GNSS 站表，不联网、不下载、不读取认证材料，也不修改原始站表。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SOURCES = {
    "igs_whu": "IGSwhu_formatted.txt",
    "suominet": "suominet_formatted.txt",
    "australia": "austrilian_stations_formatted.txt",
    "euref": "euref_formatted.txt",
    "ucar_cosmic": "ucar_cosmic_formatted.txt",
}


@dataclass
class AuditResult:
    source: str
    file: str
    sha256: str | None = None
    total_nonempty_noncomment_lines: int = 0
    accepted_by_production_parser: int = 0
    malformed_lines: list[dict[str, Any]] = field(default_factory=list)
    coordinate_range_issues: list[dict[str, Any]] = field(default_factory=list)
    empty_station_ids: list[int] = field(default_factory=list)
    duplicate_station_ids: dict[str, list[int]] = field(default_factory=dict)
    status: str = "PASS"
    recommendation: str = "可用于下载前本地加载测试。"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def audit_station_file(source: str, path: Path) -> AuditResult:
    result = AuditResult(source=source, file=str(path))
    if not path.is_file():
        result.status = "FAIL"
        result.recommendation = "缺少下载器要求的站表文件；请先恢复文件接入。"
        return result

    result.sha256 = sha256_file(path)
    ids: dict[str, list[int]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", errors="strict") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text or text.startswith("#"):
                continue
            result.total_nonempty_noncomment_lines += 1
            parts = text.split(",")
            if len(parts) < 3:
                result.malformed_lines.append({"line": line_no, "reason": "列数少于 3", "text": text})
                continue
            try:
                station_id = parts[0].strip()
                lat = float(parts[1].strip())
                lon = float(parts[2].strip())
                height = float(parts[3].strip()) if len(parts) >= 4 else 0.0
            except (ValueError, IndexError) as exc:
                result.malformed_lines.append({"line": line_no, "reason": type(exc).__name__, "text": text})
                continue

            # 与生产解析器一致：浮点转换成功即会被加入；本工具额外报告地理语义风险。
            result.accepted_by_production_parser += 1
            if not station_id:
                result.empty_station_ids.append(line_no)
            else:
                ids[station_id].append(line_no)
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                result.coordinate_range_issues.append(
                    {"line": line_no, "station_id": station_id, "lat": lat, "lon": lon, "height": height}
                )

    result.duplicate_station_ids = {sid: lines for sid, lines in ids.items() if len(lines) > 1}
    issues = len(result.malformed_lines) + len(result.coordinate_range_issues) + len(result.empty_station_ids)
    if issues:
        result.status = "CONDITIONAL"
        result.recommendation = (
            "生产解析器可跳过格式错误行，但正式使用前应在不改动原始表的前提下，"
            "生成经人工确认的候选清洗表，并复核重复站点的保留规则。"
        )
    elif result.duplicate_station_ids:
        result.status = "WARN"
        result.recommendation = "可加载，但正式批量下载前需确认重复站点的去重口径。"
    return result


def build_report(station_dir: Path) -> dict[str, Any]:
    reports = [asdict(audit_station_file(source, station_dir / filename)) for source, filename in SOURCES.items()]
    counts = {"PASS": 0, "WARN": 0, "CONDITIONAL": 0, "FAIL": 0}
    for item in reports:
        counts[item["status"]] += 1
    return {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "mode": "offline_only_no_network_no_download_no_auth_read",
        "station_list_dir": str(station_dir.resolve()),
        "summary": counts,
        "sources": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="离线审计 GNSS formatted 站表")
    parser.add_argument("--station-list-dir", type=Path, default=Path("download_lables/GNSS_list"))
    parser.add_argument("--output", type=Path, help="可选 JSON 报告路径；不指定时仅打印到标准输出")
    args = parser.parse_args()

    report = build_report(args.station_list_dir)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 2 if report["summary"]["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
