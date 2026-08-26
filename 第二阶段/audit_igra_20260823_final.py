# -*- coding: utf-8 -*-
"""Audit the merged IGRA archive set using the 2026-08-23 manifest.

The manifest is used for station presence and provenance only.  Archive size
is reported but deliberately does not decide validity because NCEI may update
the same named archive after the pinned manifest date.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter
from pathlib import Path


STATION_RE = re.compile(r"^([A-Z0-9]{11})-drvd\.txt\.zip$")
ROW_RE = re.compile(
    r'(?s)<tr>.*?href="([A-Z0-9]{11})-drvd\.txt\.zip"'
    r'.*?<td align="right">(\d+)</td>.*?</tr>'
)


def load_manifest(path: Path) -> dict[str, int]:
    return {m.group(1): int(m.group(2)) for m in ROW_RE.finditer(path.read_text(encoding="utf-8"))}


def load_targets(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sorted({
            row["station_id"]
            for row in csv.DictReader(handle)
            if 2014 <= int(row["year"]) <= 2018
        })


def zip_status(path: Path | None) -> tuple[str, str]:
    if path is None or not path.exists():
        return "missing", ""
    try:
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".txt")]
            if len(members) != 1:
                return "bad_member_count", str(len(members))
            bad = archive.testzip()
            return ("crc_valid", "") if bad is None else ("crc_failed", bad)
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        return "invalid_zip", f"{type(exc).__name__}:{exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--manual-dir", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    targets = load_targets(args.assignments)
    rows: list[dict[str, object]] = []
    for station in targets:
        expected = manifest.get(station)
        manual = args.manual_dir / f"{station}-drvd.txt.zip"
        batch = args.batch_dir / f"{station}-drvd.txt.zip"
        selected = None
        source = ""
        state = "no_igra_match" if expected is None else "missing"
        detail = ""
        for candidate, candidate_source in ((manual, "manual"), (batch, "batch")):
            candidate_state, candidate_detail = zip_status(candidate)
            if candidate_state == "crc_valid":
                selected = candidate
                source = candidate_source
                state = f"{candidate_source}_crc_valid"
                detail = ""
                break
            if candidate.exists() and state != "no_igra_match":
                state = candidate_state
                detail = candidate_detail
                source = candidate_source
        actual = selected.stat().st_size if selected else 0
        rows.append({
            "station_id": station,
            "state": state,
            "source": source,
            "expected_bytes_20260823": expected or "",
            "actual_bytes": actual,
            "size_diff_bytes": (actual - expected) if expected and actual else "",
            "selected_path": str(selected) if selected else "",
            "detail": detail,
            "url": (f"https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/"
                     f"access/derived-por/{station}-drvd.txt.zip") if expected else "",
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "manifest": str(args.manifest),
        "policy": "pinned_manifest_2026-08-23; ignore later NCEI size changes; require CRC",
        "target_stations": len(targets),
        "states": dict(Counter(row["state"] for row in rows)),
        "rows": rows,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with args.output.with_suffix(".txt").open("w", encoding="utf-8") as handle:
        handle.write("IGRA 2026-08-23 pinned-version final audit\n")
        handle.write("Policy: ignore later NCEI size changes; require station presence and ZIP CRC.\n\n")
        for row in rows:
            if row["state"] not in ("manual_crc_valid", "batch_crc_valid"):
                handle.write(f"{row['station_id']}\t{row['state']}\t{row['detail']}\t{row['url']}\n")
    print(json.dumps({"target_stations": len(targets), **summary["states"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
