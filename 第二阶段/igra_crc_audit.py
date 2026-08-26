# -*- coding: utf-8 -*-
"""Audit CRC integrity for the length-matched target archives."""
from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--status", type=Path, required=True)
    p.add_argument("--zip-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = []
    with args.status.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "length_match":
                continue
            path = args.zip_dir / f"{row['station_id']}-drvd.txt.zip"
            try:
                with zipfile.ZipFile(path) as archive:
                    error = archive.testzip()
                state = "valid" if error is None else "crc_failed"
            except Exception as exc:  # keep the audit going across all stations
                state = f"invalid:{type(exc).__name__}"
                error = str(exc)
            rows.append({
                "station_id": row["station_id"],
                "bytes": row["actual"],
                "crc_status": state,
                "error": "" if state == "valid" else str(error),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["station_id", "bytes", "crc_status", "error"])
        writer.writeheader()
        writer.writerows(rows)
    good = sum(row["crc_status"] == "valid" for row in rows)
    print({"audited": len(rows), "valid": good, "invalid": len(rows) - good})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
