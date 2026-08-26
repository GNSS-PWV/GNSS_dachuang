# -*- coding: utf-8 -*-
"""Build a current, auditable list of unresolved target archives."""
from __future__ import annotations

import argparse
import csv
import re
import zipfile
from pathlib import Path


def manifest(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    return {
        m.group(1): int(m.group(2))
        for m in re.finditer(
            r'(?s)<tr>.*?href="([A-Z0-9]{11})-drvd\.txt\.zip".*?<td align="right">(\d+)</td>.*?</tr>',
            text,
        )
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--assignments", type=Path, required=True)
    p.add_argument("--zip-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    expected = manifest(args.manifest)
    target = sorted({
        row["station_id"] for row in csv.DictReader(args.assignments.open(encoding="utf-8-sig"))
        if 2014 <= int(row["year"]) <= 2018
    })
    rows = []
    for station in target:
        exp = expected.get(station)
        path = args.zip_dir / f"{station}-drvd.txt.zip"
        actual = path.stat().st_size if path.exists() else 0
        if exp is None:
            state = "no_igra_match"
        elif actual == 0:
            state = "missing"
        elif actual != exp:
            state = "partial_or_length_mismatch"
        else:
            try:
                with zipfile.ZipFile(path) as archive:
                    state = "crc_valid" if archive.testzip() is None else "crc_failed"
            except Exception as exc:
                state = f"invalid_zip:{type(exc).__name__}"
        rows.append({
            "station_id": station,
            "expected_bytes": exp or "",
            "actual_bytes": actual,
            "size_mb": round((exp or 0) / 1048576, 1),
            "state": state,
            "url": "" if exp is None else f"https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/derived-por/{station}-drvd.txt.zip",
        })
    unresolved = [row for row in rows if row["state"] != "crc_valid"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(unresolved)
    text_path = args.output.with_suffix(".txt")
    with text_path.open("w", encoding="utf-8") as handle:
        for row in unresolved:
            handle.write(f"{row['station_id']}\t{row['size_mb']} MB\t{row['state']}\t{row['url']}\n")
    from collections import Counter
    print({"targets": len(rows), "resolved": len(rows) - len(unresolved), "unresolved": len(unresolved), **Counter(row["state"] for row in unresolved)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
