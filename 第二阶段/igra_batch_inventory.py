# -*- coding: utf-8 -*-
"""Create a low-storage inventory for downloaded IGRA derived archives."""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path


def inventory_zip(path: Path, years: set[int]) -> dict[str, object]:
    station = path.name.split("-drvd", 1)[0]
    result: dict[str, object] = {
        "station_id": station,
        "zip_bytes": path.stat().st_size,
        "zip_valid": False,
        "profiles": 0,
        "level_rows": 0,
        "profiles_by_year": {str(y): 0 for y in sorted(years)},
        "levels_by_year": {str(y): 0 for y in sorted(years)},
        "error": "",
    }
    try:
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                result["error"] = f"crc_failed:{bad}"
                return result
            members = [name for name in archive.namelist() if name.lower().endswith(".txt")]
            if len(members) != 1:
                result["error"] = f"expected_one_txt:{len(members)}"
                return result
            with archive.open(members[0], "r") as raw:
                lines = iter(raw)
                for raw_line in lines:
                    if not raw_line.startswith(b"#"):
                        continue
                    try:
                        line = raw_line.decode("ascii", "replace")
                        year = int(line[13:17])
                        nlev = int(line[31:36])
                    except (ValueError, IndexError):
                        continue
                    for _ in range(max(nlev, 0)):
                        next(lines, b"")
                    if year not in years or nlev <= 1:
                        continue
                    result["profiles"] = int(result["profiles"]) + 1
                    result["level_rows"] = int(result["level_rows"]) + nlev
                    result["profiles_by_year"][str(year)] += 1
                    result["levels_by_year"][str(year)] += nlev
            result["zip_valid"] = True
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        result["error"] = f"{type(exc).__name__}:{exc}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--years", default="2014,2015,2016,2017,2018,2019")
    args = parser.parse_args()
    years = {int(item) for item in args.years.split(",") if item.strip()}
    rows = [inventory_zip(path, years) for path in sorted(args.zip_dir.glob("*-drvd.txt.zip"))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = Counter()
    for row in rows:
        summary["valid" if row["zip_valid"] else "invalid"] += 1
    csv_path = args.output.with_suffix(".csv")
    fields = ["station_id", "zip_bytes", "zip_valid", "profiles", "level_rows", "error"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    print(json.dumps({"archives": len(rows), **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
