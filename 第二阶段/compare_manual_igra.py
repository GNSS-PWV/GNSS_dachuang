# -*- coding: utf-8 -*-
"""Compare manually downloaded IGRA archives with the live manifest."""
from __future__ import annotations

import argparse
import csv
import re
import zipfile
from pathlib import Path


STATION_RE = re.compile(r"^([A-Z0-9]{11})-drvd\.txt\.zip(?:\.fdmdownload)?$")


def crc_status(path: Path) -> str:
    if path.name.endswith(".fdmdownload"):
        return "partial_extension"
    try:
        with zipfile.ZipFile(path) as archive:
            if len([n for n in archive.namelist() if n.lower().endswith(".txt")]) != 1:
                return "bad_member_count"
            return "valid" if archive.testzip() is None else "crc_failed"
    except (OSError, zipfile.BadZipFile, RuntimeError):
        return "bad_zip"


def load_manifest(path: Path) -> dict[str, int]:
    rows = {}
    for match in re.finditer(
        r'(?s)<tr>.*?href="([A-Z0-9]{11})-drvd\.txt\.zip".*?<td align="right">(\d+)</td>.*?</tr>',
        path.read_text(encoding="utf-8"),
    ):
        rows[match.group(1)] = int(match.group(2))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-dir", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected = load_manifest(args.manifest)
    manual = {}
    for path in args.manual_dir.rglob("*"):
        if not path.is_file():
            continue
        match = STATION_RE.match(path.name)
        if match:
            station = match.group(1)
            # Prefer a finished ZIP over a downloader's temporary file.
            if station not in manual or not path.name.endswith(".fdmdownload"):
                manual[station] = path

    rows = []
    for station, path in sorted(manual.items()):
        batch_path = args.batch_dir / f"{station}-drvd.txt.zip"
        expected_bytes = expected.get(station)
        status = crc_status(path)
        batch_status = "missing"
        if batch_path.exists() and expected_bytes is not None:
            batch_status = "length_match" if batch_path.stat().st_size == expected_bytes else "length_mismatch"
        rows.append({
            "station_id": station,
            "manual_path": str(path),
            "manual_bytes": path.stat().st_size,
            "expected_bytes": expected_bytes or "",
            "manual_length_match": bool(expected_bytes and path.stat().st_size == expected_bytes),
            "manual_crc_status": status,
            "batch_status": batch_status,
            "recommendation": (
                "copy_after_crc" if status == "valid" and expected_bytes and path.stat().st_size == expected_bytes
                else "finish_manual_download" if status == "partial_extension"
                else "redownload_or_recheck"
            ),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["station_id"]
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    for row in rows:
        counts[row["recommendation"]] = counts.get(row["recommendation"], 0) + 1
    print({"manual_station_files": len(rows), **counts})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
