# -*- coding: utf-8 -*-
"""Download and parse public NOAA/NCEI IGRA 2.2 derived soundings.

This tool is intentionally conservative:
* downloads only public NCEI files; no UCAR/EarthScope credentials are read;
* supports curl resume (``-C -``) because the station archives are large;
* converts fixed-width derived levels into a profile intermediate CSV;
* does not invent ZWD/PWV labels.  IGRA's header PW is only surface--500 hPa
  precipitable water, so a later integration/label step is still required.

Example:
    python igra_public_profile.py download --station AEM00041217
    python igra_public_profile.py parse --zip-dir public_igra --station AEM00041217
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

BASE_URL = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/derived-por"
FIELD_SLICES = {
    "pressure_raw": (0, 7),
    "rep_gph": (8, 15),
    "calc_gph": (16, 23),
    "temp_raw": (24, 31),
    "vapor_pressure_raw": (72, 79),
}


def download_station(station: str, output_dir: str | Path, timeout: int = 1800) -> Path:
    if not re.fullmatch(r"[A-Z0-9]{11}", station):
        raise ValueError(f"invalid IGRA station id: {station!r}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{station}-drvd.txt.zip"
    url = f"{BASE_URL}/{station}-drvd.txt.zip"
    cmd = [
        "curl.exe", "--fail", "--location", "--retry", "2",
        "--connect-timeout", "20", "--max-time", str(timeout),
        "-C", "-", "--output", str(dest), url,
    ]
    print("downloading:", url, flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"curl failed with exit code {result.returncode}: {dest}")
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"download produced no data: {dest}")
    return dest


def _int(line: str, name: str) -> int | None:
    lo, hi = FIELD_SLICES[name]
    value = line[lo:hi].strip()
    if not value:
        return None
    n = int(value)
    return None if n == -99999 else n


def parse_derived_file(path: str | Path, station_id: str | None = None) -> list[dict[str, object]]:
    """Parse one IGRA ``-drvd.txt`` file into level records.

    Scaling follows ``igra2-derived-format.txt``: pressure is mb*100,
    temperature is K*10, vapor pressure is mb*1000, and geopotential height
    is metres. Header ``PW`` is mm*100 surface--500 hPa.
    """
    rows: list[dict[str, object]] = []
    station = station_id or Path(path).name.split("-drvd", 1)[0]
    with Path(path).open("r", encoding="ascii", errors="replace") as fh:
        lines = iter(fh)
        for line in lines:
            if not line.startswith("#"):
                continue
            try:
                year = int(line[13:17]); month = int(line[18:20]); day = int(line[21:23])
                hour = int(line[24:26]); nlev = int(line[31:36]); pw_raw = int(line[37:43])
            except ValueError:
                continue
            if hour == 99 or nlev <= 1:
                for _ in range(max(nlev, 0)):
                    next(lines, "")
                continue
            timestamp = datetime(year, month, day, hour).isoformat(sep=" ")
            for level_index in range(nlev):
                record = next(lines, "")
                if len(record) < 79:
                    continue
                pressure = _int(record, "pressure_raw")
                height = _int(record, "rep_gph")
                temp = _int(record, "temp_raw")
                vapor = _int(record, "vapor_pressure_raw")
                if None in (pressure, height, temp, vapor):
                    continue
                rows.append({
                    "station_id": station,
                    "TIME": timestamp,
                    "level_index": level_index,
                    "PRESSURE_HPA": pressure / 100.0,
                    "ELV_M": height,
                    "TEMP_K": temp / 10.0,
                    "VAPOR_PRESSURE_HPA": vapor / 1000.0,
                    "PW_SURFACE_500HPA_MM": None if pw_raw == -99999 else pw_raw / 100.0,
                })
    return rows


def parse_zip(zip_path: str | Path, output_csv: str | Path) -> dict[str, object]:
    zip_path = Path(zip_path)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".txt")]
        if len(members) != 1:
            raise ValueError(f"expected one TXT member, found {members}")
        temp = output_csv.with_suffix(".source.txt")
        # Stream large US station members instead of materializing the whole
        # uncompressed text file in RAM.
        with zf.open(members[0], "r") as source, temp.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
    station = zip_path.name.split("-drvd", 1)[0]
    rows = parse_derived_file(temp, station_id=station)
    temp.unlink(missing_ok=True)
    out = output_csv
    fields = ["station_id", "TIME", "level_index", "PRESSURE_HPA", "ELV_M", "TEMP_K", "VAPOR_PRESSURE_HPA", "PW_SURFACE_500HPA_MM"]
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary = {
        "source_zip": str(zip_path), "output_csv": str(out), "level_rows": len(rows),
        "profile_timestamps": len({str(r["TIME"]) for r in rows}),
        "label_status": "PW only surface-to-500-hPa; ZWD/PWV not generated",
    }
    out.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download"); dl.add_argument("--station", required=True); dl.add_argument("--output-dir", default="public_igra"); dl.add_argument("--timeout", type=int, default=1800)
    ps = sub.add_parser("parse"); ps.add_argument("--zip-dir", required=True); ps.add_argument("--station", required=True); ps.add_argument("--output-dir", default="public_igra_profile")
    args = parser.parse_args()
    if args.command == "download":
        print(download_station(args.station, args.output_dir, args.timeout))
    else:
        zip_path = Path(args.zip_dir) / f"{args.station}-drvd.txt.zip"
        print(json.dumps(parse_zip(zip_path, Path(args.output_dir) / f"{args.station}_levels.csv"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
