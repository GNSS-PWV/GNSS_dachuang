# -*- coding: utf-8 -*-
"""Build experimental phase-2 profiles from validated IGRA ZIP archives.

This is deliberately separate from the official training data path.  It
requires a station metadata CSV with station_id, lat, lon and elev_m, parses
only CRC-valid archives, and writes reconstructed ``*_met.txt`` files into a
dedicated output directory.  The generated PWV/ZWD labels are experimental and
must not silently replace teacher-provided labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
import zipfile
from pathlib import Path

from igra_public_profile import parse_zip
from igra_reconstruct_profile import convert


def _field(row: dict[str, str], *names: str) -> str:
    lowered = {str(k).strip().lower(): str(v).strip() for k, v in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value not in (None, ""):
            return value
    return ""


def load_metadata(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            station = _field(row, "station_id", "station", "id")
            if not station:
                continue
            try:
                result[station] = {
                    "lat": float(_field(row, "lat", "latitude")),
                    "lon": float(_field(row, "lon", "longitude")),
                    "elev_m": float(_field(row, "elev_m", "elevation", "height") or 0.0),
                }
            except ValueError:
                continue
    return result


def load_trusted_stations(path: Path | None) -> set[str]:
    if path is None:
        return set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["station_id"]
            for row in csv.DictReader(handle)
            if row.get("state", "").endswith("_crc_valid")
        }


def valid_zip(path: Path) -> tuple[bool, str]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".txt")]
            if len(members) != 1:
                return False, f"member_count={len(members)}"
            bad = archive.testzip()
            return (True, "") if bad is None else (False, f"crc_failed={bad}")
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        return False, f"{type(exc).__name__}:{exc}"


def build(args: argparse.Namespace) -> dict[str, object]:
    metadata = load_metadata(args.metadata)
    trusted = load_trusted_stations(args.trusted_audit)
    zips = sorted(args.zip_dir.glob("*-drvd.txt.zip"))
    if args.max_stations is not None:
        zips = zips[: args.max_stations]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, zip_path in enumerate(zips, start=1):
            station = zip_path.name.split("-drvd", 1)[0]
            print(f"[{index}/{len(zips)}] {station}", flush=True)
            if station not in metadata:
                rows.append({"station_id": station, "state": "missing_metadata"})
                continue
            ok, detail = (True, "trusted_audit") if station in trusted else valid_zip(zip_path)
            if not ok:
                rows.append({"station_id": station, "state": "invalid_zip", "detail": detail})
                continue
            if args.dry_run:
                rows.append({"station_id": station, "state": "ready"})
                continue
            output_txt = args.output_dir / f"{station}_met.txt"
            if args.resume and output_txt.exists() and output_txt.stat().st_size > 0:
                rows.append({"station_id": station, "state": "existing_output", "output_txt": str(output_txt)})
                continue
            try:
                with tempfile.TemporaryDirectory(prefix=f"{station}_", dir=args.temp_dir) as station_temp:
                    levels_csv = Path(station_temp) / f"{station}_levels.csv"
                    parsed = parse_zip(zip_path, levels_csv)
                    converted = convert(
                        levels_csv,
                        output_txt,
                        lat=metadata[station]["lat"],
                        lon=metadata[station]["lon"],
                        station_elv=metadata[station]["elev_m"],
                        years=set(args.years),
                    )
                    rows.append({"station_id": station, "state": "converted", **parsed, **converted})
            except Exception as exc:  # retain station-level diagnostics
                rows.append({"station_id": station, "state": "conversion_error", "detail": f"{type(exc).__name__}:{exc}"})
    summary = {
        "policy": "experimental_IGRA_hydrostatic_reconstruction; not_official_training_labels",
        "years": args.years,
        "zip_dir": str(args.zip_dir),
        "output_dir": str(args.output_dir),
        "station_count": len(zips),
        "states": {},
        "rows": rows,
    }
    from collections import Counter
    summary["states"] = dict(Counter(str(row["state"]) for row in rows))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--temp-dir", type=Path, default=None)
    parser.add_argument("--years", type=int, nargs="+", default=[2014, 2015, 2016, 2017, 2018, 2019])
    parser.add_argument("--max-stations", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="skip existing non-empty *_met.txt outputs")
    parser.add_argument("--trusted-audit", type=Path, default=None, help="CSV whose *_crc_valid rows may skip repeat CRC scans")
    args = parser.parse_args()
    if args.temp_dir is not None:
        args.temp_dir.mkdir(parents=True, exist_ok=True)
    summary = build(args)
    print(json.dumps({"station_count": summary["station_count"], "states": summary["states"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
