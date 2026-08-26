# -*- coding: utf-8 -*-
"""Align the public IGRA station list with the project's target stations."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_station_list(path: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    with path.open(encoding="ascii", errors="replace") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 4:
                continue
            station, lat, lon, elev = fields[:4]
            if len(station) != 11:
                continue
            try:
                values = {"lat": float(lat), "lon": float(lon), "elev_m": float(elev)}
            except ValueError:
                continue
            if values["lat"] <= -999 or values["lon"] <= -999:
                values = {"lat": "", "lon": "", "elev_m": ""}
            elif values["elev_m"] <= -999:
                values["elev_m"] = ""
            result[station] = values
    return result


def load_targets(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "station_id" in (reader.fieldnames or []):
            return sorted({row["station_id"] for row in reader if row.get("station_id")})
        raise ValueError(f"target file lacks station_id: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station-list", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stations = load_station_list(args.station_list)
    targets = load_targets(args.targets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["station_id", "lon", "lat", "elev_m", "source"]
    missing = 0
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for station in targets:
            row = stations.get(station)
            if row is None or row.get("lat", "") == "" or row.get("lon", "") == "":
                missing += 1
                writer.writerow({"station_id": station, "lon": "", "lat": "", "elev_m": "", "source": "missing"})
            else:
                source = "IGRA station list" if row.get("elev_m", "") != "" else "IGRA station list; elevation_missing"
                writer.writerow({"station_id": station, **row, "source": source})
    print({"target_stations": len(targets), "metadata_matches": len(targets) - missing, "metadata_missing": missing})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
