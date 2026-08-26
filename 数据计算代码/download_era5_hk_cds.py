"""
Download + regionally crop ERA5 for the GNSS-WRF-HK1K project (2020-2024)
via the Copernicus CDS API, using MULTIPLE CDS accounts in parallel.

Why this approach:
  The NSF NCAR S3 bucket only serves GLOBAL files (~1.3 GB / pl-var / day);
  on this machine the link to S3 ran at ~0.3 MB/s, so 5 years (~16 TB) was
  infeasible. CDS crops to the Hong Kong box server-side (a few MB / month),
  but a single CDS account gets rate-limited / 403 under concurrency. So we
  spread monthly requests across the 11 accounts in ERA5下载key.txt, each in
  its own process with its own temporary ~/.cdsapirc.

Accounts file (ERA5下载key.txt) format, repeated per account:
  <name>
  url: https://cds.climate.copernicus.eu/api
  key: <uuid>

Output : F:/era5_hk_cds/  (one NetCDF per group per MONTH)
           pl/<YYYY>/era5_pl_<YYYY><MM>.nc      pressure levels, 37 lvl, hourly
           sfc/<YYYY>/era5_sfc_<YYYY><MM>.nc     surface analysis, hourly
           flux/<YYYY>/era5_flux_<YYYY><MM>.nc   precip & flux rates, hourly
           invariant/era5_invariant.nc           lsm + surface geopotential

Region : plan section 6 -> CDS area = [North, West, South, East]
           [26.5, 108.0, 18.0, 119.0]
Time   : 2020-01 .. 2024-12, hourly (00..23 UTC).

Usage:
  python download_era5_hk_cds.py                       # full 2020-2024, all groups
  python download_era5_hk_cds.py --years 2020          # one year
  python download_era5_hk_cds.py --years 2020 --months 1 --groups pl
  python download_era5_hk_cds.py --test                # 2020-01, all groups
  python download_era5_hk_cds.py --workers 11          # parallelism (default=#accounts)
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from datetime import datetime
from multiprocessing import Manager, Pool
from pathlib import Path
from typing import Dict, List, Tuple


warnings.filterwarnings("ignore")

# ---- Region & time -------------------------------------------------------
AREA = [26.5, 108.0, 18.0, 119.0]            # [North, West, South, East]
START_YEAR, END_YEAR = 2020, 2024

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_KEY_FILE = SCRIPT_DIR / "ERA5下载key.txt"
OUT_ROOT = Path("data/raw/era5_targeted_cds")

ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]
ALL_TIMES = [f"{h:02d}:00" for h in range(24)]
ALL_LEVELS = [
    "1", "2", "3", "5", "7", "10", "20", "30", "50", "70",
    "100", "125", "150", "175", "200", "225", "250", "300", "350", "400",
    "450", "500", "550", "600", "650", "700", "750", "775", "800", "825",
    "850", "875", "900", "925", "950", "975", "1000",
]  # all 37 ERA5 pressure levels

# ---- Variables (CDS long names) ------------------------------------------
PL_VARS = [
    "geopotential", "temperature",
    "u_component_of_wind", "v_component_of_wind",
    "specific_humidity", "relative_humidity", "vertical_velocity",
]
SFC_VARS = [
    "surface_pressure", "mean_sea_level_pressure",
    "2m_temperature", "2m_dewpoint_temperature",
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "skin_temperature", "sea_surface_temperature",
    "total_column_water_vapour", "boundary_layer_height",
    "convective_available_potential_energy", "convective_inhibition",
    "soil_temperature_level_1", "soil_temperature_level_2",
    "soil_temperature_level_3", "soil_temperature_level_4",
    "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3", "volumetric_soil_water_layer_4",
    "snow_depth", "sea_ice_cover",
]
FLUX_VARS = [
    "mean_total_precipitation_rate",
    "mean_convective_precipitation_rate",
    "mean_large_scale_precipitation_rate",
    "mean_surface_latent_heat_flux",
    "mean_surface_sensible_heat_flux",
    "mean_surface_net_short_wave_radiation_flux",
    "mean_surface_net_long_wave_radiation_flux",
]
INV_VARS = ["land_sea_mask", "geopotential"]

GROUPS = ["pl", "sfc", "flux", "inv"]
PL_DATASET = "reanalysis-era5-pressure-levels"
SL_DATASET = "reanalysis-era5-single-levels"


# ═══════════════════════════════════════════════════════════════════════
#  Account parsing (same format as download_era5_parallel.py)
# ═══════════════════════════════════════════════════════════════════════

def parse_api_keys(key_file: str) -> List[Dict[str, str]]:
    """Parse the accounts file -> [{'name','url','key'}, ...]."""
    accounts = []
    with open(key_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current: Dict[str, str] = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("邮箱") or line.startswith("密码"):
            continue
        if line.startswith("url:") or line.startswith("url："):
            rest = line.split(":", 1)[1].strip()
            current["url"] = ("https:" + rest) if rest.startswith("//") else rest
        elif line.startswith("key:") or line.startswith("key："):
            current["key"] = line.split(":", 1)[1].strip()
        else:
            if current and "key" in current:
                accounts.append(current.copy())
                current = {}
            current["name"] = line
    if current and "key" in current:
        accounts.append(current)

    for i, a in enumerate(accounts):
        a.setdefault("url", "https://cds.climate.copernicus.eu/api")
        a.setdefault("name", f"Account_{i+1}")
    return accounts


# ═══════════════════════════════════════════════════════════════════════
#  Job construction
# ═══════════════════════════════════════════════════════════════════════

def days_in_month(year: int, month: int) -> int:
    import calendar
    return calendar.monthrange(year, month)[1]


def base_request(year: int, month: int) -> dict:
    return {
        "product_type": "reanalysis",
        "year": f"{year:04d}",
        "month": f"{month:02d}",
        "day": ALL_DAYS,          # CDS ignores non-existent days (e.g. Feb 30)
        "time": ALL_TIMES,
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def build_jobs(groups: List[str], years: List[int],
               months: List[int], days: List[int] | None = None) -> List[dict]:
    """Each job: {dataset, req, dest, label}."""
    jobs: List[dict] = []

    if "inv" in groups:
        dest = OUT_ROOT / "invariant" / "era5_invariant.nc"
        if not dest.exists():
            req = {
                "product_type": "reanalysis", "variable": INV_VARS,
                "year": "2020", "month": "01", "day": "01", "time": "00:00",
                "area": AREA, "data_format": "netcdf",
                "download_format": "unarchived",
            }
            jobs.append({"dataset": SL_DATASET, "req": req, "dest": dest,
                         "label": "inv"})

    for year in years:
        for month in months:
            if "pl" in groups:
                # ERA5 pressure-levels: CDS rejects (403) requests larger than
                # ~1 day (7 vars x 37 levels x 24 h). So split pl by DAY.
                dest_dir = OUT_ROOT / "pl" / f"{year:04d}" / f"{month:02d}"
                candidate_days = range(1, days_in_month(year, month) + 1)
                if days is not None:
                    candidate_days = [day for day in days
                                      if 1 <= day <= days_in_month(year, month)]
                for day in candidate_days:
                    dest = dest_dir / f"era5_pl_{year:04d}{month:02d}{day:02d}.nc"
                    if not dest.exists():
                        req = {
                            "product_type": "reanalysis",
                            "variable": PL_VARS,
                            "pressure_level": ALL_LEVELS,
                            "year": f"{year:04d}", "month": f"{month:02d}",
                            "day": f"{day:02d}", "time": ALL_TIMES,
                            "area": AREA, "data_format": "netcdf",
                            "download_format": "unarchived",
                        }
                        jobs.append({"dataset": PL_DATASET, "req": req, "dest": dest,
                                     "label": f"pl/{year}{month:02d}{day:02d}"})
            if "sfc" in groups:
                dest = OUT_ROOT / "sfc" / f"{year:04d}" / f"era5_sfc_{year:04d}{month:02d}.nc"
                if not dest.exists():
                    req = base_request(year, month)
                    req["variable"] = SFC_VARS
                    jobs.append({"dataset": SL_DATASET, "req": req, "dest": dest,
                                 "label": f"sfc/{year}{month:02d}"})
            if "flux" in groups:
                dest = OUT_ROOT / "flux" / f"{year:04d}" / f"era5_flux_{year:04d}{month:02d}.nc"
                if not dest.exists():
                    req = base_request(year, month)
                    req["variable"] = FLUX_VARS
                    jobs.append({"dataset": SL_DATASET, "req": req, "dest": dest,
                                 "label": f"flux/{year}{month:02d}"})
    return jobs


# ═══════════════════════════════════════════════════════════════════════
#  Worker (one process, one account)
# ═══════════════════════════════════════════════════════════════════════

def download_one(args: Tuple) -> Tuple[str, bool, str]:
    import cdsapi

    """Run one CDS request via submit + poll, with a per-request timeout so a
    request stuck in the CDS queue can't block the worker forever."""
    job, account, progress, lock, total = args
    label = job["label"]
    dest: Path = job["dest"]
    name = account.get("name", "Unknown")

    if dest.exists():
        with lock:
            progress["completed"] += 1
            progress["skipped"] += 1
        return (label, True, "skip (exists)")

    temp_rc = Path.home() / f".cdsapirc_{os.getpid()}"
    tmp = dest.with_suffix(".tmp.nc")
    max_attempts = 4
    job_timeout = 900      # seconds to wait for one request before abandoning
    poll_every = 15
    last_err = ""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_rc, "w") as f:
            f.write(f"url: {account['url']}\n")
            f.write(f"key: {account['key']}\n")
        os.environ["CDSAPI_RC"] = str(temp_rc)

        for attempt in range(1, max_attempts + 1):
            remote = None
            try:
                # submit without blocking; low retry_max so client errors surface
                client = cdsapi.Client(quiet=True, wait_until_complete=False,
                                       retry_max=2, timeout=60, delete=True)
                remote = client.retrieve(job["dataset"], job["req"])

                waited = 0
                consec_poll_err = 0
                while True:
                    try:
                        status = str(remote.status).lower()
                        consec_poll_err = 0
                    except Exception as pe:
                        # transient status-check error: tolerate a few, then bail
                        consec_poll_err += 1
                        if consec_poll_err >= 5:
                            raise RuntimeError(f"status poll failed: {pe}")
                        status = "accepted"
                    if status in ("successful", "completed"):
                        break
                    if status in ("failed", "rejected", "deleted", "dismissed"):
                        raise RuntimeError(f"request status={status}")
                    if waited >= job_timeout:
                        raise TimeoutError(f"queue timeout after {waited}s (status={status})")
                    time.sleep(poll_every)
                    waited += poll_every

                remote.download(str(tmp))
                tmp.replace(dest)
                with lock:
                    progress["completed"] += 1
                    progress["downloaded"] += 1
                mb = dest.stat().st_size / 1e6
                note = "" if attempt == 1 else f" (attempt {attempt})"
                return (label, True, f"OK [{name}] {mb:.1f} MB{note}")
            except Exception as e:
                last_err = str(e)[:120]
                if tmp.exists():
                    tmp.unlink()
                # abandon the server-side request so it doesn't pile up
                if remote is not None:
                    try:
                        remote.delete()
                    except Exception:
                        pass
                if attempt < max_attempts:
                    time.sleep(min(60, 10 * attempt))

        with lock:
            progress["completed"] += 1
        return (label, False, f"FAIL [{name}] after {max_attempts} tries: {last_err}")
    finally:
        if temp_rc.exists():
            temp_rc.unlink()



# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="ERA5 HK download via multi-account CDS")
    ap.add_argument("--years", nargs="*", type=int)
    ap.add_argument("--months", nargs="*", type=int)
    ap.add_argument("--days", nargs="*", type=int,
                    help="limit pressure-level jobs to selected days (e.g. --days 1)")
    ap.add_argument("--groups", nargs="*", choices=GROUPS)
    ap.add_argument("--test", action="store_true", help="2020-01 only, all groups")
    ap.add_argument("--key-file", type=str, default=str(DEFAULT_KEY_FILE),
                    help="CDS account file; never print its contents")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="override output root; defaults to data/raw/era5_targeted_cds")
    ap.add_argument("--area", nargs=4, type=float, metavar=("N", "W", "S", "E"),
                    help="regional crop [North West South East]")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel processes (default = number of accounts)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and print jobs without reading credentials or contacting CDS")
    args = ap.parse_args()

    global OUT_ROOT, AREA
    if args.output_dir:
        OUT_ROOT = Path(args.output_dir)
    if args.area:
        AREA = args.area

    if args.test:
        years, months = [2020], [1]
    else:
        years = args.years or list(range(START_YEAR, END_YEAR + 1))
        months = args.months or list(range(1, 13))
    groups = args.groups or list(GROUPS)
    days = args.days or None

    jobs = build_jobs(groups, years, months, days)
    if args.dry_run:
        print("ERA5 CDS dry-run (credentials were not read)")
        print(f"  area [N,W,S,E]: {AREA}")
        print(f"  years={years}  months={months}  days={days}  groups={groups}")
        print(f"  output: {OUT_ROOT}")
        print(f"  jobs: {len(jobs)}")
        for job in jobs[:10]:
            print(f"  - {job['label']} -> {job['dest']}")
        if len(jobs) > 10:
            print(f"  ... {len(jobs)-10} more jobs")
        return

    accounts = parse_api_keys(args.key_file)
    if not accounts:
        print("[ERROR] no valid CDS accounts found.")
        return
    if not jobs:
        print("Nothing to do (all targets already exist?).")
        return

    workers = args.workers or len(accounts)
    workers = min(workers, len(accounts), len(jobs))

    print("=" * 70)
    print("ERA5 HK download via CDS (multi-account, server-side crop)")
    print(f"  area [N,W,S,E]: {AREA}")
    print(f"  years={years}  months={months}  days={days}  groups={groups}")
    print(f"  accounts: {len(accounts)}  workers: {workers}")
    print(f"  monthly requests to submit: {len(jobs)}")
    print(f"  output: {OUT_ROOT}")
    print("=" * 70, flush=True)

    manager = Manager()
    progress = manager.dict({"completed": 0, "downloaded": 0, "skipped": 0})
    lock = manager.Lock()

    # round-robin accounts across jobs
    tasks = [(job, accounts[i % len(accounts)], progress, lock, len(jobs))
             for i, job in enumerate(jobs)]

    t0 = time.time()
    results = []
    with Pool(processes=workers) as pool:
        for res in pool.imap_unordered(download_one, tasks):
            results.append(res)
            label, ok, msg = res
            status = "OK" if ok else "XX"
            print(f"  [{progress['completed']}/{len(jobs)}] {status} {label}: {msg}",
                  flush=True)

    ok = sum(1 for _, s, _ in results if s)
    fail = len(results) - ok
    print("\n" + "=" * 70)
    print(f"DONE: {ok}/{len(jobs)} ok "
          f"(downloaded {progress['downloaded']}, skipped {progress['skipped']}, "
          f"failed {fail}) in {(time.time()-t0)/60:.1f} min")
    if fail:
        print("Failed:")
        for lab, s, m in results:
            if not s:
                print(f"  {lab}: {m}")
    print("=" * 70)


if __name__ == "__main__":
    main()
