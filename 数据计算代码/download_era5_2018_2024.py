"""
Download ERA5 single-level data from NSF NCAR S3 bucket (2018-2024, global).

Variables:
  - 2m temperature (2t)
  - surface pressure (sp)
  - sea surface temperature (sstk)
  - total column water vapor (tcwv)
  - mean sea level pressure (msl)
  - 10m u/v wind (10u, 10v)
  - mean total precipitation rate (mtpr)
  - surface geopotential (z, invariant, downloaded once)

Time resolution: 1 hour
Spatial resolution: 0.25° × 0.25° (global, 1440 × 721 points)
Coverage: 2018-01-01 to 2024-12-31
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

# ---- Configuration --------------------------------------------------------
START_YEAR = 2018
END_YEAR = 2024

# Variable definitions: (token, product_prefix, time_invariant)
VARIABLES = [
    # Surface analysis (monthly files)
    ("128_167_2t", "e5.oper.an.sfc", False),
    ("128_134_sp", "e5.oper.an.sfc", False),
    ("128_034_sstk", "e5.oper.an.sfc", False),
    ("128_137_tcwv", "e5.oper.an.sfc", False),
    ("128_151_msl", "e5.oper.an.sfc", False),
    ("128_165_10u", "e5.oper.an.sfc", False),
    ("128_166_10v", "e5.oper.an.sfc", False),
    # Mean total precipitation rate (half-monthly files)
    ("235_055_mtpr", "e5.oper.fc.sfc.meanflux", False),
    # Surface geopotential (time-invariant, 1979-01 only)
    ("128_129_z", "e5.oper.invariant", True),
]

OUT_DIR = Path("/f/era5_single_level")
MAX_WORKERS = 4  # Parallel downloads; increase if bandwidth allows
# ---------------------------------------------------------------------------

BUCKET = "nsf-ncar-era5"


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download_file(s3, key: str, dest: Path, expected_size: int) -> tuple[bool, str]:
    """Download a single file with size verification."""
    if dest.exists():
        if dest.stat().st_size == expected_size:
            return True, f"skip (exists): {dest.name}"
        else:
            return False, f"size mismatch (re-downloading): {dest.name}"

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3.download_file(BUCKET, key, str(dest))
        actual = dest.stat().st_size
        if actual != expected_size:
            dest.unlink()
            return False, f"FAILED size check: {dest.name} (got {actual}, expected {expected_size})"
        return True, f"downloaded: {dest.name} ({human_size(expected_size)})"
    except Exception as e:
        if dest.exists():
            dest.unlink()
        return False, f"ERROR: {dest.name} - {e}"


def list_files_for_var(s3, token: str, prefix: str, year: int, month: int | None) -> list[tuple[str, int]]:
    """List S3 keys and sizes for a given variable/year/month."""
    if prefix == "e5.oper.invariant":
        # Invariant: only 1979-01
        s3_prefix = f"{prefix}/197901/"
    else:
        ym = f"{year:04d}{month:02d}" if month else f"{year:04d}"
        s3_prefix = f"{prefix}/{ym}/"

    try:
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=s3_prefix)
        if "Contents" not in resp:
            return []

        matches = []
        for obj in resp["Contents"]:
            key = obj["Key"]
            if f".{token}." in key:
                matches.append((key, obj["Size"]))
        return matches
    except Exception as e:
        print(f"ERROR listing {s3_prefix}: {e}", file=sys.stderr)
        return []


def main() -> None:
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    # Build download task list
    tasks = []

    for token, prefix, invariant in VARIABLES:
        var_name = token.split("_")[-1]

        if invariant:
            # Invariant: download once from 1979-01
            files = list_files_for_var(s3, token, prefix, 1979, 1)
            for key, size in files:
                dest = OUT_DIR / var_name / "invariant" / Path(key).name
                tasks.append((key, dest, size, var_name))
        else:
            # Time-varying: loop over years and months
            for year in range(START_YEAR, END_YEAR + 1):
                for month in range(1, 13):
                    files = list_files_for_var(s3, token, prefix, year, month)
                    for key, size in files:
                        dest = OUT_DIR / var_name / f"{year:04d}" / f"{month:02d}" / Path(key).name
                        tasks.append((key, dest, size, var_name))

    if not tasks:
        print("No files to download. Check variable tokens and date range.")
        return

    total_size = sum(t[2] for t in tasks)
    print(f"Found {len(tasks)} files, total size: {human_size(total_size)}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Parallel workers: {MAX_WORKERS}\n")

    # Download with progress tracking
    completed = 0
    failed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_file, s3, key, dest, size): (key, var_name)
            for key, dest, size, var_name in tasks
        }

        for future in as_completed(futures):
            key, var_name = futures[future]
            completed += 1
            success, msg = future.result()

            status = "✓" if success else "✗"
            print(f"[{completed}/{len(tasks)}] {status} [{var_name}] {msg}")

            if not success:
                failed.append((key, msg))

    print("\n" + "=" * 70)
    if failed:
        print(f"FAILED: {len(failed)} file(s)")
        for key, msg in failed[:10]:  # Show first 10 failures
            print(f"  - {key}: {msg}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    else:
        print("SUCCESS: All files downloaded and verified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
