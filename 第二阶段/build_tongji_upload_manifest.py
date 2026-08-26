"""Build a secret-free manifest for the Tongji upload set."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--output", default="tongji_upload_manifest_20260826.json")
    ap.add_argument("--hash-profiles", action="store_true")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    rel_dirs = [
        Path("第二阶段/profile_reconstructed_igra_20260825"),
        Path("第二阶段/result_strict_igra_reconstructed_20260825"),
    ]
    entries = []
    for rel_dir in rel_dirs:
        directory = root / rel_dir
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            item = {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
            }
            if args.hash_profiles or rel_dir.name != "profile_reconstructed_igra_20260825":
                item["sha256"] = sha256(path)
            entries.append(item)
    result = {
        "created_by": "build_tongji_upload_manifest.py",
        "protocol": "tongji_upload_20260826",
        "secret_policy": "no credentials included",
        "profile_hashes": bool(args.hash_profiles),
        "entries": entries,
        "file_count": len(entries),
        "total_bytes": sum(x["bytes"] for x in entries),
    }
    output = root / args.output
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("file_count", "total_bytes", "profile_hashes")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
