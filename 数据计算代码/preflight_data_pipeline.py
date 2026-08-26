"""离线预检 GNSS/ERA5 数据管线，不读取认证材料、不联网、不下载数据。

用途：在启动任何 GNSS、ERA5 下载或 ERA5 积分计算前，确认本地代码、路径、依赖与
MATLAB 积分包是否齐全。默认仅输出人工可读报告；--json 适合后续自动化；--strict
在发现阻塞项时返回非零退出码。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Check:
    code: str
    severity: str  # OK / WARN / BLOCKER
    message: str
    path: str | None = None


def exists_check(code: str, path: Path, label: str, severity: str = "BLOCKER") -> Check:
    if path.exists():
        return Check(code, "OK", f"已找到：{label}。", str(path))
    return Check(code, severity, f"缺失：{label}。", str(path))


def module_check(name: str, required_by: str, severity: str = "BLOCKER") -> Check:
    if importlib.util.find_spec(name) is not None:
        return Check(f"MODULE_{name.upper()}", "OK", f"Python 依赖 {name} 可用（{required_by}）")
    return Check(
        f"MODULE_{name.upper()}",
        severity,
        f"Python 依赖 {name} 缺失（{required_by}）；请先在独立环境中安装并记录版本。",
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def find_named_file(roots: Iterable[Path], filename: str) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        try:
            found.extend(root.rglob(filename))
        except OSError:
            continue
    return sorted({p.resolve() for p in found})


def run_checks(project_root: Path) -> list[Check]:
    code_dir = project_root / "数据计算代码"
    gnss_script = code_dir / "download_gnss_stations.py"
    era5_s3 = code_dir / "download_era5_2018_2024.py"
    era5_cds = code_dir / "download_era5_hk_cds.py"
    era5_zip = code_dir / "ERA5_obtain.zip"
    checks: list[Check] = []

    checks.append(exists_check("GNSS_SCRIPT", gnss_script, "GNSS 站点下载脚本"))
    checks.append(exists_check("ERA5_S3_SCRIPT", era5_s3, "ERA5 公共数据下载脚本"))
    checks.append(exists_check("ERA5_CDS_SCRIPT", era5_cds, "ERA5 CDS 下载脚本"))
    checks.append(exists_check("ERA5_INTEGRATION_ZIP", era5_zip, "ERA5 MATLAB 积分代码压缩包"))

    # GNSS 自定义包与脚本默认的本地资源。只检查文件名与路径，不读取任何认证材料。
    expected_module = code_dir / "src" / "data" / "download" / "gnss_downloader.py"
    direct_module = code_dir / "gnss_downloader.py"
    module_candidates = find_named_file([project_root], "gnss_downloader.py")
    if expected_module.exists():
        checks.append(Check("GNSS_DOWNLOADER", "OK", "GNSSDownloader 自定义实现存在于预期包路径。", str(expected_module)))
    elif direct_module.exists():
        checks.append(
            Check(
                "GNSS_DOWNLOADER",
                "OK",
                "已找到数据计算代码目录内的独立 gnss_downloader.py；下载脚本可使用兼容导入回退。",
                str(direct_module),
            )
        )
    elif module_candidates:
        checks.append(
            Check(
                "GNSS_DOWNLOADER",
                "WARN",
                "发现 gnss_downloader.py，但不在数据计算代码目录或预期包路径；启动前需要显式整理包结构或导入路径。",
                "; ".join(map(str, module_candidates[:3])),
            )
        )
    else:
        checks.append(
            Check(
                "GNSS_DOWNLOADER",
                "BLOCKER",
                "未找到 src.data.download.gnss_downloader 所需的 gnss_downloader.py；不要运行 GNSS 下载脚本。",
                str(expected_module),
            )
        )

    labels_dir = code_dir / "download_lables"
    checks.extend(
        [
            exists_check("GNSS_CONFIG", labels_dir / "config_v2.yaml", "GNSS 默认配置文件"),
            exists_check("GNSS_STATION_LIST", labels_dir / "GNSS_list", "GNSS 站点列表目录"),
            exists_check("GNSS_EGM96", labels_dir / "data" / "geoids" / "egm96-5.pgm", "EGM96 大地水准面文件", "WARN"),
        ]
    )

    checks.extend(
        [
            module_check("boto3", "download_era5_2018_2024.py"),
            module_check("cdsapi", "download_era5_hk_cds.py"),
            module_check("pygeodesy", "GNSS 高程修正（缺失时脚本可降级，但 PWV 可能受影响）", "WARN"),
            module_check("earthscope_sdk", "可选 EarthScope 认证源", "WARN"),
        ]
    )

    if era5_s3.exists() and 'Path("/f/era5_single_level")' in read_text(era5_s3):
        checks.append(Check("ERA5_S3_WINDOWS_PATH", "WARN", "ERA5 公共数据脚本含 POSIX 风格输出路径；Windows 运行前需改为明确的本地目录。", str(era5_s3)))
    if era5_cds.exists() and 'Path("F:/era5_hk_cds")' in read_text(era5_cds):
        drive = Path("F:/")
        severity = "WARN" if drive.exists() else "BLOCKER"
        checks.append(Check("ERA5_CDS_OUTPUT_DRIVE", severity, "ERA5 CDS 脚本使用固定 F: 输出目录；当前驱动器状态已检查。", str(drive)))

    if era5_zip.exists():
        expected_entries = {"ERA5_obtain/SHAtrop_layer.m", "ERA5_obtain/Inter_fromncfile_new.m"}
        try:
            with zipfile.ZipFile(era5_zip) as archive:
                names = set(archive.namelist())
            missing = sorted(expected_entries - names)
            if missing:
                checks.append(Check("ERA5_MATLAB_CONTENT", "BLOCKER", "ERA5 积分压缩包缺少关键 MATLAB 文件。", ", ".join(missing)))
            else:
                checks.append(Check("ERA5_MATLAB_CONTENT", "OK", "ERA5 积分压缩包包含关键 MATLAB 文件。", str(era5_zip)))
        except (OSError, zipfile.BadZipFile) as exc:
            checks.append(Check("ERA5_MATLAB_CONTENT", "BLOCKER", f"无法读取 ERA5 积分压缩包：{type(exc).__name__}。", str(era5_zip)))

    matlab_available = any((Path(folder) / "matlab.exe").exists() for folder in [Path("C:/Program Files/MATLAB"), Path("C:/Program Files")])
    # 不依赖固定安装目录：PATH 中若有 matlab，使用运行解释器名称本身即可。
    import shutil
    if shutil.which("matlab") or matlab_available:
        checks.append(Check("MATLAB", "OK", "检测到 MATLAB；ERA5 积分代码可在准备好 NetCDF 输入后做小样本验证。"))
    else:
        checks.append(Check("MATLAB", "WARN", "未从 PATH 检测到 MATLAB；ERA5 积分代码当前为 MATLAB 版本。"))

    checks.append(
        Check(
            "AUTH_POLICY",
            "WARN",
            "本预检不会读取认证文件；首次网络实验应仅使用单日、单源、免认证模式，并写入隔离临时目录。",
        )
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="离线预检 GNSS/ERA5 数据管线")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--json", action="store_true", help="输出 JSON，不读取认证内容")
    parser.add_argument("--strict", action="store_true", help="存在 BLOCKER 时返回退出码 2")
    args = parser.parse_args()

    checks = run_checks(args.project_root.resolve())
    blockers = sum(c.severity == "BLOCKER" for c in checks)
    warnings = sum(c.severity == "WARN" for c in checks)
    if args.json:
        print(json.dumps({"project_root": str(args.project_root.resolve()), "blockers": blockers, "warnings": warnings, "checks": [asdict(c) for c in checks]}, ensure_ascii=False, indent=2))
    else:
        print("GNSS/ERA5 数据管线离线预检（不联网、不下载、不读取认证材料）")
        print(f"项目根目录: {args.project_root.resolve()}")
        for check in checks:
            suffix = f" | {check.path}" if check.path else ""
            print(f"[{check.severity}] {check.code}: {check.message}{suffix}")
        print(f"汇总: BLOCKER={blockers}, WARN={warnings}")
    return 2 if args.strict and blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())

