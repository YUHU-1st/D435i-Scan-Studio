from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import AppConfig, resolve_output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D435i 三维扫描、置信度显示与 OBJ/STEP 导出")
    parser.add_argument("--config", type=Path, help="YAML 配置文件")
    parser.add_argument("--bag", type=Path, help="读取 RealSense .bag，而不是实时相机")
    parser.add_argument("--reconstruct", type=Path, metavar="SESSION", help="仅重建已有扫描会话")
    parser.add_argument("--list-devices", action="store_true", help="列出 RealSense 设备")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AppConfig.load(args.config)
    output_dir = resolve_output_dir(config, args.config)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.list_devices:
        from .camera import RealSenseCamera

        devices = RealSenseCamera.list_devices()
        print(json.dumps(devices, ensure_ascii=False, indent=2))
        return 0 if devices else 1

    if args.reconstruct:
        from .offline import reconstruct_session

        def progress(value: float, message: str) -> None:
            print(f"[{value:6.1%}] {message}", flush=True)

        outputs = reconstruct_session(args.reconstruct, config, progress)
        print(json.dumps(outputs.report, ensure_ascii=False, indent=2))
        return 0

    try:
        from .app import run_gui

        run_gui(config, output_dir, args.bag)
        return 0
    except ImportError as exc:
        print(f"依赖缺失：{exc}\n请先运行 .\\setup.ps1", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
