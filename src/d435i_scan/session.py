from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .camera import FramePacket
from .config import AppConfig


class ScanSession:
    def __init__(self, output_root: str | Path, config: AppConfig) -> None:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        self.path = Path(output_root).resolve() / f"scan_{stamp}"
        self.color_dir = self.path / "color"
        self.depth_dir = self.path / "depth"
        self.color_dir.mkdir(parents=True, exist_ok=False)
        self.depth_dir.mkdir(parents=True, exist_ok=False)
        self.config = config
        self.frames: list[dict[str, Any]] = []
        self.intrinsics: dict[str, Any] | None = None
        self.depth_unit_m: float | None = None
        self.roi_center_world: list[float] | None = None
        self.finished = False
        config.save(self.path / "settings.yml")

    def set_geometry(
        self,
        frame: FramePacket,
        roi_center_world: np.ndarray,
    ) -> None:
        self.intrinsics = frame.intrinsics.to_dict()
        self.depth_unit_m = float(frame.depth_unit_m)
        self.roi_center_world = [float(v) for v in roi_center_world]
        self._write_manifest()

    def add_keyframe(
        self,
        frame: FramePacket,
        camera_to_world: np.ndarray,
        tracking_fitness: float,
        tracking_rmse_m: float,
        mean_depth_quality: float,
    ) -> int:
        index = len(self.frames)
        stem = f"{index:06d}"
        color_relative = Path("color") / f"{stem}.jpg"
        depth_relative = Path("depth") / f"{stem}.png"
        Image.fromarray(frame.color_rgb, mode="RGB").save(
            self.path / color_relative,
            quality=95,
            subsampling=0,
        )
        Image.fromarray(frame.depth_raw.astype(np.uint16)).save(
            self.path / depth_relative,
            compress_level=2,
        )
        self.frames.append(
            {
                "index": index,
                "source_frame": int(frame.frame_number),
                "timestamp_s": float(frame.timestamp_s),
                "color": color_relative.as_posix(),
                "depth": depth_relative.as_posix(),
                "camera_to_world": np.asarray(camera_to_world, dtype=float).tolist(),
                "tracking_fitness": float(tracking_fitness),
                "tracking_rmse_m": float(tracking_rmse_m),
                "mean_depth_quality": float(mean_depth_quality),
            }
        )
        self._write_manifest()
        return index

    def finish(self) -> None:
        self.finished = True
        self._write_manifest()

    def _write_manifest(self) -> None:
        payload = {
            "format_version": 1,
            "created_at": self.path.name.removeprefix("scan_"),
            "finished": self.finished,
            "intrinsics": self.intrinsics,
            "depth_unit_m": self.depth_unit_m,
            "roi_center_world": self.roi_center_world,
            "frames": self.frames,
        }
        temp = self.path / "session.json.tmp"
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path / "session.json")


def load_session(path: str | Path) -> dict[str, Any]:
    session_path = Path(path).resolve()
    return json.loads((session_path / "session.json").read_text(encoding="utf-8"))
