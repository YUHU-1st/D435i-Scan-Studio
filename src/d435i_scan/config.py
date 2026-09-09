from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class CameraConfig:
    width: int = 848
    height: int = 480
    color_width: int = 1280
    color_height: int = 720
    fps: int = 30
    depth_min_m: float = 0.22
    depth_max_m: float = 1.20
    visual_preset: str = "high_accuracy"
    spatial_alpha: float = 0.50
    spatial_delta: float = 20.0
    spatial_iterations: int = 3
    temporal_alpha: float = 0.35
    temporal_delta: float = 18.0
    enable_temporal_filter: bool = False
    enable_hole_filling: bool = False
    hole_fill: int = 1


@dataclass(slots=True)
class ReconstructionConfig:
    device: str = "CPU:0"
    voxel_size_m: float = 0.0025
    trunc_voxel_multiplier: float = 6.0
    block_count: int = 60_000
    surface_weight_threshold: float = 3.0
    max_surface_points: int = 1_500_000
    odometry_depth_diff_m: float = 0.025
    min_tracking_fitness: float = 0.12
    max_tracking_rmse_m: float = 0.025
    max_frame_translation_m: float = 0.18
    max_frame_rotation_deg: float = 35.0
    preview_every_n_frames: int = 12
    keyframe_translation_m: float = 0.012
    keyframe_rotation_deg: float = 4.0
    keyframe_max_interval_s: float = 0.70
    icp_max_correspondence_m: float = 0.012
    loop_closure_distance_m: float = 0.09
    loop_closure_min_separation: int = 15
    loop_closure_max_per_frame: int = 2
    smooth_iterations: int = 5
    target_triangles: int = 250_000


@dataclass(slots=True)
class RoiConfig:
    size_x_m: float = 0.45
    size_y_m: float = 0.45
    size_z_m: float = 0.45
    center_crop_fraction: float = 0.22
    margin_m: float = 0.015


@dataclass(slots=True)
class ExportConfig:
    export_obj: bool = True
    export_ply: bool = True
    export_step: bool = True
    freecad_cmd: str = ""
    step_tolerance_mm: float = 0.05
    step_max_triangles: int = 50_000


@dataclass(slots=True)
class AppConfig:
    output_dir: str = "output"
    camera: CameraConfig = field(default_factory=CameraConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path | None) -> AppConfig:
        if path is None:
            return cls()
        source = Path(path)
        if not source.exists():
            return cls()
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        return cls(
            output_dir=str(raw.get("output_dir", "output")),
            camera=CameraConfig(**raw.get("camera", {})),
            reconstruction=ReconstructionConfig(**raw.get("reconstruction", {})),
            roi=RoiConfig(**raw.get("roi", {})),
            export=ExportConfig(**raw.get("export", {})),
        )


def resolve_output_dir(config: AppConfig, config_path: str | Path | None) -> Path:
    output = Path(config.output_dir)
    if output.is_absolute() or config_path is None:
        return output.resolve()
    return (Path(config_path).resolve().parent / output).resolve()
