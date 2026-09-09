from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import CameraConfig


@dataclass(slots=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CameraIntrinsics:
        return cls(
            width=int(value["width"]),
            height=int(value["height"]),
            fx=float(value["fx"]),
            fy=float(value["fy"]),
            cx=float(value["cx"]),
            cy=float(value["cy"]),
        )


@dataclass(slots=True)
class FramePacket:
    color_rgb: np.ndarray
    depth_raw: np.ndarray
    timestamp_s: float
    frame_number: int
    intrinsics: CameraIntrinsics
    depth_unit_m: float


class RealSenseCamera:
    """Small, stateful librealsense wrapper with a persistent filter chain."""

    def __init__(self, config: CameraConfig, bag_path: str | Path | None = None) -> None:
        self.config = config
        self.bag_path = Path(bag_path).resolve() if bag_path else None
        self._rs: Any = None
        self._pipeline: Any = None
        self._align: Any = None
        self._filters: list[Any] = []
        self._profile: Any = None
        self._last_frame_number: int | None = None
        self.depth_unit_m = 0.001
        self.device_name = "未连接"
        self.serial_number = ""
        self.usb_type = "未知"
        self.active_stream = ""
        self.degraded_mode = False

    @staticmethod
    def list_devices() -> list[dict[str, str]]:
        try:
            import pyrealsense2 as rs
        except ImportError:
            return []
        devices: list[dict[str, str]] = []
        for device in rs.context().query_devices():
            devices.append(
                {
                    "name": device.get_info(rs.camera_info.name),
                    "serial": device.get_info(rs.camera_info.serial_number),
                    "firmware": device.get_info(rs.camera_info.firmware_version),
                }
            )
        return devices

    def start(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("缺少 pyrealsense2；请先运行 setup.ps1") from exc

        self._rs = rs
        self._pipeline = rs.pipeline()
        pipeline_config = rs.config()
        if self.bag_path is not None:
            if not self.bag_path.exists():
                raise FileNotFoundError(f"找不到 RealSense BAG: {self.bag_path}")
            rs.config.enable_device_from_file(pipeline_config, str(self.bag_path), False)
        else:
            devices = rs.context().query_devices()
            if len(devices) == 0:
                raise RuntimeError("未检测到 RealSense 相机")
            detected_device = devices[0]
            if detected_device.supports(rs.camera_info.usb_type_descriptor):
                self.usb_type = detected_device.get_info(
                    rs.camera_info.usb_type_descriptor
                )

            depth_width = self.config.width
            depth_height = self.config.height
            color_width = self.config.color_width
            color_height = self.config.color_height
            stream_fps = self.config.fps
            if self.usb_type.startswith("2"):
                # Functional fallback for USB 2.x. This preserves 30 FPS motion
                # tracking, but the UI marks it as unsuitable for final scans.
                depth_width, depth_height = 640, 480
                color_width, color_height = 640, 480
                stream_fps = min(stream_fps, 30)
                self.degraded_mode = (
                    depth_width != self.config.width
                    or depth_height != self.config.height
                    or color_width != self.config.color_width
                    or color_height != self.config.color_height
                    or stream_fps != self.config.fps
                )

            pipeline_config.enable_stream(
                rs.stream.depth,
                depth_width,
                depth_height,
                rs.format.z16,
                stream_fps,
            )
            pipeline_config.enable_stream(
                rs.stream.color,
                color_width,
                color_height,
                rs.format.rgb8,
                stream_fps,
            )
            self.active_stream = (
                f"深度 {depth_width}x{depth_height} / 彩色 "
                f"{color_width}x{color_height} @ {stream_fps} FPS"
            )

        self._profile = self._pipeline.start(pipeline_config)
        self._last_frame_number = None
        device = self._profile.get_device()
        self.device_name = device.get_info(rs.camera_info.name)
        self.serial_number = device.get_info(rs.camera_info.serial_number)
        if self.bag_path is not None:
            self.usb_type = "BAG 回放"
            self.active_stream = "录制文件配置"
        depth_sensor = device.first_depth_sensor()
        self.depth_unit_m = float(depth_sensor.get_depth_scale())

        preset_names = {
            "high_accuracy": rs.rs400_visual_preset.high_accuracy,
            "high_density": rs.rs400_visual_preset.high_density,
            "medium_density": rs.rs400_visual_preset.medium_density,
        }
        if depth_sensor.supports(rs.option.visual_preset):
            preset = preset_names.get(self.config.visual_preset)
            if preset is not None:
                depth_sensor.set_option(rs.option.visual_preset, float(preset))

        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, float(self.config.spatial_iterations))
        spatial.set_option(rs.option.filter_smooth_alpha, float(self.config.spatial_alpha))
        spatial.set_option(rs.option.filter_smooth_delta, float(self.config.spatial_delta))
        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, float(self.config.temporal_alpha))
        temporal.set_option(rs.option.filter_smooth_delta, float(self.config.temporal_delta))
        hole = rs.hole_filling_filter(int(self.config.hole_fill))
        self._filters = [rs.disparity_transform(True), spatial]
        if self.config.enable_temporal_filter:
            self._filters.append(temporal)
        self._filters.append(rs.disparity_transform(False))
        if self.config.enable_hole_filling:
            self._filters.append(hole)
        # Preserve the recommended 848x480 depth sampling grid. The RGB frame
        # is resampled onto depth, so Open3D receives same-sized RGB-D images.
        self._align = rs.align(rs.stream.depth)

        if self.bag_path is not None:
            playback = device.as_playback()
            playback.set_real_time(False)

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                try:
                    self._pipeline.stop()
                except RuntimeError:
                    # A physical disconnect can invalidate the pipeline before
                    # the worker reaches its cleanup path.
                    pass
            finally:
                self._pipeline = None
                self._profile = None
                self._last_frame_number = None

    def read(self, timeout_ms: int = 5000) -> FramePacket:
        if self._pipeline is None or self._rs is None:
            raise RuntimeError("相机尚未启动")
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            frames = self._pipeline.wait_for_frames(remaining_ms)
            aligned = self._align.process(frames)
            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()
            if not depth or not color:
                raise RuntimeError("未收到同步且对齐的彩色/深度帧")
            source_frame_number = int(depth.get_frame_number())
            if source_frame_number != self._last_frame_number:
                self._last_frame_number = source_frame_number
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("等待新的相机帧超时")
        for processing_filter in self._filters:
            depth = processing_filter.process(depth)
        depth = depth.as_depth_frame()

        intr = depth.profile.as_video_stream_profile().intrinsics
        intrinsics = CameraIntrinsics(
            width=int(intr.width),
            height=int(intr.height),
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.ppx),
            cy=float(intr.ppy),
        )
        depth_array = np.ascontiguousarray(np.asanyarray(depth.get_data()).copy())
        color_array = np.ascontiguousarray(np.asanyarray(color.get_data()).copy())
        return FramePacket(
            color_rgb=color_array,
            depth_raw=depth_array,
            timestamp_s=float(depth.get_timestamp()) / 1000.0,
            frame_number=int(depth.get_frame_number()),
            intrinsics=intrinsics,
            depth_unit_m=self.depth_unit_m,
        )
