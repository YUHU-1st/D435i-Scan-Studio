from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import CameraConfig
from .devices import CameraError, DeviceInfo, diagnose_error, discover_devices, load_sdk, resolve_streams


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
            "width": self.width, "height": self.height, "fx": self.fx,
            "fy": self.fy, "cx": self.cx, "cy": self.cy,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CameraIntrinsics:
        return cls(
            width=int(value["width"]), height=int(value["height"]),
            fx=float(value["fx"]), fy=float(value["fy"]),
            cx=float(value["cx"]), cy=float(value["cy"]),
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
    """Stateful librealsense RGB-D wrapper; selection is by actual capabilities."""

    def __init__(self, config: CameraConfig, bag_path: str | Path | None = None) -> None:
        self.config = config
        self.bag_path = Path(bag_path).resolve() if bag_path else None
        self._rs: Any = None
        self._pipeline: Any = None
        self._align: Any = None
        self._filters: list[Any] = []
        self._profile: Any = None
        self._last_frame_number: int | None = None
        self._color_format = "rgb8"
        self._has_stereo = True
        self.depth_unit_m = 0.001
        self.device_name = "未连接"
        self.serial_number = ""
        self.usb_type = "未知"
        self.active_stream = ""
        self.degraded_mode = False
        self.warnings: list[str] = []

    @staticmethod
    def list_devices() -> list[dict[str, Any]]:
        return [device.to_dict() for device in discover_devices()]

    @staticmethod
    def _find_device(devices: list[DeviceInfo], serial: str) -> DeviceInfo:
        if serial:
            for device in devices:
                if device.serial == serial:
                    return device
            raise CameraError("device_unavailable", "所选设备已断开或不再可见，请刷新设备列表。")
        if not devices:
            raise CameraError("no_device", "未发现 RealSense 设备。请检查 USB 连接、驱动及供电，然后刷新。")
        for device in devices:
            if device.can_scan:
                return device
        raise CameraError("capability_missing", devices[0].scan_reason)

    def start(self) -> None:
        if self._pipeline is not None:
            raise CameraError("already_started", "相机已经启动。")
        rs = load_sdk()
        self._rs = rs
        self.warnings = []
        self.degraded_mode = False
        pipeline = rs.pipeline()
        self._pipeline = pipeline
        pipeline_config = rs.config()
        started = False
        try:
            if self.bag_path is not None:
                if not self.bag_path.is_file():
                    raise CameraError("file_missing", f"找不到 RealSense BAG: {self.bag_path}")
                rs.config.enable_device_from_file(pipeline_config, str(self.bag_path), False)
            else:
                devices = discover_devices(rs)
                device = self._find_device(devices, self.config.device_serial)
                if not device.serial and len(devices) > 1:
                    raise CameraError("device_unavailable", "设备没有可用序列号，无法在多设备环境中安全选择。")
                depth, color, pipeline_config = resolve_streams(
                    device, self.config, rs, pipeline,
                )
                self._color_format = color.format
                self._has_stereo = device.has_stereo
                self.usb_type = device.usb_type
                self.active_stream = f"深度 {depth.label} / 彩色 {color.label}"
                self.degraded_mode = (
                    depth.width != self.config.width or depth.height != self.config.height
                    or color.width != self.config.color_width or color.height != self.config.color_height
                    or depth.fps != self.config.fps or color.fps != self.config.fps
                )
                if device.usb_type.startswith("2"):
                    self.warnings.append("USB 2.x 带宽有限，建议使用 USB 3.x；当前流已按实际能力协商。")
                if self.degraded_mode:
                    self.warnings.append("实际流与配置偏好不同，已选择设备支持的模式。")

            self._profile = pipeline.start(pipeline_config)
            started = True
            device = self._profile.get_device()
            self.device_name = device.get_info(rs.camera_info.name)
            self.serial_number = device.get_info(rs.camera_info.serial_number)
            if self.bag_path is None and self.config.device_serial and self.serial_number != self.config.device_serial:
                raise CameraError("device_unavailable", "SDK 返回的设备与所选序列号不一致，已停止采集。")
            if self.bag_path is not None:
                self.usb_type = "BAG 回放"
                self.active_stream = "录制文件配置"
                color_profile = self._profile.get_stream(rs.stream.color)
                self._color_format = str(color_profile.format()).split(".")[-1].lower()
            depth_sensor = device.first_depth_sensor()
            self.depth_unit_m = float(depth_sensor.get_depth_scale())
            if self.depth_unit_m <= 0:
                raise CameraError("initialization_failed", "设备返回无效深度比例。")
            if self.bag_path is None:
                self._apply_preset(depth_sensor)
            stereo = getattr(rs.option, "stereo_baseline", None)
            if self.bag_path is not None:
                self._has_stereo = stereo is not None and depth_sensor.supports(stereo)
            self._build_filters()
            self._align = rs.align(rs.stream.depth)
            self._last_frame_number = None
            if self.bag_path is not None:
                device.as_playback().set_real_time(False)
        except Exception as exc:
            if started:
                try:
                    pipeline.stop()
                except RuntimeError:
                    pass
            self._pipeline = None
            self._profile = None
            self._align = None
            self._filters = []
            raise diagnose_error(exc) from exc

    def _apply_preset(self, sensor: Any) -> None:
        rs = self._rs
        requested = self.config.visual_preset
        if not requested:
            return
        option = getattr(rs.option, "visual_preset", None)
        if option is None:
            return
        if not sensor.supports(option):
            self.warnings.append("当前深度传感器不支持视觉预设，已保留设备默认值。")
            return
        from .devices import _presets

        available = _presets(sensor, rs)
        if requested.lstrip("-").isdigit():
            value = requested
        else:
            normalized = requested.lower().replace("_", " ").replace("-", " ")
            match = next((p for p in available if p.label.lower().replace("_", " ").replace("-", " ") == normalized), None)
            if match is None:
                self.warnings.append(f"设备不支持预设 {requested}，已保留默认值。")
                return
            value = match.key
        if value not in {p.key for p in available}:
            self.warnings.append(f"设备不支持预设 {requested}，已保留默认值。")
            return
        try:
            if sensor.is_option_read_only(option):
                self.warnings.append("视觉预设为只读，已保留设备当前设置。")
                return
            sensor.set_option(option, float(value))
        except RuntimeError as exc:
            self.warnings.append(f"无法设置视觉预设，已保留设备当前设置：{exc}")

    def _build_filters(self) -> None:
        rs = self._rs
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, float(self.config.spatial_iterations))
        spatial.set_option(rs.option.filter_smooth_alpha, float(self.config.spatial_alpha))
        spatial.set_option(rs.option.filter_smooth_delta, float(self.config.spatial_delta))
        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, float(self.config.temporal_alpha))
        temporal.set_option(rs.option.filter_smooth_delta, float(self.config.temporal_delta))
        hole = rs.hole_filling_filter(int(self.config.hole_fill))
        self._filters = [rs.disparity_transform(True)] if self._has_stereo else []
        self._filters.append(spatial)
        if self.config.enable_temporal_filter:
            self._filters.append(temporal)
        if self._has_stereo:
            self._filters.append(rs.disparity_transform(False))
        if self.config.enable_hole_filling:
            self._filters.append(hole)

    def stop(self) -> None:
        pipeline, self._pipeline = self._pipeline, None
        self._profile = None
        self._align = None
        self._filters = []
        self._last_frame_number = None
        if pipeline is not None:
            try:
                pipeline.stop()
            except RuntimeError:
                pass

    def read(self, timeout_ms: int = 5000) -> FramePacket:
        if self._pipeline is None or self._rs is None:
            raise CameraError("not_started", "相机尚未启动")
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            try:
                frames = self._pipeline.wait_for_frames(remaining_ms)
                aligned = self._align.process(frames)
                depth = aligned.get_depth_frame()
                color = aligned.get_color_frame()
            except RuntimeError as exc:
                raise diagnose_error(exc, "读取相机帧") from exc
            if not depth or not color:
                raise CameraError("frame_missing", "未收到同步且对齐的彩色/深度帧，请检查流参数。")
            source_frame_number = int(depth.get_frame_number())
            if source_frame_number != self._last_frame_number:
                self._last_frame_number = source_frame_number
                break
            if time.monotonic() >= deadline:
                raise CameraError("frame_timeout", "等待新的相机帧超时，请检查连接或关闭占用设备的程序。")
        for processing_filter in self._filters:
            depth = processing_filter.process(depth)
        depth = depth.as_depth_frame()
        intr = depth.profile.as_video_stream_profile().intrinsics
        intrinsics = CameraIntrinsics(
            width=int(intr.width), height=int(intr.height), fx=float(intr.fx),
            fy=float(intr.fy), cx=float(intr.ppx), cy=float(intr.ppy),
        )
        depth_array = np.ascontiguousarray(np.asanyarray(depth.get_data()).copy())
        color_array = np.ascontiguousarray(np.asanyarray(color.get_data()).copy())
        if self._color_format == "bgr8":
            color_array = np.ascontiguousarray(color_array[..., ::-1])
        return FramePacket(
            color_rgb=color_array, depth_raw=depth_array,
            timestamp_s=float(depth.get_timestamp()) / 1000.0,
            frame_number=int(depth.get_frame_number()), intrinsics=intrinsics,
            depth_unit_m=self.depth_unit_m,
        )
