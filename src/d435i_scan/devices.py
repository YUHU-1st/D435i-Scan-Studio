"""RealSense discovery and capability negotiation; no camera streams are opened here."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class CameraError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class StreamMode:
    width: int
    height: int
    fps: int
    format: str

    @property
    def key(self) -> str:
        return f"{self.width}x{self.height}@{self.fps}:{self.format}"

    @property
    def label(self) -> str:
        return f"{self.width}×{self.height} @ {self.fps} FPS · {self.format}"


@dataclass(frozen=True, slots=True)
class PresetOption:
    key: str
    label: str


@dataclass(slots=True)
class DeviceInfo:
    name: str
    serial: str
    firmware: str = ""
    product_id: str = ""
    usb_type: str = "未知"
    physical_port: str = ""
    depth_modes: list[StreamMode] = field(default_factory=list)
    color_modes: list[StreamMode] = field(default_factory=list)
    has_imu: bool = False
    presets: list[PresetOption] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)

    @property
    def scan_reason(self) -> str:
        if not self.depth_modes:
            return "未提供本项目可用的 Z16 深度流。"
        if not self.color_modes:
            return "未提供 RGB8/BGR8 彩色流；当前 RGB-D 重建需要彩色和深度。"
        return ""

    @property
    def can_scan(self) -> bool:
        return not self.scan_reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "serial": self.serial, "firmware": self.firmware,
            "product_id": self.product_id, "usb_type": self.usb_type,
            "physical_port": self.physical_port,
            "depth_modes": [m.key for m in self.depth_modes],
            "color_modes": [m.key for m in self.color_modes],
            "has_imu": self.has_imu,
            "presets": [{"key": p.key, "label": p.label} for p in self.presets],
            "can_scan": self.can_scan, "scan_reason": self.scan_reason,
            "diagnostics": self.diagnostics,
        }


def load_sdk() -> Any:
    try:
        import pyrealsense2 as rs
    except (ImportError, OSError) as exc:
        raise CameraError(
            "sdk_missing", "无法加载 pyrealsense2 / RealSense SDK。请运行 setup.cmd，"
            "并确认使用 64 位 Python 3.12；若已安装，请检查运行库。"
            f" 原始错误：{exc}",
        ) from exc
    return rs


def diagnose_error(exc: Exception, operation: str = "初始化") -> CameraError:
    if isinstance(exc, CameraError):
        return exc
    detail = str(exc)
    text = detail.lower()
    if any(word in text for word in ("busy", "in use", "resource temporarily unavailable", "access denied")):
        code = "device_busy"
        hint = "设备可能被其他程序占用。请关闭 RealSense Viewer、其他采集程序后重试。"
    elif any(word in text for word in ("no device", "not found", "disconnected", "failed to resolve request")):
        code = "device_unavailable"
        hint = "设备不可用或已断开。请刷新设备列表，检查 USB 连接及所选流参数。"
    elif any(word in text for word in ("backend", "driver", "winusb", "permission")):
        code = "driver_error"
        hint = "无法访问设备后端。请检查 Windows 设备管理器、USB 驱动和 RealSense SDK 安装。"
    else:
        code = "initialization_failed"
        hint = "请检查设备连接、固件及所选流参数；可用 RealSense Viewer 进一步诊断。"
    return CameraError(code, f"{operation}失败：{hint} 原始错误：{detail}")


def _info(device: Any, rs: Any, name: str, default: str = "") -> str:
    option = getattr(rs.camera_info, name, None)
    if option is None:
        return default
    try:
        if device.supports(option):
            return str(device.get_info(option))
    except RuntimeError:
        pass
    return default


def _sensor_modes(sensor: Any, rs: Any, kind: str) -> list[StreamMode]:
    allowed = {"depth": {"z16"}, "color": {"rgb8", "bgr8"}}[kind]
    modes: set[StreamMode] = set()
    for profile in sensor.get_stream_profiles():
        if profile.stream_type() != getattr(rs.stream, kind):
            continue
        fmt = str(profile.format()).split(".")[-1].lower()
        if fmt not in allowed:
            continue
        video = profile.as_video_stream_profile()
        modes.add(StreamMode(int(video.width()), int(video.height()), int(profile.fps()), fmt))
    return sorted(modes, key=lambda m: (m.fps, m.width * m.height, m.format), reverse=True)


def _presets(sensor: Any, rs: Any) -> list[PresetOption]:
    option = getattr(rs.option, "visual_preset", None)
    if option is None or not sensor.supports(option):
        return []
    value_range = sensor.get_option_range(option)
    lo, hi = int(value_range.min), int(value_range.max)
    if hi - lo > 128:
        return []
    result = []
    for value in range(lo, hi + 1):
        try:
            label = sensor.get_option_value_description(option, float(value))
        except (AttributeError, RuntimeError):
            label = f"Preset {value}"
        result.append(PresetOption(str(value), str(label)))
    return result


def discover_devices(rs: Any = None) -> list[DeviceInfo]:
    rs = rs if rs is not None else load_sdk()
    try:
        devices = rs.context().query_devices()
    except RuntimeError as exc:
        raise diagnose_error(exc, "设备发现") from exc
    result = []
    for device in devices:
        info = DeviceInfo(
            name=_info(device, rs, "name", "未知设备"),
            serial=_info(device, rs, "serial_number"),
            firmware=_info(device, rs, "firmware_version"),
            product_id=_info(device, rs, "product_id"),
            usb_type=_info(device, rs, "usb_type_descriptor", "未知"),
            physical_port=_info(device, rs, "physical_port"),
        )
        try:
            sensors = device.query_sensors()
        except RuntimeError as exc:
            info.diagnostics.append(f"无法读取传感器：{exc}。请关闭占用设备的程序后刷新。")
            result.append(info)
            continue
        depth, color = set(), set()
        for sensor in sensors:
            try:
                depth.update(_sensor_modes(sensor, rs, "depth"))
                color.update(_sensor_modes(sensor, rs, "color"))
                if not info.presets:
                    try:
                        info.presets = _presets(sensor, rs)
                    except RuntimeError as exc:
                        info.diagnostics.append(f"无法读取视觉预设：{exc}")
                for profile in sensor.get_stream_profiles():
                    if profile.stream_type() in (rs.stream.accel, rs.stream.gyro):
                        info.has_imu = True
            except RuntimeError as exc:
                info.diagnostics.append(f"无法读取流能力：{exc}。请关闭占用设备的程序后刷新。")
        info.depth_modes = sorted(depth, key=lambda m: (m.fps, m.width * m.height, m.format), reverse=True)
        info.color_modes = sorted(color, key=lambda m: (m.fps, m.width * m.height, m.format), reverse=True)
        result.append(info)
    return result


def select_streams(device: DeviceInfo, config: Any) -> tuple[StreamMode, StreamMode]:
    if not device.can_scan:
        raise CameraError("capability_missing", device.scan_reason)

    def choose(modes: list[StreamMode], explicit: str, width: int, height: int, fps: int) -> StreamMode:
        if explicit:
            for mode in modes:
                if mode.key == explicit:
                    return mode
            raise CameraError("capability_missing", f"设备不支持所选流 {explicit}，请重新选择。")
        # Preserve the D435i defaults when available, but negotiate actual SDK profiles.
        return min(modes, key=lambda m: (
            abs(m.fps - fps),
            abs(m.width - width) + abs(m.height - height),
            0 if m.format in ("z16", "rgb8") else 1,
            -(m.width * m.height),
        ))

    depth_width, depth_height = config.width, config.height
    color_width, color_height = config.color_width, config.color_height
    if device.usb_type.startswith("2") and not (config.depth_mode or config.color_mode):
        depth_width, depth_height = min(depth_width, 640), min(depth_height, 480)
        color_width, color_height = min(color_width, 640), min(color_height, 480)
    depth = choose(device.depth_modes, config.depth_mode, depth_width, depth_height, config.fps)
    color = choose(device.color_modes, config.color_mode, color_width, color_height, depth.fps)
    return depth, color
