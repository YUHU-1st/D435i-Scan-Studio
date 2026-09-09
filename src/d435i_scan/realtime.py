from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import CameraIntrinsics, FramePacket
from .config import AppConfig


@dataclass(slots=True)
class TrackingResult:
    accepted: bool
    camera_to_world: np.ndarray
    fitness: float
    rmse_m: float
    tracking_score: float
    preview_cloud: object | None = None
    reason: str = ""


def estimate_roi_center(frame: FramePacket, crop_fraction: float) -> np.ndarray:
    depth = frame.depth_raw.astype(np.float64) * frame.depth_unit_m
    height, width = depth.shape
    half_w = max(4, int(width * crop_fraction * 0.5))
    half_h = max(4, int(height * crop_fraction * 0.5))
    x0, x1 = width // 2 - half_w, width // 2 + half_w
    y0, y1 = height // 2 - half_h, height // 2 + half_h
    region = depth[y0:y1, x0:x1]
    valid = region[(region > 0.05) & np.isfinite(region)]
    if len(valid) < 30:
        raise RuntimeError("中心框内有效深度太少；请将物体放到十字线附近")
    z = float(np.median(valid))
    x = (width * 0.5 - frame.intrinsics.cx) * z / frame.intrinsics.fx
    y = (height * 0.5 - frame.intrinsics.cy) * z / frame.intrinsics.fy
    return np.array([x, y, z], dtype=np.float64)


def roi_bounds(center: np.ndarray, config: AppConfig) -> tuple[np.ndarray, np.ndarray]:
    size = np.array(
        [config.roi.size_x_m, config.roi.size_y_m, config.roi.size_z_m],
        dtype=np.float64,
    )
    half = size * 0.5 + config.roi.margin_m
    center_array = np.asarray(center, dtype=np.float64)
    return center_array - half, center_array + half


def _rotation_angle_deg(transform: np.ndarray) -> float:
    trace = float(np.trace(transform[:3, :3]))
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def keyframe_needed(
    previous_pose: np.ndarray | None,
    current_pose: np.ndarray,
    previous_timestamp_s: float | None,
    current_timestamp_s: float,
    config: AppConfig,
) -> bool:
    if previous_pose is None or previous_timestamp_s is None:
        return True
    relative = np.linalg.inv(previous_pose) @ current_pose
    translation = float(np.linalg.norm(relative[:3, 3]))
    rotation = _rotation_angle_deg(relative)
    elapsed = max(0.0, current_timestamp_s - previous_timestamp_s)
    rcfg = config.reconstruction
    return (
        translation >= rcfg.keyframe_translation_m
        or rotation >= rcfg.keyframe_rotation_deg
        or elapsed >= rcfg.keyframe_max_interval_s
    )


class RealtimeReconstructor:
    def __init__(
        self,
        config: AppConfig,
        intrinsics: CameraIntrinsics,
        roi_center_world: np.ndarray,
    ) -> None:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError("缺少 Open3D；请先运行 setup.ps1") from exc

        self.o3d = o3d
        self.config = config
        self.intrinsics = intrinsics
        self.roi_center_world = np.asarray(roi_center_world, dtype=np.float64)
        self.roi_min, self.roi_max = roi_bounds(self.roi_center_world, config)
        self.device = o3d.core.Device(config.reconstruction.device)
        self.intrinsic_tensor = o3d.core.Tensor(intrinsics.matrix, device=self.device)
        self.model = o3d.t.pipelines.slam.Model(
            config.reconstruction.voxel_size_m,
            16,
            config.reconstruction.block_count,
            o3d.core.Tensor(np.eye(4), device=self.device),
            self.device,
        )
        self.input_frame = o3d.t.pipelines.slam.Frame(
            intrinsics.height, intrinsics.width, self.intrinsic_tensor, self.device
        )
        self.raycast_frame = o3d.t.pipelines.slam.Frame(
            intrinsics.height, intrinsics.width, self.intrinsic_tensor, self.device
        )
        self.camera_to_world = np.eye(4, dtype=np.float64)
        self.frame_index = 0
        self.initialized = False
        self.previous_frame: FramePacket | None = None
        self.use_legacy_odometry = False
        ys, xs = np.mgrid[0 : intrinsics.height, 0 : intrinsics.width]
        self.ray_x = ((xs - intrinsics.cx) / intrinsics.fx).astype(np.float32)
        self.ray_y = ((ys - intrinsics.cy) / intrinsics.fy).astype(np.float32)

    def _set_frame(self, frame: FramePacket) -> None:
        o3d = self.o3d
        depth = frame.depth_raw.copy()
        depth_m = depth.astype(np.float32) * frame.depth_unit_m
        invalid = (depth_m < self.config.camera.depth_min_m) | (
            depth_m > self.config.camera.depth_max_m
        )
        # Predict the ROI in the new frame from the previous accepted pose. This
        # keeps tables and walls out of the TSDF while leaving a movement margin.
        z = depth_m
        x = self.ray_x * z
        y = self.ray_y * z
        rotation = self.camera_to_world[:3, :3]
        translation = self.camera_to_world[:3, 3]
        world_x = rotation[0, 0] * x + rotation[0, 1] * y + rotation[0, 2] * z + translation[0]
        world_y = rotation[1, 0] * x + rotation[1, 1] * y + rotation[1, 2] * z + translation[1]
        world_z = rotation[2, 0] * x + rotation[2, 1] * y + rotation[2, 2] * z + translation[2]
        motion_margin = 0.04
        outside_roi = (
            (world_x < self.roi_min[0] - motion_margin)
            | (world_x > self.roi_max[0] + motion_margin)
            | (world_y < self.roi_min[1] - motion_margin)
            | (world_y > self.roi_max[1] + motion_margin)
            | (world_z < self.roi_min[2] - motion_margin)
            | (world_z > self.roi_max[2] + motion_margin)
        )
        invalid |= outside_roi
        depth[invalid] = 0
        depth = np.ascontiguousarray(depth)
        color = np.ascontiguousarray(frame.color_rgb)
        depth_image = o3d.t.geometry.Image(o3d.core.Tensor(depth)).to(self.device)
        color_image = o3d.t.geometry.Image(o3d.core.Tensor(color)).to(self.device)
        self.input_frame.set_data_from_image("depth", depth_image)
        self.input_frame.set_data_from_image("color", color_image)
        if not self.initialized:
            self.raycast_frame.set_data_from_image("depth", depth_image)
            self.raycast_frame.set_data_from_image("color", color_image)

    def _track_with_legacy_rgbd(self, frame: FramePacket) -> tuple[bool, np.ndarray]:
        if self.previous_frame is None:
            return False, np.eye(4, dtype=np.float64)

        o3d = self.o3d

        def rgbd(packet: FramePacket) -> object:
            depth = packet.depth_raw.copy()
            depth_m = depth.astype(np.float32) * packet.depth_unit_m
            invalid = (depth_m < self.config.camera.depth_min_m) | (
                depth_m > self.config.camera.depth_max_m
            )
            depth[invalid] = 0
            return o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(packet.color_rgb)),
                o3d.geometry.Image(np.ascontiguousarray(depth)),
                depth_scale=1.0 / packet.depth_unit_m,
                depth_trunc=self.config.camera.depth_max_m,
                convert_rgb_to_intensity=True,
            )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.intrinsics.width,
            self.intrinsics.height,
            self.intrinsics.fx,
            self.intrinsics.fy,
            self.intrinsics.cx,
            self.intrinsics.cy,
        )
        option = o3d.pipelines.odometry.OdometryOption()
        option.depth_diff_max = self.config.reconstruction.odometry_depth_diff_m
        option.depth_min = self.config.camera.depth_min_m
        option.depth_max = self.config.camera.depth_max_m
        success, transform, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
            rgbd(frame),
            rgbd(self.previous_frame),
            intrinsic,
            np.eye(4, dtype=np.float64),
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            option,
        )
        return bool(success), np.asarray(transform, dtype=np.float64)

    def process(self, frame: FramePacket, request_preview: bool) -> TrackingResult:
        self._set_frame(frame)
        rcfg = self.config.reconstruction
        depth_scale = 1.0 / frame.depth_unit_m
        fitness = 1.0
        rmse = 0.0
        accepted = True
        reason = ""

        if self.initialized:
            try:
                if self.use_legacy_odometry:
                    success, transform = self._track_with_legacy_rgbd(frame)
                    if not success:
                        raise RuntimeError("彩色纹理里程计未找到稳定对应点")
                    fitness = 0.50
                    rmse = rcfg.odometry_depth_diff_m * 0.5
                    reason = "平面场景：使用彩色纹理里程计"
                else:
                    odometry = self.model.track_frame_to_model(
                        self.input_frame,
                        self.raycast_frame,
                        depth_scale,
                        self.config.camera.depth_max_m,
                        rcfg.odometry_depth_diff_m,
                    )
                    transform = odometry.transformation.cpu().numpy()
                    fitness = float(odometry.fitness)
                    rmse = float(odometry.inlier_rmse)
                translation = float(np.linalg.norm(transform[:3, 3]))
                rotation = _rotation_angle_deg(transform)
                accepted = bool(
                    np.all(np.isfinite(transform))
                    and fitness >= rcfg.min_tracking_fitness
                    and rmse <= rcfg.max_tracking_rmse_m
                    and translation <= rcfg.max_frame_translation_m
                    and rotation <= rcfg.max_frame_rotation_deg
                )
                if accepted:
                    self.camera_to_world = self.camera_to_world @ transform
                else:
                    reason = (
                        f"跟踪拒绝: fitness={fitness:.3f}, RMSE={rmse * 1000:.1f} mm, "
                        f"位移={translation * 1000:.1f} mm, 转角={rotation:.1f}°"
                    )
            except RuntimeError as exc:
                # Dense model tracking is under-constrained on planar objects.
                # Switch once to RGB-D frame odometry, which can use the color
                # texture on the support surface and avoids repeated LAPACK log
                # spam from the same singular system.
                if not self.use_legacy_odometry:
                    self.use_legacy_odometry = True
                    success, transform = self._track_with_legacy_rgbd(frame)
                    if success:
                        translation = float(np.linalg.norm(transform[:3, 3]))
                        rotation = _rotation_angle_deg(transform)
                        accepted = bool(
                            np.all(np.isfinite(transform))
                            and translation <= rcfg.max_frame_translation_m
                            and rotation <= rcfg.max_frame_rotation_deg
                        )
                        if accepted:
                            self.camera_to_world = self.camera_to_world @ transform
                            fitness = 0.50
                            rmse = rcfg.odometry_depth_diff_m * 0.5
                            reason = "平面场景：已切换彩色纹理里程计"
                    else:
                        accepted = False
                else:
                    accepted = False
                if not accepted:
                    fitness = 0.0
                    rmse = rcfg.max_tracking_rmse_m * 2.0
                    reason = f"跟踪退化，请退回上一视角: {str(exc).splitlines()[0]}"

        if accepted:
            pose_tensor = self.o3d.core.Tensor(self.camera_to_world, device=self.device)
            self.model.update_frame_pose(self.frame_index, pose_tensor)
            self.model.integrate(
                self.input_frame,
                depth_scale,
                self.config.camera.depth_max_m,
                rcfg.trunc_voxel_multiplier,
            )
            self.model.synthesize_model_frame(
                self.raycast_frame,
                depth_scale,
                self.config.camera.depth_min_m,
                self.config.camera.depth_max_m,
                rcfg.trunc_voxel_multiplier,
                True,
                rcfg.surface_weight_threshold,
            )
            self.initialized = True
            self.frame_index += 1
            self.previous_frame = frame

        preview = self.extract_preview_cloud() if request_preview and self.initialized else None
        tracking_score = float(
            np.clip(
                0.65 * fitness + 0.35 * np.exp(-rmse / max(rcfg.max_tracking_rmse_m, 1e-6)), 0, 1
            )
        )
        return TrackingResult(
            accepted=accepted,
            camera_to_world=self.camera_to_world.copy(),
            fitness=fitness,
            rmse_m=rmse,
            tracking_score=tracking_score,
            preview_cloud=preview,
            reason=reason,
        )

    def extract_preview_cloud(self) -> object:
        cloud = self.model.voxel_grid.extract_point_cloud(
            self.config.reconstruction.surface_weight_threshold,
            self.config.reconstruction.max_surface_points,
        ).to(self.o3d.core.Device("CPU:0"))
        legacy = cloud.to_legacy()
        box = self.o3d.geometry.AxisAlignedBoundingBox(self.roi_min, self.roi_max)
        cropped = legacy.crop(box)
        return cropped.voxel_down_sample(max(self.config.reconstruction.voxel_size_m * 2.0, 0.004))

    def extract_mesh(self) -> object:
        mesh = self.model.voxel_grid.extract_triangle_mesh(
            self.config.reconstruction.surface_weight_threshold,
            self.config.reconstruction.max_surface_points,
        ).to(self.o3d.core.Device("CPU:0"))
        legacy = mesh.to_legacy()
        box = self.o3d.geometry.AxisAlignedBoundingBox(self.roi_min, self.roi_max)
        return legacy.crop(box)
