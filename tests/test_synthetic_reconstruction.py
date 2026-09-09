from __future__ import annotations

import numpy as np
import pytest

from d435i_scan.camera import CameraIntrinsics, FramePacket
from d435i_scan.config import AppConfig
from d435i_scan.offline import reconstruct_session
from d435i_scan.realtime import RealtimeReconstructor
from d435i_scan.session import ScanSession


def _look_at_pose(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, -1.0, 0.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, down, forward])
    pose[:3, 3] = position
    return pose


def _render_sphere(intr: CameraIntrinsics, pose: np.ndarray, radius: float = 0.08):
    ys, xs = np.mgrid[: intr.height, : intr.width]
    rays_camera = np.stack(
        [(xs - intr.cx) / intr.fx, (ys - intr.cy) / intr.fy, np.ones_like(xs)], axis=-1
    )
    rays_world = rays_camera @ pose[:3, :3].T
    origin = pose[:3, 3]
    b = 2.0 * np.sum(rays_world * origin, axis=-1)
    c = float(origin @ origin - radius * radius)
    discriminant = b * b - 4.0 * np.sum(rays_world * rays_world, axis=-1) * c
    valid = discriminant >= 0.0
    a = np.sum(rays_world * rays_world, axis=-1)
    t = np.zeros_like(discriminant, dtype=np.float64)
    t[valid] = (-b[valid] - np.sqrt(discriminant[valid])) / (2.0 * a[valid])
    valid &= t > 0.0
    depth = np.zeros((intr.height, intr.width), dtype=np.uint16)
    depth[valid] = np.round(t[valid] * 1000.0).astype(np.uint16)
    points = origin + rays_world * t[..., None]
    normals = np.clip((points / radius + 1.0) * 0.5, 0.0, 1.0)
    color = np.full((intr.height, intr.width, 3), 20, dtype=np.uint8)
    color[valid] = (normals[valid] * 255.0).astype(np.uint8)
    return color, depth


@pytest.mark.integration
def test_synthetic_offline_reconstruction(work_dir):
    pytest.importorskip("open3d")
    config = AppConfig(output_dir=str(work_dir))
    config.camera.depth_min_m = 0.10
    config.camera.depth_max_m = 0.60
    config.roi.size_x_m = config.roi.size_y_m = config.roi.size_z_m = 0.24
    config.roi.margin_m = 0.0
    config.reconstruction.voxel_size_m = 0.006
    config.reconstruction.icp_max_correspondence_m = 0.02
    config.reconstruction.smooth_iterations = 2
    config.reconstruction.target_triangles = 10_000
    config.export.export_step = False
    intr = CameraIntrinsics(96, 72, 105.0, 105.0, 47.5, 35.5)
    session = ScanSession(work_dir, config)

    for index, angle in enumerate(np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)):
        position = np.array([0.26 * np.sin(angle), 0.025 * np.sin(2 * angle), 0.26 * np.cos(angle)])
        pose = _look_at_pose(position, np.zeros(3))
        color, depth = _render_sphere(intr, pose)
        frame = FramePacket(color, depth, float(index), index, intr, 0.001)
        if index == 0:
            session.set_geometry(frame, np.zeros(3))
        session.add_keyframe(frame, pose, 0.95, 0.001, 0.95)
    session.finish()

    outputs = reconstruct_session(session.path, config)
    assert outputs.obj_path and outputs.obj_path.is_file()
    assert outputs.ply_path and outputs.ply_path.is_file()
    assert outputs.confidence_ply_path and outputs.confidence_ply_path.is_file()
    assert outputs.report["mesh"]["triangles"] > 100


@pytest.mark.integration
def test_realtime_tsdf_smoke():
    pytest.importorskip("open3d")
    config = AppConfig()
    config.camera.depth_min_m = 0.10
    config.camera.depth_max_m = 0.50
    config.roi.size_x_m = config.roi.size_y_m = config.roi.size_z_m = 0.25
    config.reconstruction.voxel_size_m = 0.006
    config.reconstruction.block_count = 1_000
    config.reconstruction.surface_weight_threshold = 1.0
    intr = CameraIntrinsics(96, 72, 105.0, 105.0, 47.5, 35.5)
    pose = _look_at_pose(np.array([0.0, 0.0, 0.26]), np.zeros(3))
    color, depth = _render_sphere(intr, pose)
    frame = FramePacket(color, depth, 0.0, 0, intr, 0.001)
    reconstructor = RealtimeReconstructor(config, intr, np.array([0.0, 0.0, 0.26]))

    first = reconstructor.process(frame, request_preview=True)
    second = reconstructor.process(frame, request_preview=True)
    assert first.accepted
    # A perfect sphere has no rotational constraints and may be rejected as a
    # singular tracking problem, but it must not terminate the scan worker.
    assert isinstance(second.accepted, bool)
    assert second.preview_cloud is not None
