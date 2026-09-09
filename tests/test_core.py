from __future__ import annotations

import json

import numpy as np

from d435i_scan.camera import CameraIntrinsics, FramePacket
from d435i_scan.config import AppConfig
from d435i_scan.coverage import CoverageEstimator
from d435i_scan.quality import depth_quality, turbo_colormap
from d435i_scan.realtime import keyframe_needed
from d435i_scan.session import ScanSession, load_session


def test_config_round_trip(work_dir):
    config = AppConfig()
    config.roi.size_x_m = 0.31
    target = work_dir / "settings.yml"
    config.save(target)
    loaded = AppConfig.load(target)
    assert loaded.roi.size_x_m == 0.31
    assert loaded.camera.width == 848


def test_depth_quality_and_colormap():
    depth = np.full((8, 8), 400, dtype=np.uint16)
    depth[0, 0] = 0
    depth[4, 4] = 900
    score = depth_quality(depth, 0.001, 0.2, 1.0, depth.copy())
    assert score.shape == depth.shape
    assert score[0, 0] == 0.0
    assert score[3, 3] > score[4, 4]
    colors = turbo_colormap(score)
    assert colors.shape == (8, 8, 3)
    assert np.all((colors >= 0.0) & (colors <= 1.0))


def test_coverage_updates_view_and_surface():
    coverage = CoverageEstimator(0.01, azimuth_bins=12, elevation_bins=5)
    coverage.set_center(np.array([0.0, 0.0, 0.5]))
    pose = np.eye(4)
    coverage.update_viewpoint(pose, 0.9)
    assert coverage.coverage_percent > 0.0
    image = coverage.coverage_image(4, 3)
    assert image.shape == (15, 48, 3)

    depth = np.full((10, 12), 500, dtype=np.uint16)
    quality = np.full((10, 12), 0.8, dtype=np.float32)
    intrinsic = np.array([[100.0, 0.0, 6.0], [0.0, 100.0, 5.0], [0.0, 0.0, 1.0]])
    coverage.update_surface(depth, quality, intrinsic, pose, 0.001, 0.2, 1.0, stride=2)
    assert coverage.evidence
    confidence = coverage.confidences_for_points(np.array([[0.0, 0.0, 0.5]]))
    assert confidence[0] > 0.03


def test_keyframe_thresholds():
    config = AppConfig()
    identity = np.eye(4)
    assert keyframe_needed(None, identity, None, 0.0, config)
    assert not keyframe_needed(identity, identity, 0.0, 0.1, config)
    moved = identity.copy()
    moved[0, 3] = config.reconstruction.keyframe_translation_m * 1.1
    assert keyframe_needed(identity, moved, 0.0, 0.1, config)


def test_scan_session_writes_lossless_depth(work_dir):
    config = AppConfig(output_dir=str(work_dir))
    intrinsics = CameraIntrinsics(8, 6, 50.0, 50.0, 4.0, 3.0)
    frame = FramePacket(
        color_rgb=np.full((6, 8, 3), 127, dtype=np.uint8),
        depth_raw=np.full((6, 8), 432, dtype=np.uint16),
        timestamp_s=1.25,
        frame_number=7,
        intrinsics=intrinsics,
        depth_unit_m=0.001,
    )
    session = ScanSession(work_dir, config)
    session.set_geometry(frame, np.array([0.0, 0.0, 0.43]))
    session.add_keyframe(frame, np.eye(4), 0.8, 0.002, 0.9)
    session.finish()
    manifest = load_session(session.path)
    assert manifest["finished"] is True
    assert manifest["frames"][0]["source_frame"] == 7
    assert (session.path / manifest["frames"][0]["depth"]).is_file()
    json.dumps(manifest)
