from __future__ import annotations

import copy
import gc
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .camera import CameraIntrinsics
from .config import AppConfig
from .coverage import CoverageEstimator
from .quality import depth_quality, turbo_colormap
from .realtime import roi_bounds
from .session import load_session
from .step_export import export_faceted_step

ProgressCallback = Callable[[float, str], None]


@dataclass(slots=True)
class ReconstructionOutputs:
    session_dir: Path
    obj_path: Path | None
    ply_path: Path | None
    confidence_ply_path: Path | None
    step_path: Path | None
    report_path: Path
    report: dict[str, object]


def _progress(callback: ProgressCallback | None, fraction: float, message: str) -> None:
    if callback is not None:
        callback(float(np.clip(fraction, 0.0, 1.0)), message)


def _load_rgbd_point_cloud(
    o3d: object,
    session_path: Path,
    item: dict[str, object],
    intrinsics: CameraIntrinsics,
    depth_scale: float,
    depth_min_m: float,
    depth_max_m: float,
) -> tuple[object, np.ndarray, np.ndarray]:
    color = np.asarray(Image.open(session_path / str(item["color"])).convert("RGB"))
    depth = np.asarray(Image.open(session_path / str(item["depth"])), dtype=np.uint16)
    depth = depth.copy()
    depth_m = depth.astype(np.float32) / depth_scale
    depth[(depth_m < depth_min_m) | (depth_m > depth_max_m)] = 0
    color_image = o3d.geometry.Image(np.ascontiguousarray(color))
    depth_image = o3d.geometry.Image(np.ascontiguousarray(depth))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image,
        depth_image,
        depth_scale=depth_scale,
        depth_trunc=depth_max_m,
        convert_rgb_to_intensity=False,
    )
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width,
        intrinsics.height,
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.cx,
        intrinsics.cy,
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    return cloud, color, depth


def _crop_cloud_in_world(
    o3d: object,
    cloud_local: object,
    camera_to_world: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> object:
    world = copy.deepcopy(cloud_local)
    world.transform(camera_to_world)
    cropped_world = world.crop(o3d.geometry.AxisAlignedBoundingBox(minimum, maximum))
    cropped_world.transform(np.linalg.inv(camera_to_world))
    return cropped_world


def _register_pair(
    o3d: object,
    source: object,
    target: object,
    initial: np.ndarray,
    max_distance: float,
) -> tuple[object, np.ndarray]:
    loss = o3d.pipelines.registration.TukeyLoss(k=max_distance * 0.7)
    estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_distance,
        initial,
        estimator,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=45),
    )
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source,
        target,
        max_distance,
        result.transformation,
    )
    return result, information


def _clean_mesh(o3d: object, mesh: object, config: AppConfig) -> tuple[object, dict[str, object]]:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    removed_components = 0
    if len(mesh.triangles) > 0:
        labels, triangle_counts, areas = mesh.cluster_connected_triangles()
        labels_array = np.asarray(labels)
        counts_array = np.asarray(triangle_counts)
        areas_array = np.asarray(areas)
        if len(counts_array) > 1:
            keep_min_count = max(60, int(np.max(counts_array) * 0.008))
            keep_min_area = float(np.max(areas_array) * 0.004)
            keep_clusters = np.flatnonzero(
                (counts_array >= keep_min_count) & (areas_array >= keep_min_area)
            )
            remove_mask = ~np.isin(labels_array, keep_clusters)
            removed_components = int(len(counts_array) - len(keep_clusters))
            mesh.remove_triangles_by_mask(remove_mask)
            mesh.remove_unreferenced_vertices()

    target = config.reconstruction.target_triangles
    if target > 0 and len(mesh.triangles) > target:
        mesh = mesh.simplify_quadric_decimation(target)
    if config.reconstruction.smooth_iterations > 0 and len(mesh.triangles) > 0:
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=config.reconstruction.smooth_iterations,
            lambda_filter=0.45,
            mu=-0.47,
        )
    mesh.compute_vertex_normals()
    mesh.normalize_normals()
    self_intersecting: bool | None = None
    if len(mesh.triangles) <= 100_000:
        self_intersecting = bool(mesh.is_self_intersecting())
    metrics = {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "removed_small_components": removed_components,
        "edge_manifold_closed": bool(mesh.is_edge_manifold(allow_boundary_edges=False)),
        "edge_manifold_with_boundary": bool(mesh.is_edge_manifold(allow_boundary_edges=True)),
        "vertex_manifold": bool(mesh.is_vertex_manifold()),
        "self_intersecting": self_intersecting,
        "watertight": bool(mesh.is_watertight()),
        "orientable": bool(mesh.is_orientable()),
    }
    return mesh, metrics


def reconstruct_session(
    session_dir: str | Path,
    config: AppConfig | None = None,
    progress: ProgressCallback | None = None,
) -> ReconstructionOutputs:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("缺少 Open3D；请先运行 setup.ps1") from exc

    session_path = Path(session_dir).resolve()
    manifest = load_session(session_path)
    if config is None:
        config = AppConfig.load(session_path / "settings.yml")
    frame_items = list(manifest.get("frames", []))
    if len(frame_items) < 3:
        raise RuntimeError("至少需要 3 个关键帧才能重建")
    if manifest.get("intrinsics") is None or manifest.get("roi_center_world") is None:
        raise RuntimeError("session.json 缺少内参或 ROI 信息")

    intrinsics = CameraIntrinsics.from_dict(manifest["intrinsics"])
    depth_unit_m = float(manifest["depth_unit_m"])
    depth_scale = 1.0 / depth_unit_m
    center = np.asarray(manifest["roi_center_world"], dtype=np.float64)
    roi_min, roi_max = roi_bounds(center, config)
    voxel = config.reconstruction.voxel_size_m

    _progress(progress, 0.02, "读取关键帧并生成局部点云")
    clouds: list[object] = []
    initial_poses: list[np.ndarray] = []
    for index, item in enumerate(frame_items):
        cloud, _, _ = _load_rgbd_point_cloud(
            o3d,
            session_path,
            item,
            intrinsics,
            depth_scale,
            config.camera.depth_min_m,
            config.camera.depth_max_m,
        )
        pose = np.asarray(item["camera_to_world"], dtype=np.float64)
        cloud = _crop_cloud_in_world(o3d, cloud, pose, roi_min, roi_max)
        cloud = cloud.voxel_down_sample(max(voxel * 1.7, 0.003))
        if len(cloud.points) > 30:
            cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.2)
            cloud.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel * 5.0, 0.012), max_nn=35)
            )
        clouds.append(cloud)
        initial_poses.append(pose)
        _progress(
            progress,
            0.02 + 0.20 * (index + 1) / len(frame_items),
            f"点云 {index + 1}/{len(frame_items)}",
        )

    _progress(progress, 0.23, "构建并优化位姿图")
    pose_graph = o3d.pipelines.registration.PoseGraph()
    for pose in initial_poses:
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose))

    accepted_loops = 0
    rejected_edges = 0
    max_corr = config.reconstruction.icp_max_correspondence_m
    candidate_pairs: list[tuple[int, int, bool]] = []
    for index in range(len(clouds) - 1):
        candidate_pairs.append((index, index + 1, False))
    centers = np.asarray([pose[:3, 3] for pose in initial_poses])
    for source in range(len(clouds)):
        candidates: list[tuple[float, int]] = []
        for target in range(
            source + config.reconstruction.loop_closure_min_separation, len(clouds)
        ):
            distance = float(np.linalg.norm(centers[source] - centers[target]))
            if distance <= config.reconstruction.loop_closure_distance_m:
                candidates.append((distance, target))
        for _, target in sorted(candidates)[: config.reconstruction.loop_closure_max_per_frame]:
            candidate_pairs.append((source, target, True))

    for edge_index, (source_id, target_id, uncertain) in enumerate(candidate_pairs):
        source, target = clouds[source_id], clouds[target_id]
        initial = np.linalg.inv(initial_poses[target_id]) @ initial_poses[source_id]
        if len(source.points) < 30 or len(target.points) < 30:
            rejected_edges += 1
            continue
        result, information = _register_pair(o3d, source, target, initial, max_corr)
        threshold = 0.18 if uncertain else 0.08
        if result.fitness < threshold:
            rejected_edges += 1
            if uncertain:
                continue
            transformation = initial
        else:
            transformation = result.transformation
            if uncertain:
                accepted_loops += 1
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source_id,
                target_id,
                transformation,
                information,
                uncertain,
            )
        )
        _progress(
            progress,
            0.23 + 0.22 * (edge_index + 1) / max(len(candidate_pairs), 1),
            f"配准边 {edge_index + 1}/{len(candidate_pairs)}",
        )

    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_corr,
            edge_prune_threshold=0.25,
            preference_loop_closure=1.5,
            reference_node=0,
        ),
    )
    optimized_poses = [np.asarray(node.pose) for node in pose_graph.nodes]
    del clouds
    gc.collect()

    _progress(progress, 0.47, "高分辨率 TSDF 重融合")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=voxel * config.reconstruction.trunc_voxel_multiplier,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    intrinsic_legacy = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width,
        intrinsics.height,
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.cx,
        intrinsics.cy,
    )
    coverage = CoverageEstimator(voxel_size_m=voxel * 2.0)
    coverage.set_center(center)
    previous_depth: np.ndarray | None = None
    for index, (item, pose) in enumerate(zip(frame_items, optimized_poses)):
        color = np.asarray(Image.open(session_path / str(item["color"])).convert("RGB"))
        depth = np.asarray(Image.open(session_path / str(item["depth"])), dtype=np.uint16)
        depth = depth.copy()
        depth_m = depth.astype(np.float32) * depth_unit_m
        depth[(depth_m < config.camera.depth_min_m) | (depth_m > config.camera.depth_max_m)] = 0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color)),
            o3d.geometry.Image(np.ascontiguousarray(depth)),
            depth_scale=depth_scale,
            depth_trunc=config.camera.depth_max_m,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic_legacy, np.linalg.inv(pose))
        confidence = depth_quality(
            depth,
            depth_unit_m,
            config.camera.depth_min_m,
            config.camera.depth_max_m,
            previous_depth,
        )
        coverage.update_viewpoint(pose, float(frame_items[index]["tracking_fitness"]))
        coverage.update_surface(
            depth,
            confidence,
            intrinsics.matrix,
            pose,
            depth_unit_m,
            config.camera.depth_min_m,
            config.camera.depth_max_m,
            stride=5,
            roi_min=roi_min,
            roi_max=roi_max,
        )
        previous_depth = depth
        _progress(
            progress,
            0.47 + 0.28 * (index + 1) / len(frame_items),
            f"融合 {index + 1}/{len(frame_items)}",
        )

    raw_mesh = volume.extract_triangle_mesh()
    raw_mesh.compute_vertex_normals()
    raw_mesh = raw_mesh.crop(o3d.geometry.AxisAlignedBoundingBox(roi_min, roi_max))
    raw_path = session_path / "model_raw.ply"
    o3d.io.write_triangle_mesh(str(raw_path), raw_mesh, write_ascii=False)

    _progress(progress, 0.77, "网格清理、去噪和平滑")
    mesh, metrics = _clean_mesh(o3d, raw_mesh, config)
    if len(mesh.triangles) == 0:
        raise RuntimeError("ROI 内没有可导出的表面；请检查扫描距离和 ROI 尺寸")

    ply_path = session_path / "model.ply" if config.export.export_ply else None
    obj_path = session_path / "model.obj" if config.export.export_obj else None
    if ply_path is not None:
        o3d.io.write_triangle_mesh(str(ply_path), mesh, write_ascii=False)
    if obj_path is not None:
        o3d.io.write_triangle_mesh(
            str(obj_path),
            mesh,
            write_ascii=False,
            write_vertex_normals=True,
            write_vertex_colors=True,
        )

    _progress(progress, 0.84, "生成模型置信度伪彩")
    confidence_mesh = copy.deepcopy(mesh)
    vertex_confidence = coverage.confidences_for_points(np.asarray(mesh.vertices))
    confidence_mesh.vertex_colors = o3d.utility.Vector3dVector(turbo_colormap(vertex_confidence))
    confidence_path = session_path / "model_confidence.ply"
    o3d.io.write_triangle_mesh(str(confidence_path), confidence_mesh, write_ascii=False)
    Image.fromarray(coverage.coverage_image()).save(session_path / "view_coverage.png")

    step_path: Path | None = None
    step_result: dict[str, object] = {"ok": False, "reason": "配置中已关闭 STEP 导出"}
    if config.export.export_step:
        _progress(progress, 0.90, "通过 FreeCAD/OpenCascade 导出分面 STEP")
        step_mesh = copy.deepcopy(mesh)
        if len(step_mesh.triangles) > config.export.step_max_triangles:
            step_mesh = step_mesh.simplify_quadric_decimation(config.export.step_max_triangles)
        # OBJ/PLY coordinates remain in metres. STL is unitless and FreeCAD reads
        # its coordinates as millimetres, so scale explicitly for a metric STEP.
        step_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(step_mesh.vertices) * 1000.0)
        step_mesh.compute_triangle_normals()
        step_source = session_path / "model_for_step.stl"
        o3d.io.write_triangle_mesh(str(step_source), step_mesh, write_ascii=False)
        candidate_step = session_path / "model.step"
        step_result = export_faceted_step(
            step_source,
            candidate_step,
            config.export.freecad_cmd,
            config.export.step_tolerance_mm,
        )
        if bool(step_result.get("ok")):
            step_path = candidate_step

    warnings: list[str] = []
    if coverage.coverage_percent < 55.0:
        warnings.append("视角覆盖低于 55%，模型背面或顶部很可能缺失。")
    if not bool(metrics["watertight"]):
        warnings.append("网格不是水密实体；分面 STEP 可能是壳体而非 Solid。")
    if not bool(step_result.get("ok")) and config.export.export_step:
        warnings.append(str(step_result.get("reason", "STEP 导出失败")))
    warnings.append("STEP 为三角面 B-Rep；若需要可编辑机械 CAD，请另做平面/圆柱/NURBS 逆向拟合。")

    report: dict[str, object] = {
        "session": str(session_path),
        "keyframes": len(frame_items),
        "accepted_loop_closures": accepted_loops,
        "rejected_registration_edges": rejected_edges,
        "view_coverage_percent": coverage.coverage_percent,
        "mean_surface_confidence": float(np.mean(vertex_confidence)),
        "mesh": metrics,
        "outputs": {
            "raw_ply": str(raw_path),
            "ply": str(ply_path) if ply_path else None,
            "obj": str(obj_path) if obj_path else None,
            "confidence_ply": str(confidence_path),
            "step": str(step_path) if step_path else None,
        },
        "units": {"obj": "metre (unitless convention)", "ply": "metre", "step": "millimetre"},
        "step_export": step_result,
        "warnings": warnings,
    }
    report_path = session_path / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _progress(progress, 1.0, "重建完成")
    return ReconstructionOutputs(
        session_dir=session_path,
        obj_path=obj_path,
        ply_path=ply_path,
        confidence_ply_path=confidence_path,
        step_path=step_path,
        report_path=report_path,
        report=report,
    )
