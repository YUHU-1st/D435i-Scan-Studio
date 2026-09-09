from __future__ import annotations

import shutil
import tempfile
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering

from .camera import FramePacket, RealSenseCamera
from .config import AppConfig
from .coverage import CoverageEstimator
from .offline import ReconstructionOutputs, reconstruct_session
from .quality import depth_color_image, depth_quality, quality_image, turbo_colormap
from .realtime import (
    RealtimeReconstructor,
    estimate_roi_center,
    keyframe_needed,
    roi_bounds,
)
from .session import ScanSession


class ScanStudioWindow:
    def __init__(
        self,
        config: AppConfig,
        output_dir: Path,
        bag_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self.bag_path = bag_path
        self.window = gui.Application.instance.create_window("D435i 三维扫描工作站", 1520, 920)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        self.done = False
        self.connect_requested = False
        self.camera_connected = False
        self.scanning = False
        self.finalize_requested = False
        self.busy_reconstructing = False
        self.camera: RealSenseCamera | None = None
        self.reconstructor: RealtimeReconstructor | None = None
        self.coverage: CoverageEstimator | None = None
        self.session: ScanSession | None = None
        self.previous_depth: np.ndarray | None = None
        self.previous_keyframe_pose: np.ndarray | None = None
        self.previous_keyframe_timestamp: float | None = None
        self.frame_count = 0
        self.accepted_count = 0
        self.latest_status = "请连接 D435i"
        self.last_preview_time = 0.0
        self.last_ui_time = 0.0
        self.show_confidence_model = True

        self.panel = self._build_control_panel()
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.035, 0.045, 0.055, 1.0])
        self.scene_widget.scene.show_axes(True)
        self.window.add_child(self.panel)
        self.window.add_child(self.scene_widget)

        self.point_material = rendering.MaterialRecord()
        self.point_material.shader = "defaultUnlit"
        self.point_material.point_size = 3.0
        self.point_material.sRGB_color = True
        self.line_material = rendering.MaterialRecord()
        self.line_material.shader = "unlitLine"
        self.line_material.line_width = 2.0

        self._set_controls()
        threading.Thread(target=self._worker_loop, name="D435iScanWorker", daemon=True).start()

    def _build_control_panel(self) -> gui.Widget:
        em = self.window.theme.font_size
        spacing = int(0.42 * em)
        margin = int(0.55 * em)
        panel = gui.Vert(spacing, gui.Margins(margin))

        title = gui.Label("D435i 高精度物体扫描")
        title.text_color = gui.Color(0.35, 0.85, 1.0)
        panel.add_child(title)

        actions = gui.Horiz(spacing)
        self.connect_button = gui.Button("连接相机")
        self.connect_button.set_on_clicked(self._on_connect)
        self.start_button = gui.Button("开始扫描")
        self.start_button.set_on_clicked(self._on_start)
        self.finish_button = gui.Button("结束并建模")
        self.finish_button.set_on_clicked(self._on_finish)
        actions.add_child(self.connect_button)
        actions.add_child(self.start_button)
        actions.add_child(self.finish_button)
        panel.add_child(actions)

        settings = gui.CollapsableVert("扫描区域与精度", spacing, gui.Margins(0))
        grid = gui.VGrid(2, spacing)
        self.roi_x = self._number_edit(0.08, 2.0, self.config.roi.size_x_m)
        self.roi_y = self._number_edit(0.08, 2.0, self.config.roi.size_y_m)
        self.roi_z = self._number_edit(0.08, 2.0, self.config.roi.size_z_m)
        self.voxel_mm = self._number_edit(1.0, 10.0, self.config.reconstruction.voxel_size_m * 1000)
        self.depth_min = self._number_edit(0.10, 3.0, self.config.camera.depth_min_m)
        self.depth_max = self._number_edit(0.20, 5.0, self.config.camera.depth_max_m)
        for label, editor in (
            ("ROI 宽 X (m)", self.roi_x),
            ("ROI 高 Y (m)", self.roi_y),
            ("ROI 深 Z (m)", self.roi_z),
            ("体素 (mm)", self.voxel_mm),
            ("最近距离 (m)", self.depth_min),
            ("最远距离 (m)", self.depth_max),
        ):
            grid.add_child(gui.Label(label))
            grid.add_child(editor)
        settings.add_child(grid)
        panel.add_child(settings)

        self.confidence_model_checkbox = gui.Checkbox("三维模型显示置信度伪彩")
        self.confidence_model_checkbox.checked = True
        self.confidence_model_checkbox.set_on_checked(self._on_confidence_checked)
        panel.add_child(self.confidence_model_checkbox)

        tabs = gui.TabControl()
        self.color_widget = gui.ImageWidget()
        self.depth_widget = gui.ImageWidget()
        self.quality_widget = gui.ImageWidget()
        self.coverage_widget = gui.ImageWidget()
        tabs.add_tab("彩色", self.color_widget)
        tabs.add_tab("深度", self.depth_widget)
        tabs.add_tab("像素质量", self.quality_widget)
        tabs.add_tab("未扫/已扫", self.coverage_widget)
        panel.add_child(tabs)

        self.progress_bar = gui.ProgressBar()
        self.progress_bar.value = 0.0
        panel.add_child(self.progress_bar)
        self.status_label = gui.Label("初始化中")
        panel.add_child(self.status_label)
        self.stats_label = gui.Label("FPS: --\n关键帧: 0\n覆盖率: 0%\n模型置信度: --")
        panel.add_child(self.stats_label)
        self.tip_label = gui.Label(
            "操作：物体静止，相机缓慢绕行；保持 0.3–0.8 m。\n"
            "红色视角格=未扫描，绿色=已扫描；白框=当前视角。"
        )
        self.tip_label.text_color = gui.Color(0.75, 0.78, 0.82)
        panel.add_child(self.tip_label)
        return panel

    @staticmethod
    def _number_edit(minimum: float, maximum: float, value: float) -> gui.NumberEdit:
        editor = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        editor.set_limits(minimum, maximum)
        editor.double_value = value
        editor.decimal_precision = 3
        return editor

    def _on_layout(self, context: gui.LayoutContext) -> None:
        rect = self.window.content_rect
        panel_width = min(int(31 * context.theme.font_size), int(rect.width * 0.38))
        self.panel.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)
        self.scene_widget.frame = gui.Rect(
            rect.x + panel_width,
            rect.y,
            rect.width - panel_width,
            rect.height,
        )

    def _set_controls(self) -> None:
        self.connect_button.enabled = not self.camera_connected and not self.busy_reconstructing
        self.start_button.enabled = (
            self.camera_connected and not self.scanning and not self.busy_reconstructing
        )
        self.finish_button.enabled = self.scanning and not self.busy_reconstructing
        for editor in (
            self.roi_x,
            self.roi_y,
            self.roi_z,
            self.voxel_mm,
            self.depth_min,
            self.depth_max,
        ):
            editor.enabled = not self.scanning and not self.busy_reconstructing

    def _on_connect(self) -> None:
        self.connect_requested = True
        self.connect_button.enabled = False
        self.status_label.text = "正在连接并预热相机…"

    def _on_confidence_checked(self, checked: bool) -> None:
        self.show_confidence_model = checked

    def _sync_settings(self) -> None:
        self.config.roi.size_x_m = self.roi_x.double_value
        self.config.roi.size_y_m = self.roi_y.double_value
        self.config.roi.size_z_m = self.roi_z.double_value
        self.config.reconstruction.voxel_size_m = self.voxel_mm.double_value / 1000.0
        self.config.camera.depth_min_m = self.depth_min.double_value
        self.config.camera.depth_max_m = self.depth_max.double_value
        if self.config.camera.depth_min_m >= self.config.camera.depth_max_m:
            raise ValueError("最近距离必须小于最远距离")

    def _on_start(self) -> None:
        try:
            self._sync_settings()
        except ValueError as exc:
            self.status_label.text = str(exc)
            return
        self.reconstructor = None
        self.coverage = None
        self.session = None
        self.previous_depth = None
        self.previous_keyframe_pose = None
        self.previous_keyframe_timestamp = None
        self.frame_count = 0
        self.accepted_count = 0
        self.scanning = True
        self.status_label.text = "扫描启动：请缓慢移动相机"
        self._set_controls()

    def _on_finish(self) -> None:
        self.scanning = False
        self.finalize_requested = True
        self.finish_button.enabled = False
        self.status_label.text = "正在保存并进行离线高质量重建…"

    def _on_close(self) -> bool:
        self.done = True
        self.scanning = False
        if self.camera is not None:
            self.camera.stop()
        return True

    def _post(self, callback: object) -> None:
        if not self.done:
            gui.Application.instance.post_to_main_thread(self.window, callback)

    def _worker_loop(self) -> None:
        while not self.done:
            try:
                if self.finalize_requested and not self.busy_reconstructing:
                    self._finalize()
                elif self.connect_requested and self.camera is None:
                    self._connect_camera()
                elif self.camera is not None and not self.busy_reconstructing:
                    started = time.perf_counter()
                    frame = self.camera.read()
                    elapsed = max(time.perf_counter() - started, 1e-6)
                    self._process_frame(frame, 1.0 / elapsed)
                else:
                    time.sleep(0.04)
            except Exception as exc:  # noqa: BLE001 - keep the capture GUI alive
                detail = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                self.latest_status = detail
                self.scanning = False
                self.finalize_requested = False
                if self.camera is not None and not self.busy_reconstructing:
                    self.camera.stop()
                    self.camera = None
                    self.camera_connected = False
                self._post(lambda d=detail: self._show_error(d))
                time.sleep(0.5)

    def _connect_camera(self) -> None:
        self.connect_requested = False
        camera = RealSenseCamera(self.config.camera, self.bag_path)
        camera.start()
        self.camera = camera
        self.camera_connected = True
        device_text = (
            f"已连接：{camera.device_name} | USB {camera.usb_type} | "
            f"{camera.active_stream}"
        )
        if camera.degraded_mode:
            device_text += " | 警告：USB 2 降级模式，仅用于功能测试"
        self.latest_status = device_text
        self._post(lambda: self._on_connected_ui(device_text))

    def _on_connected_ui(self, text: str) -> None:
        self.status_label.text = text
        self._set_controls()

    def _process_frame(self, frame: FramePacket, capture_fps: float) -> None:
        score = depth_quality(
            frame.depth_raw,
            frame.depth_unit_m,
            self.config.camera.depth_min_m,
            self.config.camera.depth_max_m,
            self.previous_depth,
        )
        self.previous_depth = frame.depth_raw.copy()
        color_preview = self._decorate_color(frame.color_rgb)
        depth_preview = depth_color_image(
            frame.depth_raw,
            frame.depth_unit_m,
            self.config.camera.depth_min_m,
            self.config.camera.depth_max_m,
        )
        quality_preview = quality_image(score)
        preview_cloud = None
        pose = np.eye(4)
        tracking_fitness = 0.0
        tracking_rmse = 0.0
        tracking_score = 0.0

        if self.scanning:
            if self.reconstructor is None:
                center = estimate_roi_center(frame, self.config.roi.center_crop_fraction)
                self.reconstructor = RealtimeReconstructor(self.config, frame.intrinsics, center)
                self.coverage = CoverageEstimator(self.config.reconstruction.voxel_size_m * 2.0)
                self.coverage.set_center(center)
                self.session = ScanSession(self.output_dir, self.config)
                self.session.set_geometry(frame, center)
                self._make_roi_geometry(center)

            request_preview = (
                self.frame_count % self.config.reconstruction.preview_every_n_frames == 0
            )
            result = self.reconstructor.process(frame, request_preview)
            self.frame_count += 1
            pose = result.camera_to_world
            tracking_fitness = result.fitness
            tracking_rmse = result.rmse_m
            tracking_score = result.tracking_score
            if result.accepted:
                self.accepted_count += 1
                assert self.coverage is not None
                self.coverage.update_viewpoint(pose, tracking_score)
                if keyframe_needed(
                    self.previous_keyframe_pose,
                    pose,
                    self.previous_keyframe_timestamp,
                    frame.timestamp_s,
                    self.config,
                ):
                    roi_min, roi_max = roi_bounds(self.coverage.center_world, self.config)
                    self.coverage.update_surface(
                        frame.depth_raw,
                        score,
                        frame.intrinsics.matrix,
                        pose,
                        frame.depth_unit_m,
                        self.config.camera.depth_min_m,
                        self.config.camera.depth_max_m,
                        stride=5,
                        roi_min=roi_min,
                        roi_max=roi_max,
                    )
                    assert self.session is not None
                    self.session.add_keyframe(
                        frame,
                        pose,
                        tracking_fitness,
                        tracking_rmse,
                        float(np.mean(score[score > 0])) if np.any(score > 0) else 0.0,
                    )
                    self.previous_keyframe_pose = pose.copy()
                    self.previous_keyframe_timestamp = frame.timestamp_s
            self.latest_status = result.reason or "跟踪正常；请继续覆盖红色视角"
            preview_cloud = result.preview_cloud
            if preview_cloud is not None and self.coverage is not None:
                points = np.asarray(preview_cloud.points)
                if self.show_confidence_model:
                    values = self.coverage.confidences_for_points(points)
                    preview_cloud.colors = o3d.utility.Vector3dVector(turbo_colormap(values))

        coverage_image = (
            self.coverage.coverage_image()
            if self.coverage is not None
            else np.full((154, 432, 3), [120, 20, 20], dtype=np.uint8)
        )
        keyframes = len(self.session.frames) if self.session is not None else 0
        coverage_percent = self.coverage.coverage_percent if self.coverage is not None else 0.0
        mean_confidence = float(np.mean(score[score > 0])) if np.any(score > 0) else 0.0
        now = time.monotonic()
        if now - self.last_ui_time >= 1.0 / 12.0:
            self.last_ui_time = now
            self._post(
                lambda: self._update_ui(
                    color_preview,
                    depth_preview,
                    quality_preview,
                    coverage_image,
                    preview_cloud,
                    pose,
                    capture_fps,
                    keyframes,
                    coverage_percent,
                    mean_confidence,
                    tracking_fitness,
                    tracking_rmse,
                )
            )

    @staticmethod
    def _decorate_color(color_rgb: np.ndarray) -> np.ndarray:
        image = color_rgb.copy()
        height, width = image.shape[:2]
        cx, cy = width // 2, height // 2
        image[max(0, cy - 16) : min(height, cy + 17), max(0, cx - 1) : cx + 2] = [255, 255, 255]
        image[max(0, cy - 1) : cy + 2, max(0, cx - 16) : min(width, cx + 17)] = [255, 255, 255]
        return np.ascontiguousarray(image)

    def _make_roi_geometry(self, center: np.ndarray) -> None:
        minimum, maximum = roi_bounds(center, self.config)
        box = o3d.geometry.AxisAlignedBoundingBox(minimum, maximum)
        lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
        lines.paint_uniform_color([0.1, 0.8, 1.0])
        self._post(lambda: self._show_roi(lines, box, center))

    def _show_roi(self, lines: object, box: object, center: np.ndarray) -> None:
        self.scene_widget.scene.remove_geometry("roi")
        self.scene_widget.scene.add_geometry("roi", lines, self.line_material)
        self.scene_widget.setup_camera(60.0, box, center)

    def _update_ui(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        quality: np.ndarray,
        coverage: np.ndarray,
        cloud: object | None,
        camera_to_world: np.ndarray,
        fps: float,
        keyframes: int,
        coverage_percent: float,
        mean_confidence: float,
        fitness: float,
        rmse: float,
    ) -> None:
        if self.done:
            return
        self.color_widget.update_image(o3d.geometry.Image(color))
        self.depth_widget.update_image(o3d.geometry.Image(depth))
        self.quality_widget.update_image(o3d.geometry.Image(quality))
        self.coverage_widget.update_image(o3d.geometry.Image(coverage))
        if cloud is not None and len(cloud.points) > 0:
            self.scene_widget.scene.remove_geometry("model")
            self.scene_widget.scene.add_geometry("model", cloud, self.point_material)
        if self.reconstructor is not None:
            frustum = o3d.geometry.LineSet.create_camera_visualization(
                self.reconstructor.intrinsics.width,
                self.reconstructor.intrinsics.height,
                self.reconstructor.intrinsics.matrix,
                np.linalg.inv(camera_to_world),
                0.08,
            )
            frustum.paint_uniform_color([1.0, 0.55, 0.05])
            self.scene_widget.scene.remove_geometry("camera")
            self.scene_widget.scene.add_geometry("camera", frustum, self.line_material)
        self.status_label.text = self.latest_status
        self.stats_label.text = (
            f"采集 FPS: {fps:5.1f}\n"
            f"关键帧: {keyframes}\n"
            f"视角覆盖: {coverage_percent:5.1f}%\n"
            f"像素质量: {mean_confidence:5.1%}\n"
            f"跟踪: {fitness:.3f} / {rmse * 1000:.1f} mm"
        )

    def _finalize(self) -> None:
        self.finalize_requested = False
        if self.session is None or len(self.session.frames) < 3:
            self._post(lambda: self._show_error("有效关键帧不足 3 个，无法建模"))
            return
        self.busy_reconstructing = True
        self.session.finish()
        session_path = self.session.path
        if self.camera is not None:
            self.camera.stop()
            self.camera = None
            self.camera_connected = False
        self.reconstructor = None
        self._post(self._set_controls)

        def report_progress(value: float, message: str) -> None:
            self._post(lambda v=value, m=message: self._show_progress(v, m))

        try:
            outputs = reconstruct_session(session_path, self.config, report_progress)
            self._post(lambda o=outputs: self._show_completed(o))
        finally:
            self.busy_reconstructing = False
            self._post(self._set_controls)

    def _show_progress(self, value: float, message: str) -> None:
        self.progress_bar.value = value
        self.status_label.text = message

    def _show_completed(self, outputs: ReconstructionOutputs) -> None:
        self.progress_bar.value = 1.0
        step_text = "已生成" if outputs.step_path else "未生成（见报告）"
        self.status_label.text = f"完成：OBJ/PLY 已生成；STEP {step_text}"
        final_mesh = o3d.io.read_triangle_mesh(str(outputs.confidence_ply_path))
        if len(final_mesh.triangles) > 0:
            self.scene_widget.scene.remove_geometry("model")
            material = rendering.MaterialRecord()
            material.shader = "defaultLit"
            self.scene_widget.scene.add_geometry("model", final_mesh, material)

    def _show_error(self, detail: str) -> None:
        self.status_label.text = detail
        self.progress_bar.value = 0.0
        self._set_controls()


def run_gui(config: AppConfig, output_dir: Path, bag_path: str | Path | None = None) -> None:
    app = gui.Application.instance
    resource_dir = Path(o3d.__file__).resolve().parent / "resources"
    # Open3D 0.19's Windows renderer uses a narrow path internally. When the
    # virtual environment lives below a non-ASCII workspace, pass it an ASCII
    # cache path so the Filament resources can still be opened.
    if any(ord(char) > 127 for char in str(resource_dir)):
        cached_resources = Path(tempfile.gettempdir()) / "d435i_scan_o3d_resources"
        marker = cached_resources / "defaultLit.filamat"
        if not marker.exists():
            shutil.copytree(resource_dir, cached_resources, dirs_exist_ok=True)
        resource_dir = cached_resources
    app.initialize(str(resource_dir))
    chinese_font = Path("C:/Windows/Fonts/msyh.ttc")
    if chinese_font.exists():
        font = gui.FontDescription()
        font.add_typeface_for_language(str(chinese_font), "zh")
        app.set_font(gui.Application.DEFAULT_FONT_ID, font)
    ScanStudioWindow(config, output_dir, bag_path)
    app.run()
