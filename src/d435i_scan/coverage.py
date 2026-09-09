from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quality import turbo_colormap


@dataclass(slots=True)
class VoxelEvidence:
    observations: int = 0
    quality_sum: float = 0.0
    view_mask: int = 0


class CoverageEstimator:
    """Tracks viewpoint coverage plus per-surface-voxel evidence."""

    def __init__(
        self,
        voxel_size_m: float,
        azimuth_bins: int = 24,
        elevation_bins: int = 7,
    ) -> None:
        self.voxel_size_m = float(voxel_size_m)
        self.azimuth_bins = int(azimuth_bins)
        self.elevation_bins = int(elevation_bins)
        self.view_scores = np.zeros((elevation_bins, azimuth_bins), dtype=np.float32)
        self.evidence: dict[tuple[int, int, int], VoxelEvidence] = {}
        self.center_world: np.ndarray | None = None
        self.current_view: tuple[int, int] | None = None

    def set_center(self, center_world: np.ndarray) -> None:
        self.center_world = np.asarray(center_world, dtype=np.float64).reshape(3)

    def update_viewpoint(self, camera_to_world: np.ndarray, tracking_score: float) -> None:
        if self.center_world is None:
            return
        camera = np.asarray(camera_to_world, dtype=np.float64)[:3, 3]
        offset = camera - self.center_world
        radius = float(np.linalg.norm(offset))
        if radius < 1e-6:
            return
        azimuth = np.arctan2(offset[0], offset[2])
        elevation = np.arcsin(np.clip(offset[1] / radius, -1.0, 1.0))
        az_idx = int(np.floor((azimuth + np.pi) / (2.0 * np.pi) * self.azimuth_bins))
        az_idx %= self.azimuth_bins
        # The useful object-scanning band is -60..+60 degrees.
        el_norm = np.clip((elevation + np.pi / 3.0) / (2.0 * np.pi / 3.0), 0.0, 0.999999)
        el_idx = int(np.floor(el_norm * self.elevation_bins))
        self.current_view = (el_idx, az_idx)

        score = float(np.clip(tracking_score, 0.0, 1.0))
        self.view_scores[el_idx, az_idx] = max(self.view_scores[el_idx, az_idx], score)
        for da, weight in ((-1, 0.35), (1, 0.35)):
            neighbor = (az_idx + da) % self.azimuth_bins
            self.view_scores[el_idx, neighbor] = max(
                self.view_scores[el_idx, neighbor], score * weight
            )

    @property
    def coverage_percent(self) -> float:
        # A bin at 0.5 or higher is considered deliberately observed.
        return float(np.mean(self.view_scores >= 0.5) * 100.0)

    def coverage_image(self, cell_width: int = 18, cell_height: int = 22) -> np.ndarray:
        """Return an equirectangular red/amber/green viewpoint coverage chart."""
        values = np.clip(self.view_scores, 0.0, 1.0)
        red = np.clip(1.4 - 1.5 * values, 0.0, 1.0)
        green = np.clip(1.7 * values, 0.0, 1.0)
        blue = np.full_like(values, 0.08)
        cells = np.stack([red, green, blue], axis=-1)
        cells = (cells * 255.0).astype(np.uint8)
        image = np.repeat(np.repeat(cells, cell_height, axis=0), cell_width, axis=1)

        # Thin grid lines make individual requested viewpoints explicit.
        image[::cell_height, :, :] = 28
        image[:, ::cell_width, :] = 28
        if self.current_view is not None:
            row, column = self.current_view
            y0, y1 = row * cell_height, (row + 1) * cell_height
            x0, x1 = column * cell_width, (column + 1) * cell_width
            image[y0 : y0 + 2, x0:x1] = 255
            image[y1 - 2 : y1, x0:x1] = 255
            image[y0:y1, x0 : x0 + 2] = 255
            image[y0:y1, x1 - 2 : x1] = 255
        return np.ascontiguousarray(image)

    def update_surface(
        self,
        depth_raw: np.ndarray,
        quality: np.ndarray,
        intrinsics: np.ndarray,
        camera_to_world: np.ndarray,
        depth_unit_m: float,
        depth_min_m: float,
        depth_max_m: float,
        stride: int = 5,
        roi_min: np.ndarray | None = None,
        roi_max: np.ndarray | None = None,
    ) -> None:
        ys, xs = np.mgrid[0 : depth_raw.shape[0] : stride, 0 : depth_raw.shape[1] : stride]
        z = depth_raw[::stride, ::stride].astype(np.float64) * float(depth_unit_m)
        q = quality[::stride, ::stride].astype(np.float64)
        valid = (z >= depth_min_m) & (z <= depth_max_m) & (q > 0.0)
        if not np.any(valid):
            return

        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
        points_camera = np.column_stack(
            (
                (xs[valid] - cx) * z[valid] / fx,
                (ys[valid] - cy) * z[valid] / fy,
                z[valid],
            )
        )
        pose = np.asarray(camera_to_world, dtype=np.float64)
        points_world = points_camera @ pose[:3, :3].T + pose[:3, 3]
        values = q[valid]

        if roi_min is not None and roi_max is not None:
            inside = np.all(points_world >= roi_min, axis=1) & np.all(
                points_world <= roi_max, axis=1
            )
            points_world = points_world[inside]
            values = values[inside]
        if len(points_world) == 0:
            return

        voxel_indices = np.floor(points_world / self.voxel_size_m).astype(np.int64)
        unique, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        quality_sums = np.bincount(inverse, weights=values)
        frame_quality_means = quality_sums / np.maximum(counts, 1)

        camera = pose[:3, 3]
        centers = (unique.astype(np.float64) + 0.5) * self.voxel_size_m
        directions = centers - camera
        major_axes = np.argmax(np.abs(directions), axis=1)
        signs = directions[np.arange(len(directions)), major_axes] >= 0.0
        view_bits = 1 << (major_axes * 2 + signs.astype(np.int64))

        for idx, key_array in enumerate(unique):
            key = tuple(int(v) for v in key_array)
            item = self.evidence.get(key)
            if item is None:
                item = VoxelEvidence()
                self.evidence[key] = item
            # Count independent keyframes, not multiple neighboring pixels from
            # one image. Otherwise a single dense view would look over-confident.
            item.observations += 1
            item.quality_sum += float(frame_quality_means[idx])
            item.view_mask |= int(view_bits[idx])

    def confidences_for_points(self, points_world: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world, dtype=np.float64)
        if len(points) == 0:
            return np.empty((0,), dtype=np.float32)
        indices = np.floor(points / self.voxel_size_m).astype(np.int64)
        result = np.full(len(points), 0.03, dtype=np.float32)
        for index, key_array in enumerate(indices):
            item = self.evidence.get(tuple(int(v) for v in key_array))
            if item is None:
                continue
            mean_quality = item.quality_sum / max(item.observations, 1)
            observation_score = 1.0 - np.exp(-item.observations / 4.0)
            view_count = int(item.view_mask).bit_count()
            diversity_score = min(view_count / 3.0, 1.0)
            result[index] = float(
                np.clip(
                    0.45 * mean_quality + 0.40 * observation_score + 0.15 * diversity_score,
                    0.0,
                    1.0,
                )
            )
        return result

    def colorize_points(self, points_world: np.ndarray) -> np.ndarray:
        return turbo_colormap(self.confidences_for_points(points_world))
