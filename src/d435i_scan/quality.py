from __future__ import annotations

import numpy as np


def depth_quality(
    depth_raw: np.ndarray,
    depth_unit_m: float,
    depth_min_m: float,
    depth_max_m: float,
    previous_depth_raw: np.ndarray | None = None,
) -> np.ndarray:
    """Return a conservative 0..1 confidence estimate for every depth pixel.

    D435i does not expose a calibrated per-pixel confidence stream.  This score is
    therefore an explicit heuristic based on validity, range, local discontinuity,
    and temporal agreement.  It is useful as operator guidance, not uncertainty in mm.
    """
    depth_m = np.asarray(depth_raw, dtype=np.float32) * float(depth_unit_m)
    valid = (depth_m >= depth_min_m) & (depth_m <= depth_max_m)

    span = max(depth_max_m - depth_min_m, 1e-6)
    normalized_range = np.clip((depth_m - depth_min_m) / span, 0.0, 1.0)
    range_score = 1.0 - 0.55 * normalized_range**2

    dx = np.abs(depth_m - np.roll(depth_m, 1, axis=1))
    dy = np.abs(depth_m - np.roll(depth_m, 1, axis=0))
    discontinuity = np.maximum(dx, dy)
    edge_score = np.exp(-discontinuity / 0.012)
    edge_score[:, 0] = 0.0
    edge_score[0, :] = 0.0

    if previous_depth_raw is not None and previous_depth_raw.shape == depth_raw.shape:
        previous_m = previous_depth_raw.astype(np.float32) * float(depth_unit_m)
        temporal_score = np.exp(-np.abs(depth_m - previous_m) / 0.008)
        temporal_score[previous_m <= 0.0] = 0.45
    else:
        temporal_score = np.full(depth_m.shape, 0.70, dtype=np.float32)

    score = (0.35 * range_score + 0.35 * edge_score + 0.30 * temporal_score) * valid
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def turbo_colormap(values: np.ndarray) -> np.ndarray:
    """Small dependency-free approximation of Google's Turbo colormap."""
    x = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    # Polynomial coefficients published with the Turbo colormap reference code.
    k_red = np.array(
        [0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943]
    )
    k_green = np.array([0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604])
    k_blue = np.array(
        [0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973]
    )
    powers = np.stack([np.ones_like(x), x, x**2, x**3, x**4, x**5], axis=-1)
    rgb = np.stack(
        [powers @ k_red, powers @ k_green, powers @ k_blue],
        axis=-1,
    )
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def quality_image(score: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray((turbo_colormap(score) * 255.0).astype(np.uint8))


def depth_color_image(
    depth_raw: np.ndarray,
    depth_unit_m: float,
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray:
    depth_m = np.asarray(depth_raw, dtype=np.float32) * float(depth_unit_m)
    span = max(depth_max_m - depth_min_m, 1e-6)
    normalized = 1.0 - np.clip((depth_m - depth_min_m) / span, 0.0, 1.0)
    rgb = turbo_colormap(normalized)
    invalid = (depth_m < depth_min_m) | (depth_m > depth_max_m)
    rgb[invalid] = 0.0
    return np.ascontiguousarray((rgb * 255.0).astype(np.uint8))
