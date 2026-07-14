from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pano_geometry import make_equirectangular_remap, rotate_points, rotation_matrix_zyx


def prior_pixels_to_xyz(points_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    theta = ((x + 0.5) / float(width) - 0.5) * 2.0 * math.pi
    phi = (0.5 - (y + 0.5) / float(height)) * math.pi
    return np.stack(
        [np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)],
        axis=1,
    )


def prior_xyz_to_pixels(xyz: np.ndarray, width: int, height: int) -> np.ndarray:
    unit = xyz / np.maximum(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-12)
    theta = np.arctan2(unit[:, 1], unit[:, 0])
    phi = np.arcsin(np.clip(unit[:, 2], -1.0, 1.0))
    x = (theta / (2.0 * math.pi) + 0.5) * float(width) - 0.5
    y = (0.5 - phi / math.pi) * float(height) - 0.5
    return np.stack([x, y], axis=1)


def prior_apply(points_xy: np.ndarray, width: int, height: int, rotation: np.ndarray) -> np.ndarray:
    """NumPy copy of PriOr-Flow generate_samplegrid geometry for selected pixels."""
    xyz = prior_pixels_to_xyz(points_xy, width, height)
    rotated = xyz @ rotation.T
    return prior_xyz_to_pixels(rotated, width, height)


def check_remap_matches_prior_grid() -> None:
    width, height = 2048, 480
    rotation = rotation_matrix_zyx(0.0, 0.0, -math.pi / 2.0)
    points = np.array(
        [[0, 0], [512, 120], [1024, 240], [1536, 360], [2047, 479]],
        dtype=np.float64,
    )
    expected = prior_apply(points, width, height, rotation)
    map_x, map_y = make_equirectangular_remap(width, height, rotation)
    actual = np.array([[map_x[int(y), int(x)], map_y[int(y), int(x)]] for x, y in points], dtype=np.float64)
    diff = np.max(np.abs(actual - expected))
    if diff > 1e-3:
        raise AssertionError(f"remap differs from PriOr-Flow grid: max_diff={diff}\nactual={actual}\nexpected={expected}")


def check_label_motion_matches_prior_image_content() -> None:
    width, height = 2048, 480
    rotation = rotation_matrix_zyx(0.0, 0.0, -math.pi / 2.0)
    points = np.array(
        [[10, 10], [512, 120], [1024, 240], [1536, 360], [2000, 450]],
        dtype=np.float64,
    )
    # PriOr image content moves by the inverse of the output-to-source grid.
    expected = prior_apply(points, width, height, rotation.T)
    actual = rotate_points(points, width, height, rotation)
    diff = np.max(np.abs(actual - expected))
    if diff > 1e-3:
        raise AssertionError(f"label motion differs from PriOr-Flow content motion: max_diff={diff}")


def main() -> None:
    check_remap_matches_prior_grid()
    check_label_motion_matches_prior_image_content()
    print("PriOr-Flow geometry checks passed.")


if __name__ == "__main__":
    main()
