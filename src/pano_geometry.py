"""Spherical geometry helpers for oriented panoramic MOT.

The helpers assume equirectangular images. Pixel coordinates follow OpenCV
convention: x grows right, y grows down.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class OrientedBox:
    cx: float
    cy: float
    w: float
    h: float
    angle_rad: float


def rotation_matrix_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return R = Rz(yaw) @ Ry(pitch) @ Rx(roll), with angles in radians."""
    cz, sz = cos(yaw), sin(yaw)
    cy, sy = cos(pitch), sin(pitch)
    cx, sx = cos(roll), sin(roll)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ ry @ rx


def rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a rotation matrix that maps source direction to target direction."""
    a = np.asarray(source, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    cross = np.cross(a, b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    norm_cross = float(np.linalg.norm(cross))
    if norm_cross < 1e-12:
        if dot > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(axis, a))) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(a, axis)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        return axis_angle_rotation(axis, pi)
    axis = cross / norm_cross
    angle = float(np.arctan2(norm_cross, dot))
    return axis_angle_rotation(axis, angle)


def axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix."""
    x, y, z = np.asarray(axis, dtype=np.float64)
    c = cos(angle)
    s = sin(angle)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )


def pixels_to_lonlat(
    x: np.ndarray,
    y: np.ndarray,
    width: float,
    height: float,
) -> Tuple[np.ndarray, np.ndarray]:
    # Match PriOr-Flow ERP.m2theta / ERP.n2phi: pixel coordinates are centers.
    lon = 2.0 * pi * ((np.asarray(x, dtype=np.float64) + 0.5) / float(width) - 0.5)
    lat = pi * (0.5 - (np.asarray(y, dtype=np.float64) + 0.5) / float(height))
    return lon, lat


def lonlat_to_pixels(
    lon: np.ndarray,
    lat: np.ndarray,
    width: float,
    height: float,
) -> Tuple[np.ndarray, np.ndarray]:
    # Match PriOr-Flow ERP.theta2m / ERP.phi2n.
    x = (np.asarray(lon, dtype=np.float64) / (2.0 * pi) + 0.5) * float(width) - 0.5
    y = (0.5 - np.asarray(lat, dtype=np.float64) / pi) * float(height) - 0.5
    return x, y


def lonlat_to_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    clat = np.cos(lat)
    return np.stack([clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)], axis=-1)


def pixel_to_xyz(x: np.ndarray, y: np.ndarray, width: float, height: float) -> np.ndarray:
    lon, lat = pixels_to_lonlat(x, y, width, height)
    return lonlat_to_xyz(lon, lat)


def xyz_to_lonlat(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float64)
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    unit = xyz / np.maximum(norm, 1e-12)
    lon = np.arctan2(unit[..., 1], unit[..., 0])
    lat = np.arcsin(np.clip(unit[..., 2], -1.0, 1.0))
    return lon, lat


def make_equirectangular_remap(
    width: int,
    height: int,
    rotation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build PriOr-Flow-style remap arrays for cv2.remap.

    PriOr-Flow generate_samplegrid applies R to each output ray to obtain the
    source sampling ray. With row-vector numpy arrays this is xyz_out @ R.T.
    """
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    lon, lat = pixels_to_lonlat(xx, yy, width, height)
    xyz_out = lonlat_to_xyz(lon, lat)
    xyz_src = xyz_out @ np.asarray(rotation, dtype=np.float64).T
    src_lon, src_lat = xyz_to_lonlat(xyz_src)
    map_x, map_y = lonlat_to_pixels(src_lon, src_lat, width, height)
    return (
        np.mod(map_x, float(width)).astype(np.float32),
        np.clip(map_y, 0.0, float(height - 1)).astype(np.float32),
    )


def sample_xyxy_edges(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    samples_per_side: int = 24,
) -> np.ndarray:
    n = max(int(samples_per_side), 2)
    top_x = np.linspace(x1, x2, n, endpoint=False)
    right_y = np.linspace(y1, y2, n, endpoint=False)
    bottom_x = np.linspace(x2, x1, n, endpoint=False)
    left_y = np.linspace(y2, y1, n, endpoint=False)
    return np.concatenate(
        [
            np.stack([top_x, np.full_like(top_x, y1)], axis=1),
            np.stack([np.full_like(right_y, x2), right_y], axis=1),
            np.stack([bottom_x, np.full_like(bottom_x, y2)], axis=1),
            np.stack([np.full_like(left_y, x1), left_y], axis=1),
        ],
        axis=0,
    )


def rotate_points(
    points_xy: np.ndarray,
    width: int,
    height: int,
    rotation: np.ndarray,
) -> np.ndarray:
    """Move source-plane points consistently with PriOr-Flow img_rotate.

    PriOr-Flow samples source = R @ output, so visible image content moves as
    output = R.T @ source. With row-vector numpy arrays this is xyz @ R.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    lon, lat = pixels_to_lonlat(pts[:, 0], pts[:, 1], width, height)
    xyz = lonlat_to_xyz(lon, lat)
    rotated = xyz @ np.asarray(rotation, dtype=np.float64)
    out_lon, out_lat = xyz_to_lonlat(rotated)
    out_x, out_y = lonlat_to_pixels(out_lon, out_lat, width, height)
    return np.stack(
        [np.mod(out_x, float(width)), np.clip(out_y, 0.0, float(height - 1))],
        axis=1,
    )


def rotate_points_unwrapped(
    points_xy: np.ndarray,
    width: int,
    height: int,
    rotation: np.ndarray,
    reference_x: float | None = None,
) -> np.ndarray:
    rotated = rotate_points(points_xy, width, height, rotation)
    return unwrap_x(rotated, width, reference_x=reference_x)


def unwrap_x(
    points_xy: np.ndarray,
    width: int,
    reference_x: float | None = None,
) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64).copy()
    if pts.size == 0:
        return pts
    period = float(width)
    if reference_x is None:
        angles = pts[:, 0] / period * 2.0 * pi
        mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
        reference_x = (mean_angle % (2.0 * pi)) / (2.0 * pi) * period
    pts[:, 0] = reference_x + ((pts[:, 0] - reference_x + 0.5 * period) % period - 0.5 * period)
    return pts


def polygon_aabb(points_xy: np.ndarray) -> Tuple[float, float, float, float]:
    pts = np.asarray(points_xy, dtype=np.float64)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return float(x1), float(y1), float(x2), float(y2)


def normalize_rect_angle(angle: float) -> float:
    """Normalize rectangle orientation to [-pi/2, pi/2)."""
    return float((angle + 0.5 * pi) % pi - 0.5 * pi)


def pca_oriented_box(points_xy: np.ndarray) -> OrientedBox:
    """Fit a PCA oriented box to points.

    For production use, compare this with a true minimum-area rectangle.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    center = pts.mean(axis=0)
    centered = pts - center
    if len(pts) < 2 or np.allclose(centered, 0.0):
        return OrientedBox(float(center[0]), float(center[1]), 0.0, 0.0, 0.0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axes = vh[:2]
    proj = centered @ axes.T
    mins = proj.min(axis=0)
    maxs = proj.max(axis=0)
    size = maxs - mins
    obb_center = center + ((mins + maxs) * 0.5) @ axes
    angle = normalize_rect_angle(float(np.arctan2(axes[0, 1], axes[0, 0])))
    return OrientedBox(
        float(obb_center[0]),
        float(obb_center[1]),
        float(size[0]),
        float(size[1]),
        angle,
    )


def rotate_xyxy_to_obb(
    xyxy: Iterable[float],
    width: int,
    height: int,
    rotation: np.ndarray,
    samples_per_side: int = 24,
) -> Tuple[OrientedBox, np.ndarray, Tuple[float, float, float, float]]:
    """Rotate an image-plane AABB on the sphere and fit an output OBB."""
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    edge_points = sample_xyxy_edges(x1, y1, x2, y2, samples_per_side=samples_per_side)
    rotated_points = rotate_points(edge_points, width, height, rotation)
    unwrapped = unwrap_x(rotated_points, width)
    return pca_oriented_box(unwrapped), unwrapped, polygon_aabb(unwrapped)


def distortion_score_from_y(y: float, height: int, floor: float = 0.05) -> float:
    """ERP horizontal stretching grows as 1 / cos(latitude)."""
    _, lat = pixels_to_lonlat(np.asarray([0.0]), np.asarray([y]), 1.0, float(height))
    return float(1.0 / max(float(np.cos(abs(lat[0]))), float(floor)))
