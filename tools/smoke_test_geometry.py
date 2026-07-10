from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pano_geometry import rotation_matrix_zyx, rotate_xyxy_to_obb


def main() -> None:
    rotation = rotation_matrix_zyx(math.radians(30.0), math.radians(0.0), math.radians(0.0))
    obb, points, aabb = rotate_xyxy_to_obb(
        [100.0, 100.0, 180.0, 220.0],
        width=2048,
        height=1024,
        rotation=rotation,
        samples_per_side=24,
    )
    print(
        "obb=",
        (round(obb.cx, 2), round(obb.cy, 2), round(obb.w, 2), round(obb.h, 2), round(obb.angle_rad, 4)),
    )
    print("points_shape=", points.shape)
    print("aabb=", tuple(round(v, 2) for v in aabb))


if __name__ == "__main__":
    main()

