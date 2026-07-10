# QuadTrack Orientation Benchmark Conversion

This project includes a script that converts QuadTrack-style panoramic MOT
detections into an orientation benchmark by applying PriOr-Flow-style spherical
rotations.

## Reference From PriOr-Flow

PriOr-Flow uses `projection_prim_ortho.py` for the core panoramic geometry:

- `generate_rotation_metrix(axis_list=['z','y','x'], theta_list=...)`
- `ERP.plane2spherical`
- `Spherical2Cartesian`
- `rotate_cartesian`
- `ERP.spherical2plane`
- `generate_samplegrid`
- `img_rotate`
- `flo_rotate`

The conversion script mirrors the same idea on CPU/OpenCV:

```text
ERP pixel -> longitude/latitude -> unit sphere -> SO(3) rotation -> ERP pixel
```

For labels it samples each box edge densely before rotation, because after a
spherical rotation a rectangle edge is generally not a rectangle edge in the ERP
image plane.

## Usage

Label-only conversion for the available QuadTrack detection files:

```powershell
py -B tools\convert_quadtrack_to_orientation_benchmark.py `
  --quadtrack-root D:\googledownload\omnitrack\QuadTrack_test\OmniTrack_Omnidet_test `
  --out-root outputs\quadtrack_orientation_benchmark `
  --image-width 2048 `
  --image-height 480 `
  --variants prior_a2b,polar_up,target_north_80 `
  --input-kind detections
```

If original images are available, add:

```powershell
  --image-root D:\path\to\quadtrack\images `
  --rotate-images
```

## Output

```text
outputs/quadtrack_orientation_benchmark/
  orientation_benchmark_manifest.json
  detections/
    oriented_csv/<variant>/<sequence>.csv
    mot_aabb/<variant>/<sequence>.txt
  images/<variant>/<sequence>/*.jpg   # only when --rotate-images is used
```

The oriented CSV contains:

```text
frame, object_id, class_name,
cx_unwrapped, cy, w, h, angle_rad, score,
poly_x1, poly_y1, ..., poly_x4, poly_y4,
aabb_x, aabb_y, aabb_w, aabb_h,
distortion_score, seam_crossing, valid
```

Coordinates are intentionally allowed to be horizontally unwrapped. This keeps
objects crossing the ERP seam usable for oriented IoU. A consumer should wrap
only for visualization, not for matching.

## Rotation Variants

- `prior_a2b`: PriOr-Flow orthogonal view, Euler z-y-x `[0, 0, -90deg]`.
- `prior_b2a`: inverse orthogonal view, Euler z-y-x `[0, 0, +90deg]`.
- `polar_up`: fixed `Ry(+75deg)` strong ERP polar distortion.
- `polar_down`: fixed `Ry(-75deg)` strong ERP polar distortion.
- `target_north_80`: per-sequence rotation that moves the median detection ray
  near latitude `+80deg`.
- `target_south_80`: per-sequence rotation that moves the median detection ray
  near latitude `-80deg`.
- `custom`: use `--yaw-deg --pitch-deg --roll-deg`.

For strong distortion, `target_north_80` and `target_south_80` are usually more
aggressive than yaw-only rotation. Yaw mostly tests seam continuity; polar
rotations create the severe ERP stretching needed by an orientation benchmark.

