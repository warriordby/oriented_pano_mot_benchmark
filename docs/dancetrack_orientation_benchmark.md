# DanceTrack Orientation Benchmark Conversion

This guide shows how to convert a DanceTrack or MOTChallenge-style dataset into
an oriented panoramic MOT benchmark. The converter expects the standard layout:

```text
DanceTrack/
  train/
    dancetrack0001/
      seqinfo.ini
      img1/00000001.jpg
      gt/gt.txt
      det/det.txt
  val/
    dancetrackxxxx/
      seqinfo.ini
      img1/
      gt/gt.txt
      det/det.txt
```

DanceTrack is not originally panoramic. Use this converter when your data is in
DanceTrack/MOTChallenge format but the images are equirectangular panoramas, or
when you intentionally want to create a panoramic stress-test benchmark from
MOT-style annotations.

## Label-Only Conversion

```bash
python -B tools/convert_dancetrack_to_orientation_benchmark.py \
  --dancetrack-root /data/DanceTrack \
  --split val \
  --label-source gt \
  --out-root outputs/dancetrack_orientation_benchmark \
  --variants prior_a2b,polar_up,target_north_80 \
  --edge-samples 32
```

Use detections instead of ground truth:

```bash
python -B tools/convert_dancetrack_to_orientation_benchmark.py \
  --dancetrack-root /data/DanceTrack \
  --split val \
  --label-source det \
  --out-root outputs/dancetrack_orientation_det_benchmark \
  --variants prior_a2b,polar_up,target_north_80 \
  --edge-samples 32
```

## Rotated Images And Labels

```bash
python -B tools/convert_dancetrack_to_orientation_benchmark.py \
  --dancetrack-root /data/DanceTrack \
  --split train \
  --label-source gt \
  --out-root outputs/dancetrack_orientation_train \
  --variants prior_a2b,polar_up,target_north_80,target_south_80 \
  --edge-samples 32 \
  --rotate-images
```

If `seqinfo.ini` does not contain image size, pass it explicitly:

```bash
python -B tools/convert_dancetrack_to_orientation_benchmark.py \
  --dancetrack-root /data/DanceTrack \
  --split val \
  --label-source gt \
  --out-root outputs/dancetrack_orientation_benchmark \
  --image-width 2048 \
  --image-height 1024 \
  --variants custom \
  --yaw-deg 0 \
  --pitch-deg 80 \
  --roll-deg 0
```

## Output Structure

```text
outputs/dancetrack_orientation_benchmark/
  orientation_benchmark_manifest.json
  gt/
    oriented_csv/<variant>/<sequence>.csv
    mot_aabb/<variant>/<sequence>.txt
  det/
    oriented_csv/<variant>/<sequence>.csv
    mot_aabb/<variant>/<sequence>.txt
  images/<variant>/<sequence>/*.jpg
```

The oriented CSV stores:

```text
frame, object_id, class_name,
cx_unwrapped, cy, w, h, angle_rad, score,
poly_x1, poly_y1, ..., poly_x4, poly_y4,
aabb_x, aabb_y, aabb_w, aabb_h,
distortion_score, seam_crossing, valid
```

`cx_unwrapped` and polygon x coordinates may be outside `[0, W)`. This is
intentional: seam-crossing objects remain geometrically compact for rotated IoU.
Only wrap coordinates for visualization.

## Rotation Variants

- `prior_a2b`: PriOr-Flow orthogonal view, z-y-x Euler `[0, 0, -90deg]`.
- `prior_b2a`: inverse orthogonal view, z-y-x Euler `[0, 0, +90deg]`.
- `polar_up`: fixed `Ry(+75deg)` to create strong ERP polar distortion.
- `polar_down`: fixed `Ry(-75deg)` to create strong ERP polar distortion.
- `target_north_80`: per-sequence rotation moving the mean object ray toward
  latitude `+80deg`.
- `target_south_80`: per-sequence rotation moving the mean object ray toward
  latitude `-80deg`.
- `custom`: use `--yaw-deg`, `--pitch-deg`, and `--roll-deg`.

For orientation stress testing, prefer `polar_up`, `target_north_80`, and
`target_south_80`. Yaw-only rotations mainly test panorama seam handling and do
not produce the strongest ERP shape distortion.

## Minimal Smoke Test

```bash
python -B tools/convert_dancetrack_to_orientation_benchmark.py \
  --dancetrack-root /data/DanceTrack \
  --split val \
  --label-source gt \
  --out-root outputs/dancetrack_smoke \
  --variants target_north_80 \
  --limit-seqs 1 \
  --limit-frames 10
```

