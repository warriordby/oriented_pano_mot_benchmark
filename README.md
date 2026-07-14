# Oriented Pano MOT Benchmark

This repository provides projection-aware SO(3) rotation tools for building
oriented multi-object tracking benchmarks from panoramic datasets.

The default `erp` projection follows PriOr-Flow-style panoramic geometry:
pixel-center ERP coordinates are mapped to longitude/latitude, lifted to the
unit sphere, rotated with SO(3), and projected back to equirectangular image
coordinates. The `cylinder` projection maps x to yaw and y to `tan(elevation)`
before using the same SO(3) ray rotation. Image remapping follows PriOr-Flow
`generate_samplegrid`: each output ray is rotated to locate the source pixel.

It supports:

- QuadTrack/OmniTrack-style MOT detection files.
- DanceTrack/MOTChallenge-style datasets with `seqinfo.ini`, `img1`, `gt`,
  and `det` folders.
- Label-only conversion and optional rotated image generation.
- Output as oriented CSV plus MOT-compatible AABB txt files.

## Layout

- `docs/research_summary.md`: notes from the referenced papers and open-source projects.
- `docs/implementation_plan.md`: implementation plan for oriented panoramic MOT.
- `docs/quadtrack_orientation_benchmark.md`: QuadTrack conversion examples.
- `docs/dancetrack_orientation_benchmark.md`: DanceTrack conversion examples.
- `docs/linux_reproducibility.md`: Linux environment and reproduction notes.
- `src/pano_geometry.py`: reusable ERP/cylindrical SO(3) projection helpers.
- `configs/pipeline.example.yaml`: generic experiment configuration.
- `configs/quadtrack_orientation.example.ps1`: Windows QuadTrack example.
- `configs/dancetrack_orientation.example.sh`: Linux DanceTrack example.
- `tools/smoke_test_geometry.py`: small geometry smoke test.
- `tools/check_prior_flow_geometry.py`: checks geometry against PriOr-Flow's ERP/sample-grid convention.
- `tools/convert_quadtrack_to_orientation_benchmark.py`: QuadTrack converter.
- `tools/convert_dancetrack_to_orientation_benchmark.py`: DanceTrack converter.
- `tools/visualize_orientation_benchmark.py`: draw oriented labels on rotated images and write per-sequence videos.

## Quick Check

```bash
python -B tools/smoke_test_geometry.py
python -B tools/check_prior_flow_geometry.py
```

## Linux Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On headless Linux machines, install the OpenCV runtime libraries if needed:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

## DanceTrack Example

Label-only conversion:

```bash
python -B tools/convert_dancetrack_to_orientation_benchmark.py \
  --dancetrack-root /data/DanceTrack \
  --split val \
  --label-source gt \
  --out-root outputs/dancetrack_orientation_benchmark \
  --variants prior_a2b,polar_up,target_north_80 \
  --edge-samples 32
```

With rotated images:

```bash
python -B tools/convert_dancetrack_to_orientation_benchmark.py \
  --dancetrack-root /data/DanceTrack \
  --split val \
  --label-source gt \
  --out-root outputs/dancetrack_orientation_benchmark \
  --variants prior_a2b,polar_up,target_north_80 \
  --edge-samples 32 \
  --rotate-images
```

## QuadTrack Example

For the official MOTChallenge/DanceTrack-like QuadTrack layout:

```text
../QuadTrack/test/
  <sequence>/
    seqinfo.ini
    img1/
    gt/gt.txt   # if ground truth is available
    det/det.txt # if detection boxes are available
```

you can run the whole split directly:

```bash
python -B tools/convert_quadtrack_to_orientation_benchmark.py \
  --quadtrack-root ../QuadTrack/test \
  --out-root ./outputs/quadtrack_orientation_benchmark \
  --image-width 2048 \
  --image-height 480 \
  --projection cylinder \
  --vertical-fov-deg 120 \
  --variants prior_a2b,polar_up,target_north_55 \
  --edge-samples 32 \
  --input-kind detections
```

In this mode `--input-format auto` detects sequence folders automatically and
`--label-source auto` uses `gt/gt.txt` first, otherwise `det/det.txt`. If the
official `test` split contains only `img1` images and no box file, there is
nothing to rotate yet; run a detector/tracker export into `det/det.txt`, or use
a split that includes `gt/gt.txt`.

To also rotate images, add `--rotate-images`:

```bash
python -B tools/convert_quadtrack_to_orientation_benchmark.py \
  --quadtrack-root ../QuadTrack/test \
  --out-root ./outputs/quadtrack_orientation_benchmark_with_images \
  --image-width 2048 \
  --image-height 480 \
  --projection cylinder \
  --vertical-fov-deg 120 \
  --variants prior_a2b,polar_up,target_north_55 \
  --edge-samples 32 \
  --input-kind detections \
  --rotate-images \
  --invalid-image-fill black
```

For official sequence folders, the converter now scans the real files under
`<sequence>/<imDir>`, `<sequence>/img1`, or `<sequence>/images`, so it can handle
both zero-based names such as `000000.jpg` and one-based names such as
`000001.jpg`. If `--rotate-images` still reports `images=0`, verify that images
exist and are readable:

QuadTrack's 120-degree vertical coverage is a limited-FOV panorama, not the
full 180-degree ERP assumed by PriOr-Flow's paper figures. Use
`--projection cylinder` if the source images are cylindrical panoramas; use
`--projection erp` only when y is latitude-linear ERP. Strong non-yaw image
rotations can still request source rays that do not exist in the original
image. The converter fills those pixels with `--invalid-image-fill` instead of
stretching top/bottom rows. Use `--vertical-fov-deg 180` only when the source
image and boxes are in full-ERP coordinates.

```bash
find ../QuadTrack/test -maxdepth 3 \( -path '*/img1/*' -o -path '*/images/*' \) | head
```

The default QuadTrack path expected by the converter is:

```text
/data/QuadTrack_test/OmniTrack_Omnidet_test/
  detection_results_mot/
    <sequence>.txt
```

Each MOT row is interpreted as:

```text
frame,id,x,y,w,h,score,...
```

Label-only conversion from QuadTrack MOT txt files:

```bash
python -B tools/convert_quadtrack_to_orientation_benchmark.py \
  --quadtrack-root /data/QuadTrack_test/OmniTrack_Omnidet_test \
  --out-root outputs/quadtrack_orientation_benchmark \
  --image-width 2048 \
  --image-height 480 \
  --projection cylinder \
  --vertical-fov-deg 120 \
  --variants prior_a2b,polar_up,target_north_55 \
  --edge-samples 32 \
  --input-kind detections
```

If the MOT txt files are not under `<quadtrack-root>/detection_results_mot`,
set `--det-root` explicitly:

```bash
python -B tools/convert_quadtrack_to_orientation_benchmark.py \
  --quadtrack-root /data/QuadTrack_test/OmniTrack_Omnidet_test \
  --det-root /data/QuadTrack_test/OmniTrack_Omnidet_test/detection_results_mot \
  --out-root outputs/quadtrack_orientation_benchmark \
  --image-width 2048 \
  --image-height 480 \
  --projection cylinder \
  --vertical-fov-deg 120 \
  --variants target_north_55,target_south_55
```

With rotated images:

```bash
python -B tools/convert_quadtrack_to_orientation_benchmark.py \
  --quadtrack-root /data/QuadTrack_test/OmniTrack_Omnidet_test \
  --image-root /data/QuadTrack_test/images \
  --out-root outputs/quadtrack_orientation_benchmark \
  --projection cylinder \
  --vertical-fov-deg 120 \
  --variants prior_a2b,polar_up,target_north_55 \
  --rotate-images \
  --invalid-image-fill black
```

For JSON annotations instead of MOT txt:

```bash
python -B tools/convert_quadtrack_to_orientation_benchmark.py \
  --quadtrack-root /data/QuadTrack_test/OmniTrack_Omnidet_test \
  --input-format quadtrack_json \
  --out-root outputs/quadtrack_orientation_benchmark \
  --image-width 2048 \
  --image-height 480 \
  --projection cylinder \
  --vertical-fov-deg 120 \
  --variants target_north_55
```

Common parameter adjustments:

- `--image-width`, `--image-height`: set these to the panorama resolution used
  by the input boxes. QuadTrack examples here use `2048x480`. If `--image-root`
  is provided and images are readable, the converter uses the real image size.
- `--projection`: projection model for pixels and boxes. `erp` is the
  PriOr-Flow latitude-linear equirectangular model. `cylinder` maps x to yaw
  and y to `tan(elevation)`, which is the appropriate setting when the source
  panorama is cylindrical.
- `--vertical-fov-deg`: vertical angular coverage used by the y coordinate.
  QuadTrack uses `120`; with `--projection cylinder` it must be less than
  `180` because cylindrical y has a tangent singularity at +/-90 degrees.
- `--invalid-image-fill`: controls pixels whose rotated source ray falls
  outside the available vertical FOV when `--rotate-images` is used. `black`
  is closest to PriOr-Flow's `grid_sample` mask behavior. `edge` reproduces the
  old top/bottom row stretching and is mainly for debugging.
- `--variants`: choose rotations. `prior_a2b`/`prior_b2a` reproduce
  PriOr-Flow orthogonal rotations; `polar_up`/`polar_down` create fixed strong
  polar projection distortion; under QuadTrack's 120-degree y coverage use
  `target_north_55`/`target_south_55` to stay inside the visible +/-60-degree
  latitude range. Use `target_north_80`/`target_south_80` only with
  `--vertical-fov-deg 180` full-ERP geometry. `custom` uses `--yaw-deg
  --pitch-deg --roll-deg`.
- `--edge-samples`: samples each original box side before rotation. Use `16`
  for quick checks, `32` for normal conversion, and `64` for large boxes or
  severe polar distortion.
- `--mot-frame-to-image-offset`: controls MOT frame to image filename mapping.
  The default `-1` maps frame `1` to `000000.jpg`. Use `0` if frame `1` maps to
  `000001.jpg`.
- `--frame-name-width`, `--frame-image-ext`: adjust synthesized image names,
  for example `--frame-name-width 5 --frame-image-ext .png`.
- `--min-score`: filters low-confidence detections before conversion.
- `--label-source`: for official sequence folders, choose `gt`, `det`, or
  `auto`; `auto` prefers ground truth when it exists.
- `--seq-glob`, `--limit-seqs`, `--limit-frames`: restrict conversion while
  debugging, for example `--seq-glob 'scene_*.txt' --limit-frames 100`.

More details are in `docs/quadtrack_orientation_benchmark.md`.

## Visualization

After running conversion with `--rotate-images`, create one video per
variant/sequence:

```bash
python -B tools/visualize_orientation_benchmark.py \
  --benchmark-root ./outputs/quadtrack_orientation_benchmark_with_images \
  --label-kind detections \
  --out-root ./outputs/quadtrack_orientation_benchmark_with_images/visualizations \
  --fps 10
```

Outputs are written as:

```text
outputs/quadtrack_orientation_benchmark_with_images/
  visualizations/
    prior_a2b/0000.mp4
    polar_up/0000.mp4
    target_north_55/0000.mp4
    visualization_manifest.json
```

To visualize only selected data while debugging:

```bash
python -B tools/visualize_orientation_benchmark.py \
  --benchmark-root ./outputs/quadtrack_orientation_benchmark_with_images \
  --variants prior_a2b,target_north_55 \
  --seqs 0000,0001 \
  --max-frames 100 \
  --draw-aabb
```

The visualizer reads `images/<variant>/<sequence>/` and
`detections/oriented_csv/<variant>/<sequence>.csv`. It draws oriented polygons,
object IDs, scores, and wraps seam-crossing boxes horizontally for display.
