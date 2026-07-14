# Linux Reproducibility

## Tested Assumptions

- Ubuntu 20.04 or 22.04
- Python 3.10 or newer
- CPU conversion is sufficient for labels
- OpenCV is required for rotated image generation

The conversion code does not require PyTorch or a GPU. PriOr-Flow is used as a
geometric reference; this repository reimplements the required rotation mapping
with NumPy and OpenCV.

## System Packages

For headless servers:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev libgl1 libglib2.0-0
```

## Python Environment

```bash
git clone <REMOTE_URL> oriented_pano_mot_benchmark
cd oriented_pano_mot_benchmark

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

## Verify Installation

```bash
python -B tools/smoke_test_geometry.py
python -B -m compileall src tools
```

Expected smoke-test output shape:

```text
points_shape= (96, 2)
```

## Reproduce QuadTrack Conversion

```bash
python -B tools/convert_quadtrack_to_orientation_benchmark.py \
  --quadtrack-root /data/QuadTrack_test/OmniTrack_Omnidet_test \
  --out-root outputs/quadtrack_orientation_benchmark \
  --image-width 2048 \
  --image-height 480 \
  --vertical-fov-deg 120 \
  --variants prior_a2b,polar_up,target_north_55 \
  --edge-samples 32 \
  --input-kind detections
```

## Reproduce DanceTrack Conversion

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

## Output Policy

Generated files are written under `outputs/` by default and are ignored by git.
Do not commit converted datasets or downloaded third-party repositories.
