# Oriented Pano MOT Benchmark

This repository provides spherical rotation tools for building oriented
multi-object tracking benchmarks from panoramic datasets.

The implementation follows the geometry used by PriOr-Flow-style panoramic
projection: pixels are mapped to longitude/latitude, lifted to the unit sphere,
rotated with SO(3), and projected back to equirectangular image coordinates.

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
- `src/pano_geometry.py`: reusable equirectangular SO(3) geometry helpers.
- `configs/pipeline.example.yaml`: generic experiment configuration.
- `configs/quadtrack_orientation.example.ps1`: Windows QuadTrack example.
- `configs/dancetrack_orientation.example.sh`: Linux DanceTrack example.
- `tools/smoke_test_geometry.py`: small geometry smoke test.
- `tools/convert_quadtrack_to_orientation_benchmark.py`: QuadTrack converter.
- `tools/convert_dancetrack_to_orientation_benchmark.py`: DanceTrack converter.

## Quick Check

```bash
python -B tools/smoke_test_geometry.py
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

```bash
python -B tools/convert_quadtrack_to_orientation_benchmark.py \
  --quadtrack-root /data/QuadTrack_test/OmniTrack_Omnidet_test \
  --out-root outputs/quadtrack_orientation_benchmark \
  --image-width 2048 \
  --image-height 480 \
  --variants prior_a2b,polar_up,target_north_80 \
  --edge-samples 32 \
  --input-kind detections
```
