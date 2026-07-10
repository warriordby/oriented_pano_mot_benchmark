#!/usr/bin/env bash
set -euo pipefail

DANCETRACK_ROOT="${DANCETRACK_ROOT:-/data/DanceTrack}"
OUT_ROOT="${OUT_ROOT:-outputs/dancetrack_orientation_benchmark}"

python -B tools/convert_dancetrack_to_orientation_benchmark.py \
  --dancetrack-root "${DANCETRACK_ROOT}" \
  --split val \
  --label-source gt \
  --out-root "${OUT_ROOT}" \
  --variants prior_a2b,polar_up,target_north_80 \
  --edge-samples 32

