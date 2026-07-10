# Implementation Plan

## 1. Data Model

Use a label format that preserves orientation and still allows compatibility
with MOT-style tools.

Recommended per-object fields:

```text
frame, object_id, class_id, cx, cy, w, h, angle_rad, score,
poly_x1, poly_y1, poly_x2, poly_y2, poly_x3, poly_y3, poly_x4, poly_y4,
depth
```

Keep three derived views of every object:

- `polygon`: most faithful representation for rotated IoU.
- `obb`: compact representation for detector and tracker state.
- `aabb`: compatibility export for legacy MOT visualization and evaluation.

## 2. Spherical Rotation

For equirectangular panoramas:

1. Convert every output pixel to longitude and latitude.
2. Convert longitude and latitude to a unit 3D ray.
3. Apply inverse SO(3) rotation to find the source ray.
4. Project the source ray back to source pixel coordinates.
5. Resample with horizontal wrap and vertical clamp.

For labels:

1. Sample each source box edge densely.
2. Convert sampled edge points to unit rays.
3. Apply the same SO(3) rotation used for the image.
4. Project rotated rays back to image coordinates.
5. Unwrap points around the object's center longitude to avoid seam artifacts.
6. Fit an oriented rectangle and store both polygon and OBB.

## 3. Detector Baseline

Begin with oracle rotated detections. Rotate existing boxes or ground truth
labels and pass them directly into the tracker. This isolates tracking quality
from detector quality.

Then train an OBB detector:

- YOLOv8-OBB for a fast baseline.
- MMRotate or RTMDet-R for stronger rotated detection.
- Segment-anything-style masks only if instance masks are available or needed.

Avoid ordinary planar rotation augmentation for panoramic training. Prefer
spherical SO(3) augmentation plus photometric augmentation.

## 4. Tracker Baseline

The minimum viable tracker can be ByteTrack-style:

1. Split detections into high-score and low-score sets.
2. Match high-score detections to active tracks with OBB IoU.
3. Match remaining tracks to low-score detections with relaxed thresholds.
4. Start new tracks from unmatched high-score detections.
5. Keep track state as OBB plus optional spherical center state.

Recommended cost:

```text
cost = 1 - rotated_iou
     + lambda_center * spherical_center_distance
     + lambda_depth * normalized_depth_distance
     + lambda_angle * periodic_angle_distance
```

## 5. Kalman State

Use a state that handles panorama wrapping:

```text
theta, lat_or_y, log_w, log_h, alpha,
v_theta, v_lat_or_y, v_log_w, v_log_h, v_alpha
```

Where:

- `theta` is longitude and wraps by `2*pi`.
- `alpha` is OBB angle and wraps by `pi`.
- `log_w` and `log_h` avoid invalid negative size predictions.

## 6. Evaluation

Report both:

- AABB MOT metrics for compatibility.
- Oriented metrics using polygon or rotated-rectangle IoU.

Track extra diagnostics:

- seam-crossing count,
- invalid rotated labels,
- orientation error,
- ID switches near panorama seams,
- effect of depth and optical flow cues.

## 7. Milestones

1. Spherical image rotation smoke test.
2. Label rotation visual check on a few frames.
3. Oracle rotated detection tracking.
4. OBB IoU association.
5. Periodic-angle Kalman filter.
6. OBB detector training.
7. Depth and optical-flow ablations.

