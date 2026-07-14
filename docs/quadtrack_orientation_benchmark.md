# QuadTrack Orientation Benchmark Conversion

This project includes a script that converts QuadTrack-style panoramic MOT
detections into an orientation benchmark by applying SO(3) rotations through an
explicit panorama projection model.

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

With `--projection erp`, the conversion script mirrors the same idea on
CPU/OpenCV:

```text
PriOr-Flow pixel-center ERP coordinate
  -> longitude/latitude
  -> unit sphere
  -> SO(3) rotation
  -> PriOr-Flow pixel-center ERP coordinate
```

For original PriOr-Flow full-ERP geometry, the vertical coverage is 180 degrees.
For QuadTrack, the image y axis covers 120 degrees, so use
`--vertical-fov-deg 120`. If the source panorama is cylindrical, also use
`--projection cylinder`. In that mode, x is yaw and y is linear in
`tan(elevation)` before each sample is lifted to a unit ray. The OBB path still
uses the same high-level logic: sample box edges, lift samples to the unit ray,
rotate/reproject them, then fit an oriented box to the rotated samples.

This also means a 120-degree QuadTrack image is not a complete sphere. Some
PriOr-Flow paper visual effects require source rays outside the available
QuadTrack crop. The converter therefore fills those rotated-image pixels with
`--invalid-image-fill black` by default instead of stretching the top/bottom
rows. To reproduce PriOr-Flow full-ERP visuals exactly, use a full ERP source
with `--projection erp --vertical-fov-deg 180`.

For images, the converter follows PriOr-Flow `generate_samplegrid`: for each
output pixel, rotate the output ray by `R` to find the source pixel. Therefore
visible image content moves by `R.T`. Labels are moved with that same visible
content motion so rotated images and oriented boxes stay aligned.

For labels it samples each box edge densely before rotation, because after an
SO(3) projection rotation a rectangle edge is generally not a rectangle edge in
the output image plane.

## Supported QuadTrack Inputs

The converter supports three common QuadTrack export forms.

### Official MOTChallenge/DanceTrack-like sequence folders

Official releases are often laid out as:

```text
<quadtrack-root>/
  <sequence>/
    seqinfo.ini
    img1/
    gt/gt.txt
    det/det.txt
```

`gt/gt.txt` and `det/det.txt` use MOTChallenge rows:

```text
frame,id,x,y,w,h,score,...
```

`--input-format auto` detects this layout automatically. `--label-source auto`
prefers `gt/gt.txt` when present and falls back to `det/det.txt`.

For the server layout in which the split itself is passed as the root:

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

If the official `test` split has only images and no `gt/gt.txt` or `det/det.txt`,
the converter cannot create oriented boxes because there are no boxes to rotate.
In that case, first export detections to each sequence's `det/det.txt`, or run
the converter on a split that provides `gt/gt.txt`.

To rotate both labels and images:

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

For sequence folders, the converter scans real source images under
`<sequence>/<imDir>`, `<sequence>/img1`, and `<sequence>/images`. Numeric image
names are matched automatically:

- `000000.jpg` is treated as frame 1 when the sequence starts at 0.
- `000001.jpg` is treated as frame 1 when the sequence starts at 1.
- `00000001.jpg` and PNG/JPEG variants are also accepted when present.

### MOT txt

Default layout:

```text
<quadtrack-root>/
  detection_results_mot/
    <sequence>.txt
```

Each row is parsed as MOTChallenge-style detection data:

```text
frame,id,x,y,w,h,score,...
```

`x,y,w,h` are input panorama AABBs. They are converted to an oriented box after
SO(3) rotation by sampling each original box edge before projecting it back
through the selected `--projection` model.

### QuadTrack JSON

When `--input-format quadtrack_json` is used, the script reads JSON files under
`--quadtrack-root`. It accepts either:

```json
{"detections": {"000000.jpg": [{"box": [x1, y1, x2, y2], "score": 0.9, "label_id": "person:1"}]}}
```

or the inner frame dictionary directly:

```json
{"000000.jpg": [{"box": [x1, y1, x2, y2], "score": 0.9, "label_id": "person:1"}]}
```

For JSON input, `box` is interpreted as `x1,y1,x2,y2`.

## Usage

Label-only conversion for the available QuadTrack detection files:

```powershell
py -B tools\convert_quadtrack_to_orientation_benchmark.py `
  --quadtrack-root D:\googledownload\omnitrack\QuadTrack_test\OmniTrack_Omnidet_test `
  --out-root outputs\quadtrack_orientation_benchmark `
  --image-width 2048 `
  --image-height 480 `
  --projection cylinder `
  --vertical-fov-deg 120 `
  --variants prior_a2b,polar_up,target_north_55 `
  --edge-samples 32 `
  --input-kind detections
```

The equivalent Linux command is:

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

If the MOT txt files are not in the default `detection_results_mot` folder, set
`--det-root`:

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

If original images are available, add `--image-root` and `--rotate-images`:

```powershell
py -B tools\convert_quadtrack_to_orientation_benchmark.py `
  --quadtrack-root D:\googledownload\omnitrack\QuadTrack_test\OmniTrack_Omnidet_test `
  --image-root D:\googledownload\omnitrack\QuadTrack_test\images `
  --out-root outputs\quadtrack_orientation_benchmark `
  --projection cylinder `
  --vertical-fov-deg 120 `
  --variants prior_a2b,polar_up,target_north_55 `
  --edge-samples 32 `
  --rotate-images `
  --invalid-image-fill black
```

For JSON annotations:

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

For a quick debugging run:

```bash
python -B tools/convert_quadtrack_to_orientation_benchmark.py \
  --quadtrack-root /data/QuadTrack_test/OmniTrack_Omnidet_test \
  --out-root outputs/quadtrack_orientation_debug \
  --projection cylinder \
  --vertical-fov-deg 120 \
  --variants custom \
  --yaw-deg 0 \
  --pitch-deg 60 \
  --roll-deg 0 \
  --edge-samples 16 \
  --limit-seqs 2 \
  --limit-frames 100
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

Rows with `valid=0` are kept in the oriented CSV for auditing. They usually
mean the rotated box leaves the available vertical FOV. The MOT-compatible AABB
txt files omit those invalid rows because MOT txt has no validity flag.

## Rotation Variants

- `prior_a2b`: PriOr-Flow orthogonal view, Euler z-y-x `[0, 0, -90deg]`.
- `prior_b2a`: inverse orthogonal view, Euler z-y-x `[0, 0, +90deg]`.
- `polar_up`: fixed `Ry(+75deg)` strong polar projection distortion.
- `polar_down`: fixed `Ry(-75deg)` strong polar projection distortion.
- `target_north_55`: per-sequence rotation that moves the average detection ray
  near latitude `+55deg`, inside QuadTrack's visible +/-60-degree range.
- `target_south_55`: per-sequence rotation that moves the average detection ray
  near latitude `-55deg`, inside QuadTrack's visible +/-60-degree range.
- `custom`: use `--yaw-deg --pitch-deg --roll-deg`.

For 120-degree QuadTrack data, `target_north_55` and `target_south_55` are the
strongest target-elevation presets that stay inside the visible y range with a
small margin. Use `target_north_80` and `target_south_80` only when
`--vertical-fov-deg 180` is used with full-ERP sources. Yaw mostly tests seam
continuity; polar rotations create stronger ERP stretching but can create
invalid image pixels under limited vertical FOV.

## Parameter Meanings and When To Adjust Them

| Parameter | Meaning | When to change it |
| --- | --- | --- |
| `--quadtrack-root` | Dataset/conversion root. In default MOT mode it should contain `detection_results_mot/*.txt`. | Always set this to the QuadTrack split or export you want to convert. |
| `--det-root` | Explicit MOT txt directory. | Use when MOT txt files are not under `<quadtrack-root>/detection_results_mot`. |
| `--input-format` | `auto`, `mot`, `motchallenge`, or `quadtrack_json`. `auto` tries `detection_results_mot/*.txt`, then sequence folders, then JSON. | Use `motchallenge` to force official sequence-folder parsing; use `quadtrack_json` for JSON annotations. |
| `--label-source` | For sequence folders, chooses `gt/gt.txt`, `det/det.txt`, or `auto`. | Use `--label-source det` when converting detector boxes on an official test split; use `gt` when building from ground truth. |
| `--seq-glob` | Glob used to select sequence files. | Use for partial conversion, e.g. `--seq-glob 'scene_*.txt'`. |
| `--out-root` | Output benchmark root. | Use a new folder for each experiment setting to avoid mixing variants. |
| `--image-width`, `--image-height` | Panorama resolution used for label-only geometry. | Must match the coordinate system of the boxes. For QuadTrack examples use `2048x480`; change if your exported boxes use another resolution. |
| `--projection` | Pixel projection model: `erp` or `cylinder`. | Use `erp` for PriOr-Flow/full ERP latitude-linear panoramas. Use `cylinder` when the panorama is cylindrical: x is yaw and y is `tan(elevation)`. |
| `--vertical-fov-deg` | Vertical angular coverage represented by image y. | Use `120` for QuadTrack. With `cylinder`, this must be `<180` because `tan(90deg)` is singular. |
| `--image-root` | Optional root of original panorama images. | Needed for `--rotate-images`; also lets the script infer real image size when images are readable. |
| `--rotate-images` | Writes rotated images with the same projection-aware SO(3) transform used for labels. | Enable when you want a complete image+label benchmark. Omit for faster label-only conversion. |
| `--invalid-image-fill` | Fill policy for pixels whose source ray is outside the available vertical FOV. | Default `black` is closest to PriOr-Flow's masked out-of-bounds behavior. `edge` reproduces the old top/bottom row stretching and should mainly be used for debugging. |
| `--variants` | Comma-separated rotation variants. | Use `target_north_55,target_south_55` for QuadTrack 120-degree data; add `prior_a2b,prior_b2a` to reproduce PriOr-Flow orthogonal rotations. Use 80-degree target variants only with full ERP 180-degree geometry. |
| `--yaw-deg`, `--pitch-deg`, `--roll-deg` | Euler angles for `--variants custom`. | Use for ablations such as `--variants custom --pitch-deg 60`. |
| `--edge-samples` | Number of points sampled per original box side before rotation. | Use `16` for smoke tests, `32` for normal runs, `64` for large boxes or severe polar distortion. |
| `--min-score` | Detection score threshold. | Raise it to build a cleaner detection benchmark; keep default for GT or already filtered detections. |
| `--input-kind` | Output namespace: `detections` or `gt`. | Use `gt` if the input txt/json is ground truth. |
| `--mot-frame-to-image-offset` | For MOT txt, image filename index = MOT frame + offset. | Default `-1` maps frame `1` to `000000.jpg`. Use `0` when frame `1` maps to `000001.jpg`. |
| `--json-frame-offset` | For JSON input, output frame = numeric image stem + offset. | Default `1` maps `000000.jpg` to frame `1`. Use `0` if JSON keys are already one-based. |
| `--frame-name-width` | Zero padding width for synthesized MOT image names. | Use `5` for `00001.jpg`, `8` for `00000001.jpg`, etc. |
| `--frame-image-ext` | Input image extension for synthesized MOT frame names. | Use `.png` if the original images are PNG. |
| `--image-ext` | Output extension for rotated images. | Use `.png` for lossless rotated images. |
| `--mot-class-name` | Class name assigned to MOT txt rows. | Change when the MOT file is not person-only. |
| `--limit-seqs`, `--limit-frames` | Debug limits. | Use before long runs to verify path mapping and output format. |

## When You See "No input files found"

That error means the script found none of the supported box sources. It checks:

```text
<quadtrack-root>/detection_results_mot/*.txt
<quadtrack-root>/<sequence>/gt/gt.txt
<quadtrack-root>/<sequence>/det/det.txt
<quadtrack-root>/*.json
```

For the official `../QuadTrack/test` layout, run this quick check on Linux:

```bash
find ../QuadTrack/test -maxdepth 3 \( -path '*/gt/gt.txt' -o -path '*/det/det.txt' -o -name seqinfo.ini \) | head -30
```

If it prints only `seqinfo.ini` files, the split is image-only. The converter
needs boxes, so add `det/det.txt` files or switch to a labeled split.

If `--rotate-images` runs but every line still says `images=0`, the labels were
converted but source images were not found or could not be decoded by OpenCV.
Check the image folders:

```bash
find ../QuadTrack/test -maxdepth 3 \( -path '*/img1/*' -o -path '*/images/*' \) | head -30
```

Also check one sequence's `seqinfo.ini`:

```bash
cat ../QuadTrack/test/0000/seqinfo.ini
find ../QuadTrack/test/0000 -maxdepth 2 -type f | head -20
```

The script prints a warning with `image_root`, `first_frame`, and
`first_frame_file` when `--rotate-images` cannot write any image for a sequence.

## Practical Settings

Use these presets as starting points.

Fast geometry check:

```bash
python -B tools/convert_quadtrack_to_orientation_benchmark.py \
  --quadtrack-root /data/QuadTrack_test/OmniTrack_Omnidet_test \
  --out-root outputs/quadtrack_debug \
  --projection cylinder \
  --vertical-fov-deg 120 \
  --variants target_north_55 \
  --edge-samples 16 \
  --limit-seqs 1 \
  --limit-frames 50
```

Strong orientation benchmark without images:

```bash
python -B tools/convert_quadtrack_to_orientation_benchmark.py \
  --quadtrack-root /data/QuadTrack_test/OmniTrack_Omnidet_test \
  --out-root outputs/quadtrack_orientation_benchmark \
  --image-width 2048 \
  --image-height 480 \
  --projection cylinder \
  --vertical-fov-deg 120 \
  --variants prior_a2b,prior_b2a,polar_up,polar_down,target_north_55,target_south_55 \
  --edge-samples 32
```

Complete image+label benchmark with one-based image filenames:

```bash
python -B tools/convert_quadtrack_to_orientation_benchmark.py \
  --quadtrack-root /data/QuadTrack_test/OmniTrack_Omnidet_test \
  --image-root /data/QuadTrack_test/images \
  --out-root outputs/quadtrack_orientation_benchmark \
  --projection cylinder \
  --vertical-fov-deg 120 \
  --variants target_north_55,target_south_55 \
  --edge-samples 64 \
  --mot-frame-to-image-offset 0 \
  --rotate-images \
  --invalid-image-fill black
```
