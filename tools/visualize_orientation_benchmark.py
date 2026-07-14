from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class OrientedRecord:
    frame: int
    object_id: int
    class_name: str
    score: float
    polygon: np.ndarray
    aabb: tuple[float, float, float, float]
    valid: bool
    seam_crossing: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize converted oriented panoramic MOT benchmarks as per-sequence videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python -B tools/visualize_orientation_benchmark.py \\
    --benchmark-root outputs/quadtrack_orientation_benchmark_with_images \\
    --label-kind detections \\
    --out-root outputs/quadtrack_orientation_benchmark_with_images/visualizations

  python -B tools/visualize_orientation_benchmark.py \\
    --benchmark-root outputs/quadtrack_orientation_benchmark_with_images \\
    --variants prior_a2b,target_north_80 \\
    --seqs 0000,0001 \\
    --fps 15 \\
    --draw-aabb
""",
    )
    parser.add_argument("--benchmark-root", type=Path, required=True, help="Root produced by the conversion script.")
    parser.add_argument(
        "--label-kind",
        choices=["detections", "gt"],
        default="detections",
        help="Label namespace under <benchmark-root>, matching --input-kind from conversion.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output video root. Defaults to <benchmark-root>/visualizations.",
    )
    parser.add_argument(
        "--variants",
        default="",
        help="Comma-separated variants to visualize. Default: all variants with images.",
    )
    parser.add_argument("--seqs", default="", help="Comma-separated sequence names to visualize. Default: all.")
    parser.add_argument("--fps", type=float, default=10.0, help="Output video FPS.")
    parser.add_argument("--codec", default="mp4v", help="OpenCV fourcc codec, e.g. mp4v or XVID.")
    parser.add_argument("--output-ext", default=".mp4", help="Output video extension.")
    parser.add_argument("--score-thr", type=float, default=-1.0, help="Skip boxes below this score.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames per video; 0 means all.")
    parser.add_argument("--line-thickness", type=int, default=2, help="Box line thickness.")
    parser.add_argument("--font-scale", type=float, default=0.45, help="Object label font scale.")
    parser.add_argument("--draw-aabb", action="store_true", help="Also draw the wrapped AABB in a faint color.")
    parser.add_argument("--include-invalid", action="store_true", help="Draw records where valid=0.")
    parser.add_argument("--no-text", action="store_true", help="Do not draw object labels or frame headers.")
    parser.add_argument(
        "--resize-width",
        type=int,
        default=0,
        help="Resize output frames to this width while preserving aspect ratio; 0 keeps original size.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned videos without writing files.")
    return parser.parse_args()


def normalize_ext(ext: str) -> str:
    ext = ext.strip()
    if not ext:
        return ".mp4"
    return ext if ext.startswith(".") else f".{ext}"


def split_csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def available_variants(root: Path) -> list[str]:
    images_root = root / "images"
    if not images_root.exists():
        return []
    return sorted(p.name for p in images_root.iterdir() if p.is_dir())


def sequence_dirs(root: Path, variant: str, selected: set[str]) -> list[Path]:
    variant_dir = root / "images" / variant
    if not variant_dir.exists():
        return []
    seqs = [p for p in sorted(variant_dir.iterdir()) if p.is_dir()]
    if selected:
        seqs = [p for p in seqs if p.name in selected]
    return seqs


def image_files(seq_image_dir: Path) -> list[Path]:
    return sorted(p for p in seq_image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def load_records(path: Path, score_thr: float, include_invalid: bool) -> dict[int, list[OrientedRecord]]:
    if not path.exists():
        return {}
    records: dict[int, list[OrientedRecord]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame = int(float(row["frame"]))
                score = float(row.get("score", "1"))
                valid = bool(int(float(row.get("valid", "1"))))
            except (KeyError, ValueError):
                continue
            if score < score_thr:
                continue
            if not valid and not include_invalid:
                continue
            try:
                poly = np.array(
                    [
                        [float(row["poly_x1"]), float(row["poly_y1"])],
                        [float(row["poly_x2"]), float(row["poly_y2"])],
                        [float(row["poly_x3"]), float(row["poly_y3"])],
                        [float(row["poly_x4"]), float(row["poly_y4"])],
                    ],
                    dtype=np.float32,
                )
                aabb = (
                    float(row.get("aabb_x", "0")),
                    float(row.get("aabb_y", "0")),
                    float(row.get("aabb_w", "0")),
                    float(row.get("aabb_h", "0")),
                )
                object_id = int(float(row.get("object_id", "-1")))
                seam_crossing = bool(int(float(row.get("seam_crossing", "0"))))
            except ValueError:
                continue
            record = OrientedRecord(
                frame=frame,
                object_id=object_id,
                class_name=row.get("class_name", "object"),
                score=score,
                polygon=poly,
                aabb=aabb,
                valid=valid,
                seam_crossing=seam_crossing,
            )
            records.setdefault(frame, []).append(record)
    return records


def build_frame_image_map(paths: list[Path], label_frames: Iterable[int]) -> dict[int, Path]:
    if not paths:
        return {}
    labels = sorted(set(int(frame) for frame in label_frames))
    min_label = labels[0] if labels else 1
    numeric: list[tuple[int, Path]] = []
    for path in paths:
        try:
            numeric.append((int(path.stem), path))
        except ValueError:
            continue
    if len(numeric) == len(paths):
        min_image = min(value for value, _ in numeric)
        offset = 1 if min_image == 0 and min_label >= 1 else 0
        if min_image == min_label:
            offset = 0
        return {value + offset: path for value, path in numeric}
    return {min_label + idx: path for idx, path in enumerate(paths)}


def color_for_object(object_id: int) -> tuple[int, int, int]:
    value = int(object_id if object_id >= 0 else 9973)
    r = (37 * value + 89) % 255
    g = (17 * value + 163) % 255
    b = (97 * value + 41) % 255
    return int(b), int(g), int(r)


def wrap_polygon_for_display(poly: np.ndarray, width: int) -> list[np.ndarray]:
    if poly.size == 0:
        return []
    center_x = float(poly[:, 0].mean())
    base_shift = round(center_x / float(width)) * width
    wrapped = poly.copy()
    wrapped[:, 0] -= base_shift
    return [wrapped + np.array([shift, 0.0], dtype=np.float32) for shift in (-width, 0, width)]


def draw_aabb(image: np.ndarray, record: OrientedRecord, color: tuple[int, int, int]) -> None:
    x, y, w, h = record.aabb
    height, width = image.shape[:2]
    for shift in (-width, 0, width):
        x1 = int(round(x + shift))
        y1 = int(round(y))
        x2 = int(round(x + w + shift))
        y2 = int(round(y + h))
        if x2 < 0 or x1 >= width or y2 < 0 or y1 >= height:
            continue
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)


def draw_records(
    image: np.ndarray,
    records: list[OrientedRecord],
    args: argparse.Namespace,
) -> None:
    height, width = image.shape[:2]
    for record in records:
        color = color_for_object(record.object_id)
        if record.seam_crossing:
            color = (0, 255, 255)
        for poly in wrap_polygon_for_display(record.polygon, width):
            if poly[:, 0].max() < 0 or poly[:, 0].min() >= width or poly[:, 1].max() < 0 or poly[:, 1].min() >= height:
                continue
            pts = np.round(poly).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=args.line_thickness, lineType=cv2.LINE_AA)
        if args.draw_aabb:
            draw_aabb(image, record, tuple(int(c * 0.5) for c in color))
        if args.no_text:
            continue
        anchor = wrap_polygon_for_display(record.polygon, width)[1]
        x = int(np.clip(np.min(anchor[:, 0]), 0, max(width - 1, 0)))
        y = int(np.clip(np.min(anchor[:, 1]) - 4, 12, max(height - 1, 12)))
        label = f"{record.class_name}:{record.object_id} {record.score:.2f}"
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, args.font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, args.font_scale, color, 1, cv2.LINE_AA)


def resize_frame(image: np.ndarray, target_width: int) -> np.ndarray:
    if target_width <= 0 or image.shape[1] == target_width:
        return image
    scale = target_width / float(image.shape[1])
    target_height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def draw_header(image: np.ndarray, variant: str, seq: str, frame: int, record_count: int, args: argparse.Namespace) -> None:
    if args.no_text:
        return
    text = f"{variant}/{seq} frame={frame} boxes={record_count}"
    cv2.putText(image, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def open_video_writer(path: Path, frame_size: tuple[int, int], args: argparse.Namespace):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*args.codec[:4])
    writer = cv2.VideoWriter(str(path), fourcc, float(args.fps), frame_size)
    if writer.isOpened():
        return writer
    writer.release()
    fallback = cv2.VideoWriter_fourcc(*"XVID")
    fallback_path = path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(fallback_path), fallback, float(args.fps), frame_size)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open VideoWriter for {path} or {fallback_path}")
    print(f"[WARN] codec={args.codec} failed for {path}; wrote AVI fallback: {fallback_path}")
    return writer


def visualize_sequence(benchmark_root: Path, variant: str, seq_dir: Path, args: argparse.Namespace) -> dict[str, object] | None:
    label_path = benchmark_root / args.label_kind / "oriented_csv" / variant / f"{seq_dir.name}.csv"
    records_by_frame = load_records(label_path, args.score_thr, args.include_invalid)
    if not records_by_frame:
        print(f"[WARN] missing or empty labels: {label_path}")
        return None

    images = image_files(seq_dir)
    if not images:
        print(f"[WARN] no images under: {seq_dir}")
        return None

    frame_to_image = build_frame_image_map(images, records_by_frame.keys())
    frame_ids = sorted(frame for frame in frame_to_image if frame in records_by_frame or not records_by_frame)
    if args.max_frames:
        frame_ids = frame_ids[: args.max_frames]
    if not frame_ids:
        print(f"[WARN] no image/label frame overlap: images={seq_dir}, labels={label_path}")
        return None

    first = cv2.imread(str(frame_to_image[frame_ids[0]]), cv2.IMREAD_COLOR)
    if first is None:
        print(f"[WARN] could not read first image: {frame_to_image[frame_ids[0]]}")
        return None
    first = resize_frame(first, args.resize_width)
    frame_size = (first.shape[1], first.shape[0])
    video_path = (args.out_root or (benchmark_root / "visualizations")) / variant / f"{seq_dir.name}{args.output_ext}"
    if args.dry_run:
        print(f"[DRY] {variant}/{seq_dir.name}: frames={len(frame_ids)}, labels={label_path}, video={video_path}")
        return {"sequence": seq_dir.name, "variant": variant, "frames": len(frame_ids), "video": str(video_path)}

    writer = open_video_writer(video_path, frame_size, args)
    written = 0
    for frame in frame_ids:
        image = cv2.imread(str(frame_to_image[frame]), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARN] skip unreadable image: {frame_to_image[frame]}")
            continue
        records = records_by_frame.get(frame, [])
        draw_records(image, records, args)
        draw_header(image, variant, seq_dir.name, frame, len(records), args)
        image = resize_frame(image, args.resize_width)
        if image.shape[1] != frame_size[0] or image.shape[0] != frame_size[1]:
            image = cv2.resize(image, frame_size, interpolation=cv2.INTER_AREA)
        writer.write(image)
        written += 1
    writer.release()
    print(f"[OK] {variant}/{seq_dir.name}: frames={written}, video={video_path}")
    return {"sequence": seq_dir.name, "variant": variant, "frames": written, "video": str(video_path)}


def main() -> None:
    if cv2 is None:
        raise SystemExit("OpenCV is required. Install with: python -m pip install opencv-python")
    args = parse_args()
    args.output_ext = normalize_ext(args.output_ext)
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.line_thickness < 1:
        raise SystemExit("--line-thickness must be >= 1")

    variants = available_variants(args.benchmark_root)
    selected_variants = split_csv(args.variants)
    if selected_variants:
        variants = [variant for variant in variants if variant in selected_variants]
    if not variants:
        raise SystemExit(f"No image variants found under {args.benchmark_root / 'images'}")

    selected_seqs = split_csv(args.seqs)
    summaries: list[dict[str, object]] = []
    for variant in variants:
        seq_dirs = sequence_dirs(args.benchmark_root, variant, selected_seqs)
        if not seq_dirs:
            print(f"[WARN] no sequences for variant: {variant}")
            continue
        for seq_dir in seq_dirs:
            result = visualize_sequence(args.benchmark_root, variant, seq_dir, args)
            if result is not None:
                summaries.append(result)

    out_root = args.out_root or (args.benchmark_root / "visualizations")
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "visualization_manifest.json"
    if not args.dry_run:
        manifest_path.write_text(json.dumps({"videos": summaries}, indent=2), encoding="utf-8")
        print(f"[DONE] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
