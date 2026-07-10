from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pano_geometry import (
    distortion_score_from_y,
    make_equirectangular_remap,
    pixel_to_xyz,
    polygon_aabb,
    rotate_points,
    rotation_between_vectors,
    rotation_matrix_zyx,
    sample_xyxy_edges,
    unwrap_x,
)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


ORIENTED_HEADER = [
    "frame",
    "object_id",
    "class_name",
    "cx_unwrapped",
    "cy",
    "w",
    "h",
    "angle_rad",
    "score",
    "poly_x1",
    "poly_y1",
    "poly_x2",
    "poly_y2",
    "poly_x3",
    "poly_y3",
    "poly_x4",
    "poly_y4",
    "aabb_x",
    "aabb_y",
    "aabb_w",
    "aabb_h",
    "distortion_score",
    "seam_crossing",
    "valid",
]


@dataclass(frozen=True)
class Detection:
    frame: int
    object_id: int
    class_name: str
    xyxy: tuple[float, float, float, float]
    score: float
    frame_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert QuadTrack/MOT detections to a spherical-rotation oriented benchmark."
    )
    parser.add_argument("--quadtrack-root", type=Path, required=True, help="Root containing QuadTrack JSONs or detection_results_mot.")
    parser.add_argument("--out-root", type=Path, required=True, help="Output benchmark root.")
    parser.add_argument("--image-root", type=Path, default=None, help="Optional panorama image root.")
    parser.add_argument("--det-root", type=Path, default=None, help="Optional MOT txt root. Defaults to quadtrack-root/detection_results_mot.")
    parser.add_argument("--input-format", choices=["auto", "mot", "quadtrack_json"], default="auto")
    parser.add_argument("--image-width", type=int, default=2048)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument(
        "--variants",
        default="prior_a2b,polar_up,target_north_80",
        help=(
            "Comma-separated variants. Built-ins: prior_a2b, prior_b2a, "
            "polar_up, polar_down, target_north_80, target_south_80, custom."
        ),
    )
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="Custom Rz angle for variant custom.")
    parser.add_argument("--pitch-deg", type=float, default=0.0, help="Custom Ry angle for variant custom.")
    parser.add_argument("--roll-deg", type=float, default=0.0, help="Custom Rx angle for variant custom.")
    parser.add_argument("--edge-samples", type=int, default=32, help="Samples per original box side.")
    parser.add_argument("--min-score", type=float, default=-1.0)
    parser.add_argument("--input-kind", choices=["detections", "gt"], default="detections")
    parser.add_argument("--rotate-images", action="store_true", help="Write rotated images when --image-root is available.")
    parser.add_argument("--image-ext", default=".jpg", help="Output image extension.")
    parser.add_argument("--limit-seqs", type=int, default=0)
    parser.add_argument("--limit-frames", type=int, default=0)
    return parser.parse_args()


def mot_files(det_root: Path) -> list[Path]:
    return sorted(p for p in det_root.glob("*.txt") if p.is_file())


def json_files(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*.json") if p.is_file() and p.name != "conversion_manifest.json")


def load_mot_file(path: Path) -> list[Detection]:
    rows: list[Detection] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            try:
                frame = int(float(parts[0]))
                object_id = int(float(parts[1]))
                x, y, w, h = [float(v) for v in parts[2:6]]
                score = float(parts[6])
            except ValueError:
                continue
            frame_file = f"{max(frame - 1, 0):06d}.jpg"
            rows.append(
                Detection(
                    frame=frame,
                    object_id=object_id,
                    class_name="person",
                    xyxy=(x, y, x + w, y + h),
                    score=score,
                    frame_file=frame_file,
                )
            )
    return rows


def parse_label_id(label_id: str) -> tuple[str, int]:
    if ":" not in label_id:
        return label_id or "object", -1
    cls, obj = label_id.rsplit(":", 1)
    try:
        return cls or "object", int(float(obj))
    except ValueError:
        return cls or "object", -1


def load_quadtrack_json(path: Path) -> list[Detection]:
    data = json.loads(path.read_text(encoding="utf-8"))
    detections = data.get("detections", data)
    rows: list[Detection] = []
    if not isinstance(detections, dict):
        return rows
    for frame_file, dets in sorted(detections.items()):
        stem = Path(frame_file).stem
        try:
            frame = int(stem) + 1
        except ValueError:
            frame = len(rows) + 1
        for det in dets:
            box = det.get("box")
            if not box or len(box) < 4:
                continue
            class_name, object_id = parse_label_id(str(det.get("label_id", "object:-1")))
            score = float(det.get("score", 1.0))
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            rows.append(
                Detection(
                    frame=frame,
                    object_id=object_id,
                    class_name=class_name,
                    xyxy=(x1, y1, x2, y2),
                    score=score,
                    frame_file=frame_file,
                )
            )
    return rows


def choose_input_files(args: argparse.Namespace) -> tuple[str, list[Path]]:
    det_root = args.det_root or (args.quadtrack_root / "detection_results_mot")
    if args.input_format in {"auto", "mot"} and det_root.exists():
        files = mot_files(det_root)
        if files or args.input_format == "mot":
            return "mot", files
    files = json_files(args.quadtrack_root)
    return "quadtrack_json", files


def load_sequence(path: Path, input_format: str) -> list[Detection]:
    if input_format == "mot":
        return load_mot_file(path)
    return load_quadtrack_json(path)


def variant_rotation(name: str, detections: list[Detection], width: int, height: int, args: argparse.Namespace) -> np.ndarray:
    key = name.lower()
    if key == "prior_a2b":
        return rotation_matrix_zyx(0.0, 0.0, -math.pi / 2.0)
    if key == "prior_b2a":
        return rotation_matrix_zyx(0.0, 0.0, math.pi / 2.0)
    if key == "polar_up":
        return rotation_matrix_zyx(0.0, math.radians(75.0), 0.0)
    if key == "polar_down":
        return rotation_matrix_zyx(0.0, math.radians(-75.0), 0.0)
    if key == "custom":
        return rotation_matrix_zyx(
            math.radians(args.yaw_deg),
            math.radians(args.pitch_deg),
            math.radians(args.roll_deg),
        )
    if key.startswith("target_north_") or key.startswith("target_south_"):
        return target_pole_rotation(key, detections, width, height)
    raise ValueError(f"unknown rotation variant: {name}")


def target_pole_rotation(key: str, detections: list[Detection], width: int, height: int) -> np.ndarray:
    centers = []
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        centers.append([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
    if not centers:
        return np.eye(3, dtype=np.float64)
    centers_arr = np.asarray(centers, dtype=np.float64)
    source_dirs = pixel_to_xyz(centers_arr[:, 0], centers_arr[:, 1], width, height)
    source = source_dirs.mean(axis=0)
    source = source / max(float(np.linalg.norm(source)), 1e-12)
    degrees = float(key.rsplit("_", 1)[-1])
    lat = math.radians(abs(degrees))
    if key.startswith("target_south_"):
        lat = -lat
    target = np.array([math.cos(lat), 0.0, math.sin(lat)], dtype=np.float64)
    return rotation_between_vectors(source, target)


def fit_min_area_rect(points_unwrapped: np.ndarray) -> tuple[float, float, float, float, float, np.ndarray]:
    if cv2 is None:
        from src.pano_geometry import pca_oriented_box

        box = pca_oriented_box(points_unwrapped)
        poly = obb_to_polygon(box.cx, box.cy, box.w, box.h, box.angle_rad)
        return box.cx, box.cy, box.w, box.h, box.angle_rad, poly
    rect = cv2.minAreaRect(points_unwrapped.astype(np.float32))
    (cx, cy), (w, h), angle_deg = rect
    if w < h:
        w, h = h, w
        angle_deg += 90.0
    angle_rad = normalize_angle_pi(math.radians(angle_deg))
    poly = cv2.boxPoints(((cx, cy), (w, h), math.degrees(angle_rad))).astype(np.float64)
    return float(cx), float(cy), float(w), float(h), float(angle_rad), poly


def normalize_angle_pi(angle: float) -> float:
    return float((angle + math.pi / 2.0) % math.pi - math.pi / 2.0)


def obb_to_polygon(cx: float, cy: float, w: float, h: float, angle: float) -> np.ndarray:
    ca, sa = math.cos(angle), math.sin(angle)
    local = np.array(
        [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]],
        dtype=np.float64,
    )
    rot = np.array([[ca, -sa], [sa, ca]], dtype=np.float64)
    return local @ rot.T + np.array([cx, cy], dtype=np.float64)


def rotate_detection(det: Detection, rotation: np.ndarray, width: int, height: int, edge_samples: int) -> dict[str, object]:
    x1, y1, x2, y2 = det.xyxy
    edge = sample_xyxy_edges(x1, y1, x2, y2, samples_per_side=edge_samples)
    rotated = rotate_points(edge, width, height, rotation)
    seam_crossing = bool(np.ptp(rotated[:, 0]) > width * 0.5)
    unwrapped = unwrap_x(rotated, width)
    cx, cy, w, h, angle, poly = fit_min_area_rect(unwrapped)
    aabb_x1, aabb_y1, aabb_x2, aabb_y2 = polygon_aabb(poly)
    valid = bool(w > 1.0 and h > 1.0 and np.isfinite(poly).all())
    distortion = distortion_score_from_y(cy, height)
    return {
        "frame": det.frame,
        "object_id": det.object_id,
        "class_name": det.class_name,
        "cx_unwrapped": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "angle_rad": angle,
        "score": det.score,
        "polygon": poly,
        "aabb": (aabb_x1, aabb_y1, aabb_x2 - aabb_x1, aabb_y2 - aabb_y1),
        "distortion_score": distortion,
        "seam_crossing": seam_crossing,
        "valid": valid,
    }


def oriented_row(record: dict[str, object]) -> list[object]:
    poly = np.asarray(record["polygon"], dtype=np.float64).reshape(4, 2)
    aabb = record["aabb"]
    return [
        record["frame"],
        record["object_id"],
        record["class_name"],
        fmt(record["cx_unwrapped"]),
        fmt(record["cy"]),
        fmt(record["w"]),
        fmt(record["h"]),
        fmt(record["angle_rad"]),
        fmt(record["score"]),
        *[fmt(v) for v in poly.reshape(-1)],
        *[fmt(v) for v in aabb],
        fmt(record["distortion_score"]),
        int(bool(record["seam_crossing"])),
        int(bool(record["valid"])),
    ]


def fmt(value: object) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.6f}"


def mot_aabb_row(record: dict[str, object]) -> list[object]:
    x, y, w, h = record["aabb"]
    return [
        int(record["frame"]),
        int(record["object_id"]),
        f"{float(x):.3f}",
        f"{float(y):.3f}",
        f"{float(w):.3f}",
        f"{float(h):.3f}",
        f"{float(record['score']):.6f}",
        -1,
        -1,
        -1,
    ]


def write_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))
            count += 1
    return count


def write_mot_txt(path: Path, rows: Iterable[Iterable[object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")
            count += 1
    return count


def find_image(image_root: Path, seq: str, frame_file: str, frame: int) -> Path | None:
    candidates = [
        image_root / seq / frame_file,
        image_root / seq / "img1" / frame_file,
        image_root / seq / "images" / frame_file,
        image_root / frame_file,
        image_root / seq / f"{max(frame - 1, 0):06d}.jpg",
        image_root / seq / f"{max(frame - 1, 0):06d}.png",
        image_root / f"{seq}_{max(frame - 1, 0):06d}.jpg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def rotate_images_for_sequence(
    detections: list[Detection],
    rotation: np.ndarray,
    seq: str,
    variant: str,
    args: argparse.Namespace,
) -> tuple[int, tuple[int, int]]:
    if not args.rotate_images or args.image_root is None:
        return 0, (args.image_width, args.image_height)
    if cv2 is None:
        raise RuntimeError("OpenCV is required for --rotate-images")
    by_frame: dict[int, Detection] = {}
    for det in detections:
        by_frame.setdefault(det.frame, det)
    written = 0
    out_dir = args.out_root / "images" / variant / seq
    out_dir.mkdir(parents=True, exist_ok=True)
    actual_size = (args.image_width, args.image_height)
    for frame, det in sorted(by_frame.items()):
        if args.limit_frames and frame > args.limit_frames:
            continue
        src = find_image(args.image_root, seq, det.frame_file, frame)
        if src is None:
            continue
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        actual_size = (width, height)
        map_x, map_y = make_equirectangular_remap(width, height, rotation)
        rotated = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        dst = out_dir / (Path(det.frame_file).stem + args.image_ext)
        cv2.imwrite(str(dst), rotated)
        written += 1
    return written, actual_size


def process_sequence(path: Path, input_format: str, args: argparse.Namespace, variants: list[str]) -> list[dict[str, object]]:
    seq = path.stem
    detections = [d for d in load_sequence(path, input_format) if d.score >= args.min_score]
    if args.limit_frames:
        detections = [d for d in detections if d.frame <= args.limit_frames]
    summary: list[dict[str, object]] = []
    for variant in variants:
        rotation = variant_rotation(variant, detections, args.image_width, args.image_height, args)
        image_count, image_size = rotate_images_for_sequence(detections, rotation, seq, variant, args)
        width, height = image_size
        records = [rotate_detection(det, rotation, width, height, args.edge_samples) for det in detections]
        oriented_path = args.out_root / args.input_kind / "oriented_csv" / variant / f"{seq}.csv"
        aabb_path = args.out_root / args.input_kind / "mot_aabb" / variant / f"{seq}.txt"
        oriented_count = write_csv(oriented_path, ORIENTED_HEADER, (oriented_row(r) for r in records))
        aabb_count = write_mot_txt(aabb_path, (mot_aabb_row(r) for r in records))
        summary.append(
            {
                "sequence": seq,
                "variant": variant,
                "input_records": len(detections),
                "oriented_records": oriented_count,
                "aabb_records": aabb_count,
                "rotated_images": image_count,
                "image_width": width,
                "image_height": height,
                "rotation_matrix": rotation.tolist(),
                "oriented_csv": str(oriented_path),
                "mot_aabb": str(aabb_path),
            }
        )
        print(f"[OK] {seq}/{variant}: labels={oriented_count}, images={image_count}")
    return summary


def main() -> None:
    args = parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    input_format, files = choose_input_files(args)
    if args.limit_seqs:
        files = files[: args.limit_seqs]
    if not files:
        raise SystemExit(f"No input files found under {args.quadtrack_root}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    all_summary: list[dict[str, object]] = []
    for path in files:
        all_summary.extend(process_sequence(path, input_format, args, variants))
    manifest = {
        "source": "QuadTrack conversion with PriOr-Flow-style spherical rotation",
        "input_format": input_format,
        "quadtrack_root": str(args.quadtrack_root),
        "image_root": str(args.image_root) if args.image_root else None,
        "variants": variants,
        "edge_samples": args.edge_samples,
        "input_kind": args.input_kind,
        "summary": all_summary,
    }
    manifest_path = args.out_root / "orientation_benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
