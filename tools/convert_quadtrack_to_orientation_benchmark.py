from __future__ import annotations

import argparse
import configparser
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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
        description="Convert QuadTrack/MOT detections to a spherical-rotation oriented benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # QuadTrack default layout: <root>/detection_results_mot/*.txt
  python -B tools/convert_quadtrack_to_orientation_benchmark.py \\
    --quadtrack-root /data/QuadTrack_test/OmniTrack_Omnidet_test \\
    --out-root outputs/quadtrack_orientation_benchmark \\
    --image-width 2048 --image-height 480 \\
    --vertical-fov-deg 120 \\
    --variants prior_a2b,polar_up,target_north_80

  # If image files are named 000001.jpg instead of 000000.jpg for frame 1:
  python -B tools/convert_quadtrack_to_orientation_benchmark.py \\
    --quadtrack-root /data/QuadTrack_test/OmniTrack_Omnidet_test \\
    --out-root outputs/quadtrack_orientation_benchmark \\
    --vertical-fov-deg 120 \\
    --mot-frame-to-image-offset 0

  # Official MOTChallenge/DanceTrack-like layout: <root>/<sequence>/seqinfo.ini, img1, gt or det
  python -B tools/convert_quadtrack_to_orientation_benchmark.py \\
    --quadtrack-root /data/QuadTrack/test \\
    --out-root outputs/quadtrack_orientation_benchmark \\
    --vertical-fov-deg 120 \\
    --variants prior_a2b,polar_up,target_north_80
""",
    )
    parser.add_argument(
        "--quadtrack-root",
        type=Path,
        required=True,
        help=(
            "QuadTrack root. For MOT txt input this usually contains "
            "detection_results_mot/*.txt; for JSON input it contains *.json."
        ),
    )
    parser.add_argument("--out-root", type=Path, required=True, help="Output benchmark root.")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help=(
            "Optional panorama image root. The script searches common layouts: "
            "<image-root>/<seq>/<frame>, <seq>/img1/<frame>, <seq>/images/<frame>, "
            "and flat <image-root>/<frame>."
        ),
    )
    parser.add_argument(
        "--det-root",
        type=Path,
        default=None,
        help="Optional MOT txt root. Defaults to <quadtrack-root>/detection_results_mot.",
    )
    parser.add_argument(
        "--seq-glob",
        default=None,
        help=(
            "Optional file glob for sequence inputs. Defaults to *.txt for MOT "
            "and *.json for QuadTrack JSON. Example: --seq-glob 'scene_*.txt'."
        ),
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "mot", "motchallenge", "quadtrack_json"],
        default="auto",
        help=(
            "Input label format. auto tries detection_results_mot/*.txt, then "
            "MOTChallenge/DanceTrack-like sequence folders, then JSON."
        ),
    )
    parser.add_argument(
        "--label-source",
        choices=["auto", "gt", "det"],
        default="auto",
        help=(
            "For MOTChallenge/DanceTrack-like sequence folders, choose gt/gt.txt "
            "or det/det.txt. auto prefers gt when present, otherwise det."
        ),
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=2048,
        help=(
            "ERP panorama width used for label-only conversion. If --image-root is "
            "provided and images can be read, the real image size is used instead."
        ),
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=480,
        help=(
            "ERP panorama height used for label-only conversion. Must match the "
            "coordinate system of the input boxes."
        ),
    )
    parser.add_argument(
        "--vertical-fov-deg",
        type=float,
        default=120.0,
        help=(
            "Vertical angular coverage of the panorama image. QuadTrack uses 120 deg. "
            "Use 180 for original full-ERP PriOr-Flow behavior."
        ),
    )
    parser.add_argument(
        "--variants",
        default="prior_a2b,polar_up,target_north_80",
        help=(
            "Comma-separated variants. Built-ins: prior_a2b, prior_b2a, "
            "polar_up, polar_down, target_north_80, target_south_80, custom."
        ),
    )
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="Custom Rz angle in degrees for --variants custom.")
    parser.add_argument("--pitch-deg", type=float, default=0.0, help="Custom Ry angle in degrees for --variants custom.")
    parser.add_argument("--roll-deg", type=float, default=0.0, help="Custom Rx angle in degrees for --variants custom.")
    parser.add_argument(
        "--edge-samples",
        type=int,
        default=32,
        help=(
            "Samples per original box side before spherical rotation. Increase to "
            "64 for very large/polar boxes; 16 is usually enough for quick checks."
        ),
    )
    parser.add_argument("--min-score", type=float, default=-1.0, help="Drop detections with score below this threshold.")
    parser.add_argument(
        "--input-kind",
        choices=["detections", "gt"],
        default="detections",
        help="Output namespace under <out-root>. Use gt when converting ground-truth labels.",
    )
    parser.add_argument(
        "--mot-frame-to-image-offset",
        type=int,
        default=-1,
        help=(
            "For MOT txt input, synthesized image index = MOT frame + offset. "
            "Default -1 maps frame 1 to 000000.jpg, matching common QuadTrack exports. "
            "Use 0 when images are 000001.jpg for frame 1."
        ),
    )
    parser.add_argument(
        "--json-frame-offset",
        type=int,
        default=1,
        help=(
            "For QuadTrack JSON input, output frame = numeric image stem + offset. "
            "Default 1 maps 000000.jpg to frame 1. Use 0 when JSON keys are already one-based."
        ),
    )
    parser.add_argument(
        "--frame-name-width",
        type=int,
        default=6,
        help="Zero padding width when synthesizing MOT image names, for example 6 -> 000001.jpg.",
    )
    parser.add_argument(
        "--frame-image-ext",
        default=".jpg",
        help="Input image extension when synthesizing MOT image names. Use .png for PNG datasets.",
    )
    parser.add_argument("--mot-class-name", default="person", help="Class name assigned to MOT txt rows.")
    parser.add_argument("--rotate-images", action="store_true", help="Write rotated images when --image-root is available.")
    parser.add_argument("--image-ext", default=".jpg", help="Output image extension.")
    parser.add_argument("--limit-seqs", type=int, default=0, help="Process only the first N sequence files; 0 means all.")
    parser.add_argument("--limit-frames", type=int, default=0, help="Process only frames <= N; 0 means all.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_width <= 0 or args.image_height <= 0:
        raise SystemExit("--image-width and --image-height must be positive")
    if not (0.0 < args.vertical_fov_deg <= 180.0):
        raise SystemExit("--vertical-fov-deg must be in (0, 180]")
    if args.edge_samples < 2:
        raise SystemExit("--edge-samples must be >= 2")
    if args.frame_name_width < 1:
        raise SystemExit("--frame-name-width must be >= 1")
    args.frame_image_ext = normalize_ext(args.frame_image_ext)
    args.image_ext = normalize_ext(args.image_ext)


def normalize_ext(ext: str) -> str:
    ext = ext.strip()
    if not ext:
        return ".jpg"
    return ext if ext.startswith(".") else f".{ext}"


def vertical_fov_rad(args: argparse.Namespace) -> float:
    return math.radians(float(getattr(args, "vertical_fov_deg", 180.0)))


def mot_files(det_root: Path, seq_glob: str | None = None) -> list[Path]:
    return sorted(p for p in det_root.glob(seq_glob or "*.txt") if p.is_file())


def json_files(root: Path, seq_glob: str | None = None) -> list[Path]:
    return sorted(
        p
        for p in root.glob(seq_glob or "*.json")
        if p.is_file() and p.name not in {"conversion_manifest.json", "orientation_benchmark_manifest.json"}
    )


def read_seqinfo(seq_dir: Path) -> dict[str, str]:
    path = seq_dir / "seqinfo.ini"
    cfg = configparser.ConfigParser()
    if path.exists():
        cfg.read(path, encoding="utf-8")
    section = cfg["Sequence"] if cfg.has_section("Sequence") else {}
    return {
        "name": str(section.get("name", seq_dir.name)),
        "imDir": str(section.get("imDir", "img1")),
        "imExt": str(section.get("imExt", ".jpg")),
        "imWidth": str(section.get("imWidth", "0")),
        "imHeight": str(section.get("imHeight", "0")),
        "seqLength": str(section.get("seqLength", "0")),
        "nameLength": str(section.get("nameLength", "8")),
    }


def motchallenge_sequence_candidates(root: Path) -> list[Path]:
    if (root / "seqinfo.ini").exists() or (root / "img1").exists():
        return [root]
    if not root.exists():
        return []
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir()
        and (
            (p / "seqinfo.ini").exists()
            or (p / "img1").exists()
            or (p / "gt" / "gt.txt").exists()
            or (p / "det" / "det.txt").exists()
        )
    )


def motchallenge_label_file(seq_dir: Path, label_source: str) -> tuple[str | None, Path | None]:
    gt_path = seq_dir / "gt" / "gt.txt"
    det_path = seq_dir / "det" / "det.txt"
    if label_source == "gt":
        return ("gt", gt_path) if gt_path.exists() else (None, None)
    if label_source == "det":
        return ("det", det_path) if det_path.exists() else (None, None)
    if gt_path.exists():
        return "gt", gt_path
    if det_path.exists():
        return "det", det_path
    return None, None


def motchallenge_sequence_dirs(root: Path, label_source: str, seq_glob: str | None = None) -> list[Path]:
    seqs = motchallenge_sequence_candidates(root)
    if seq_glob:
        seqs = [seq for seq in seqs if seq.match(seq_glob) or seq.name == seq_glob]
    return [seq for seq in seqs if motchallenge_label_file(seq, label_source)[1] is not None]


def motchallenge_frame_file(frame: int, seqinfo: dict[str, str]) -> str:
    name_length = int(float(seqinfo.get("nameLength", "8") or 8))
    im_ext = normalize_ext(seqinfo.get("imExt", ".jpg") or ".jpg")
    return f"{frame:0{name_length}d}{im_ext}"


def motchallenge_image_index(seq_dir: Path, seqinfo: dict[str, str]) -> dict[int, str]:
    im_dir = seqinfo.get("imDir", "img1") or "img1"
    image_dirs: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for rel in [Path(im_dir), Path("img1"), Path("images")]:
        image_dir = seq_dir / rel
        if image_dir.exists() and image_dir.is_dir() and image_dir not in seen:
            image_dirs.append((rel, image_dir))
            seen.add(image_dir)
    if not image_dirs:
        return {}

    im_ext = normalize_ext(seqinfo.get("imExt", ".jpg") or ".jpg").lower()

    rel_dir, image_dir = image_dirs[0]
    files: list[Path] = []
    for candidate_rel, candidate_dir in image_dirs:
        candidate_files = sorted(
            p
            for p in candidate_dir.iterdir()
            if p.is_file() and (p.suffix.lower() == im_ext or p.suffix.lower() in IMAGE_EXTENSIONS)
        )
        if candidate_files:
            rel_dir, image_dir, files = candidate_rel, candidate_dir, candidate_files
            break
    if not files:
        return {}

    numeric: list[tuple[int, Path]] = []
    for path in files:
        try:
            numeric.append((int(path.stem), path))
        except ValueError:
            continue

    mapping: dict[int, str] = {}
    if numeric:
        offset = 1 if min(value for value, _ in numeric) == 0 else 0
        for value, path in numeric:
            mapping[value + offset] = str(rel_dir / path.name)
            if offset == 1:
                mapping.setdefault(value, str(rel_dir / path.name))
        return mapping

    for frame, path in enumerate(files, start=1):
        mapping[frame] = str(rel_dir / path.name)
    return mapping


def load_mot_file(path: Path, args: argparse.Namespace) -> list[Detection]:
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
            image_index = max(frame + args.mot_frame_to_image_offset, 0)
            frame_file = f"{image_index:0{args.frame_name_width}d}{args.frame_image_ext}"
            rows.append(
                Detection(
                    frame=frame,
                    object_id=object_id,
                    class_name=args.mot_class_name,
                    xyxy=(x, y, x + w, y + h),
                    score=score,
                    frame_file=frame_file,
                )
            )
    return rows


def load_motchallenge_sequence(seq_dir: Path, args: argparse.Namespace) -> list[Detection]:
    source, path = motchallenge_label_file(seq_dir, args.label_source)
    if source is None or path is None:
        return []
    seqinfo = read_seqinfo(seq_dir)
    image_index = motchallenge_image_index(seq_dir, seqinfo)
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
            rows.append(
                Detection(
                    frame=frame,
                    object_id=object_id,
                    class_name=args.mot_class_name,
                    xyxy=(x, y, x + w, y + h),
                    score=score,
                    frame_file=image_index.get(frame, motchallenge_frame_file(frame, seqinfo)),
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


def load_quadtrack_json(path: Path, args: argparse.Namespace) -> list[Detection]:
    data = json.loads(path.read_text(encoding="utf-8"))
    detections = data.get("detections", data)
    rows: list[Detection] = []
    if not isinstance(detections, dict):
        return rows
    frame_counter = 0
    for frame_file, dets in sorted(detections.items()):
        frame_counter += 1
        stem = Path(frame_file).stem
        try:
            frame = int(stem) + args.json_frame_offset
        except ValueError:
            frame = frame_counter
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
    if args.input_format == "mot":
        return "mot", mot_files(det_root, args.seq_glob) if det_root.exists() else []
    if args.input_format == "motchallenge":
        return "motchallenge", motchallenge_sequence_dirs(args.quadtrack_root, args.label_source, args.seq_glob)
    if args.input_format == "quadtrack_json":
        return "quadtrack_json", json_files(args.quadtrack_root, args.seq_glob)
    if det_root.exists():
        files = mot_files(det_root, args.seq_glob)
        if files:
            return "mot", files
    seq_dirs = motchallenge_sequence_dirs(args.quadtrack_root, args.label_source, args.seq_glob)
    if seq_dirs:
        return "motchallenge", seq_dirs
    return "quadtrack_json", json_files(args.quadtrack_root, args.seq_glob)


def load_sequence(path: Path, input_format: str, args: argparse.Namespace) -> list[Detection]:
    if input_format == "mot":
        return load_mot_file(path, args)
    if input_format == "motchallenge":
        return load_motchallenge_sequence(path, args)
    return load_quadtrack_json(path, args)


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
        return target_pole_rotation(key, detections, width, height, args)
    raise ValueError(f"unknown rotation variant: {name}")


def target_pole_rotation(
    key: str,
    detections: list[Detection],
    width: int,
    height: int,
    args: argparse.Namespace,
) -> np.ndarray:
    centers = []
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        centers.append([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
    if not centers:
        return np.eye(3, dtype=np.float64)
    centers_arr = np.asarray(centers, dtype=np.float64)
    source_dirs = pixel_to_xyz(
        centers_arr[:, 0],
        centers_arr[:, 1],
        width,
        height,
        vertical_fov_rad=vertical_fov_rad(args),
    )
    source = source_dirs.mean(axis=0)
    source = source / max(float(np.linalg.norm(source)), 1e-12)
    degrees = float(key.rsplit("_", 1)[-1])
    lat = math.radians(abs(degrees))
    if key.startswith("target_south_"):
        lat = -lat
    target = np.array([math.cos(lat), 0.0, math.sin(lat)], dtype=np.float64)
    # PriOr-Flow img_rotate uses source = R @ output, so image content moves as
    # output = R.T @ source. To place source near target in the output image,
    # choose R such that R.T @ source = target, equivalently R @ target = source.
    return rotation_between_vectors(target, source)


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
    return rotate_detection_with_projection(det, rotation, width, height, edge_samples, math.pi)


def rotate_detection_with_projection(
    det: Detection,
    rotation: np.ndarray,
    width: int,
    height: int,
    edge_samples: int,
    vertical_fov: float,
) -> dict[str, object]:
    x1, y1, x2, y2 = det.xyxy
    edge = sample_xyxy_edges(x1, y1, x2, y2, samples_per_side=edge_samples)
    rotated = rotate_points(edge, width, height, rotation, vertical_fov_rad=vertical_fov)
    seam_crossing = bool(np.ptp(rotated[:, 0]) > width * 0.5)
    unwrapped = unwrap_x(rotated, width)
    cx, cy, w, h, angle, poly = fit_min_area_rect(unwrapped)
    aabb_x1, aabb_y1, aabb_x2, aabb_y2 = polygon_aabb(poly)
    valid = bool(w > 1.0 and h > 1.0 and np.isfinite(poly).all())
    distortion = distortion_score_from_y(cy, height, vertical_fov_rad=vertical_fov)
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
    frame_path = Path(frame_file)
    candidates = [
        image_root / seq / frame_path,
        image_root / seq / "img1" / frame_path,
        image_root / seq / "images" / frame_path,
        image_root / "img1" / frame_path,
        image_root / "images" / frame_path,
        image_root / frame_path,
        image_root / seq / f"{max(frame - 1, 0):06d}.jpg",
        image_root / seq / f"{max(frame - 1, 0):06d}.png",
        image_root / seq / "img1" / f"{max(frame - 1, 0):06d}.jpg",
        image_root / seq / "img1" / f"{frame:06d}.jpg",
        image_root / seq / "img1" / f"{frame:08d}.jpg",
        image_root / seq / "img1" / f"{frame:08d}.png",
        image_root / f"{seq}_{max(frame - 1, 0):06d}.jpg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def infer_sequence_image_size(detections: list[Detection], seq: str, args: argparse.Namespace) -> tuple[int, int]:
    if args.image_root is None or cv2 is None:
        return args.image_width, args.image_height
    for det in sorted(detections, key=lambda item: item.frame):
        src = find_image(args.image_root, seq, det.frame_file, det.frame)
        if src is None:
            continue
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        return int(width), int(height)
    return args.image_width, args.image_height


def motchallenge_size(seq_dir: Path, args: argparse.Namespace) -> tuple[int, int]:
    seqinfo = read_seqinfo(seq_dir)
    width = int(args.image_width or float(seqinfo.get("imWidth", "0") or 0))
    height = int(args.image_height or float(seqinfo.get("imHeight", "0") or 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"Missing image size for {seq_dir}; pass --image-width and --image-height.")
    return width, height


def helper_args_for_sequence(path: Path, input_format: str, args: argparse.Namespace) -> argparse.Namespace:
    helper = argparse.Namespace(**vars(args))
    if input_format == "motchallenge":
        width, height = motchallenge_size(path, args)
        helper.image_width = width
        helper.image_height = height
        if helper.image_root is None:
            helper.image_root = path.parent
        if helper.image_ext in {"", None}:
            helper.image_ext = normalize_ext(read_seqinfo(path).get("imExt", ".jpg") or ".jpg")
    return helper


def rotate_images_for_sequence(
    detections: list[Detection],
    rotation: np.ndarray,
    seq: str,
    variant: str,
    args: argparse.Namespace,
    image_size: tuple[int, int],
) -> tuple[int, tuple[int, int]]:
    if not args.rotate_images or args.image_root is None:
        return 0, image_size
    if cv2 is None:
        raise RuntimeError("OpenCV is required for --rotate-images")
    by_frame: dict[int, Detection] = {}
    for det in detections:
        by_frame.setdefault(det.frame, det)
    written = 0
    out_dir = args.out_root / "images" / variant / seq
    out_dir.mkdir(parents=True, exist_ok=True)
    actual_size = image_size
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
        map_x, map_y = make_equirectangular_remap(width, height, rotation, vertical_fov_rad=vertical_fov_rad(args))
        rotated = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        dst = out_dir / (Path(det.frame_file).stem + args.image_ext)
        cv2.imwrite(str(dst), rotated)
        written += 1
    return written, actual_size


def process_sequence(path: Path, input_format: str, args: argparse.Namespace, variants: list[str]) -> list[dict[str, object]]:
    seq = path.name if input_format == "motchallenge" else path.stem
    helper_args = helper_args_for_sequence(path, input_format, args)
    detections = [d for d in load_sequence(path, input_format, args) if d.score >= args.min_score]
    if args.limit_frames:
        detections = [d for d in detections if d.frame <= args.limit_frames]
    base_size = infer_sequence_image_size(detections, seq, helper_args)
    summary: list[dict[str, object]] = []
    warned_no_images = False
    for variant in variants:
        rotation = variant_rotation(variant, detections, base_size[0], base_size[1], helper_args)
        image_count, image_size = rotate_images_for_sequence(detections, rotation, seq, variant, helper_args, base_size)
        if helper_args.rotate_images and image_count == 0 and detections and not warned_no_images:
            first = min(detections, key=lambda item: item.frame)
            print(
                "[WARN] "
                f"{seq}: no source images were found/read for --rotate-images. "
                f"image_root={helper_args.image_root}, first_frame={first.frame}, "
                f"first_frame_file={first.frame_file}"
            )
            warned_no_images = True
        width, height = image_size
        records = [
            rotate_detection_with_projection(
                det,
                rotation,
                width,
                height,
                args.edge_samples,
                vertical_fov_rad(helper_args),
            )
            for det in detections
        ]
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
                "vertical_fov_deg": float(helper_args.vertical_fov_deg),
                "rotation_matrix": rotation.tolist(),
                "oriented_csv": str(oriented_path),
                "mot_aabb": str(aabb_path),
            }
        )
        print(f"[OK] {seq}/{variant}: labels={oriented_count}, images={image_count}")
    return summary


def input_not_found_message(args: argparse.Namespace, selected_format: str) -> str:
    det_root = args.det_root or (args.quadtrack_root / "detection_results_mot")
    mot_count = len(mot_files(det_root, args.seq_glob)) if det_root.exists() else 0
    seq_candidates = motchallenge_sequence_candidates(args.quadtrack_root)
    seq_with_gt = sum(1 for seq in seq_candidates if (seq / "gt" / "gt.txt").exists())
    seq_with_det = sum(1 for seq in seq_candidates if (seq / "det" / "det.txt").exists())
    json_count = len(json_files(args.quadtrack_root, args.seq_glob))
    return (
        f"No input files found under {args.quadtrack_root}\n"
        f"selected input format: {selected_format}\n"
        f"checked MOT txt: {det_root}/*.txt -> {mot_count} file(s)\n"
        f"checked MOTChallenge sequence dirs: {len(seq_candidates)} candidate(s), "
        f"{seq_with_gt} with gt/gt.txt, {seq_with_det} with det/det.txt\n"
        f"checked QuadTrack JSON: {args.quadtrack_root}/*.json -> {json_count} file(s)\n"
        "If this is an official test split with only img1 images, there are no boxes to rotate; "
        "provide detections in det/det.txt or use a split that has gt/gt.txt."
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    input_format, files = choose_input_files(args)
    if args.limit_seqs:
        files = files[: args.limit_seqs]
    if not files:
        raise SystemExit(input_not_found_message(args, input_format))
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
        "vertical_fov_deg": float(args.vertical_fov_deg),
        "input_kind": args.input_kind,
        "label_source": args.label_source,
        "min_score": args.min_score,
        "mot_frame_to_image_offset": args.mot_frame_to_image_offset,
        "json_frame_offset": args.json_frame_offset,
        "frame_name_width": args.frame_name_width,
        "frame_image_ext": args.frame_image_ext,
        "mot_class_name": args.mot_class_name,
        "summary": all_summary,
    }
    manifest_path = args.out_root / "orientation_benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
