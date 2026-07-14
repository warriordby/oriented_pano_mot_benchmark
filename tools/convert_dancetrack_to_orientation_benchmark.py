from __future__ import annotations

import argparse
import configparser
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.convert_quadtrack_to_orientation_benchmark import (
    ORIENTED_HEADER,
    Detection,
    mot_aabb_row,
    oriented_row,
    rotate_detection,
    rotate_images_for_sequence,
    variant_rotation,
    write_csv,
    write_mot_txt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a DanceTrack/MOTChallenge-format dataset to a spherical-rotation oriented benchmark."
    )
    parser.add_argument("--dancetrack-root", type=Path, required=True)
    parser.add_argument("--split", default="val", help="Dataset split directory, e.g. train, val, test.")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--label-source", choices=["gt", "det"], default="gt")
    parser.add_argument(
        "--variants",
        default="prior_a2b,polar_up,target_north_80",
        help=(
            "Comma-separated variants. Built-ins: prior_a2b, prior_b2a, "
            "polar_up, polar_down, target_north_80, target_south_80, custom."
        ),
    )
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--roll-deg", type=float, default=0.0)
    parser.add_argument("--edge-samples", type=int, default=32)
    parser.add_argument("--min-score", type=float, default=-1.0)
    parser.add_argument("--class-name", default="person")
    parser.add_argument("--image-width", type=int, default=0, help="Override seqinfo width.")
    parser.add_argument("--image-height", type=int, default=0, help="Override seqinfo height.")
    parser.add_argument("--rotate-images", action="store_true")
    parser.add_argument("--image-ext", default=None, help="Override output image extension.")
    parser.add_argument("--limit-seqs", type=int, default=0)
    parser.add_argument("--limit-frames", type=int, default=0)
    return parser.parse_args()


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


def split_root(root: Path, split: str) -> Path:
    candidate = root / split
    return candidate if candidate.exists() else root


def sequence_dirs(root: Path, split: str) -> list[Path]:
    base = split_root(root, split)
    seqs = [p for p in sorted(base.iterdir()) if p.is_dir() and (p / "seqinfo.ini").exists()]
    if seqs:
        return seqs
    return [
        p
        for p in sorted(base.iterdir())
        if p.is_dir() and ((p / "gt" / "gt.txt").exists() or (p / "det" / "det.txt").exists())
    ]


def frame_file_name(frame: int, seqinfo: dict[str, str]) -> str:
    name_length = int(float(seqinfo.get("nameLength", "8") or 8))
    im_ext = seqinfo.get("imExt", ".jpg") or ".jpg"
    return f"{frame:0{name_length}d}{im_ext}"


def label_file(seq_dir: Path, source: str) -> Path:
    return seq_dir / source / f"{source}.txt"


def load_dancetrack_labels(seq_dir: Path, source: str, seqinfo: dict[str, str], class_name: str) -> list[Detection]:
    path = label_file(seq_dir, source)
    if not path.exists():
        return []
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
            cls = class_name
            if len(parts) > 7 and parts[7] not in {"", "-1"} and class_name == "class_id":
                cls = f"class_{int(float(parts[7]))}"
            rows.append(
                Detection(
                    frame=frame,
                    object_id=object_id,
                    class_name=cls,
                    xyxy=(x, y, x + w, y + h),
                    score=score,
                    frame_file=frame_file_name(frame, seqinfo),
                )
            )
    return rows


def patch_image_args(args: argparse.Namespace, seq_dir: Path, seqinfo: dict[str, str], width: int, height: int) -> argparse.Namespace:
    args.image_root = seq_dir.parent
    args.image_width = width
    args.image_height = height
    if args.image_ext is None:
        args.image_ext = seqinfo.get("imExt", ".jpg") or ".jpg"
    return args


def process_sequence(seq_dir: Path, args: argparse.Namespace, variants: list[str]) -> list[dict[str, object]]:
    seqinfo = read_seqinfo(seq_dir)
    width = args.image_width or int(float(seqinfo.get("imWidth", "0") or 0))
    height = args.image_height or int(float(seqinfo.get("imHeight", "0") or 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"Missing image size for {seq_dir}; pass --image-width and --image-height.")
    detections = [
        d for d in load_dancetrack_labels(seq_dir, args.label_source, seqinfo, args.class_name) if d.score >= args.min_score
    ]
    if args.limit_frames:
        detections = [d for d in detections if d.frame <= args.limit_frames]
    seq = seq_dir.name
    helper_args = patch_image_args(args, seq_dir, seqinfo, width, height)
    summary: list[dict[str, object]] = []
    for variant in variants:
        rotation = variant_rotation(variant, detections, width, height, helper_args)
        image_count, image_size = rotate_images_for_sequence(detections, rotation, seq, variant, helper_args, (width, height))
        width2, height2 = image_size
        records = [rotate_detection(det, rotation, width2, height2, args.edge_samples) for det in detections]
        oriented_path = args.out_root / args.label_source / "oriented_csv" / variant / f"{seq}.csv"
        aabb_path = args.out_root / args.label_source / "mot_aabb" / variant / f"{seq}.txt"
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
                "image_width": width2,
                "image_height": height2,
                "rotation_matrix": rotation.tolist(),
                "oriented_csv": str(oriented_path),
                "mot_aabb": str(aabb_path),
                "seqinfo": seqinfo,
            }
        )
        print(f"[OK] {seq}/{variant}: labels={oriented_count}, images={image_count}")
    return summary


def main() -> None:
    args = parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seqs = sequence_dirs(args.dancetrack_root, args.split)
    if args.limit_seqs:
        seqs = seqs[: args.limit_seqs]
    if not seqs:
        raise SystemExit(f"No DanceTrack sequences found under {args.dancetrack_root}/{args.split}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    all_summary: list[dict[str, object]] = []
    for seq_dir in seqs:
        all_summary.extend(process_sequence(seq_dir, args, variants))
    manifest = {
        "source": "DanceTrack conversion with PriOr-Flow-style spherical rotation",
        "dancetrack_root": str(args.dancetrack_root),
        "split": args.split,
        "label_source": args.label_source,
        "variants": variants,
        "edge_samples": args.edge_samples,
        "summary": all_summary,
    }
    manifest_path = args.out_root / "orientation_benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
