#!/usr/bin/env python3
"""Convert nuImages train annotations to GroundingDINO/MMDet COCO format.

The optional merged output rewrites image file names to absolute paths so a
single COCO file can reference both nuScenes and nuImages without copying
images into one directory. Use a training config with data_root='' and
data_prefix=dict(img='') for that merged JSON.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


LT3D_CATEGORIES = [
    {"id": 1, "name": "car", "supercategory": "none"},
    {"id": 2, "name": "truck", "supercategory": "none"},
    {"id": 3, "name": "trailer", "supercategory": "none"},
    {"id": 4, "name": "bus", "supercategory": "none"},
    {"id": 5, "name": "construction_vehicle", "supercategory": "none"},
    {"id": 6, "name": "bicycle", "supercategory": "none"},
    {"id": 7, "name": "motorcycle", "supercategory": "none"},
    {"id": 8, "name": "emergency_vehicle", "supercategory": "none"},
    {"id": 9, "name": "adult", "supercategory": "none"},
    {"id": 10, "name": "child", "supercategory": "none"},
    {"id": 11, "name": "police_officer", "supercategory": "none"},
    {"id": 12, "name": "construction_worker", "supercategory": "none"},
    {"id": 13, "name": "stroller", "supercategory": "none"},
    {"id": 14, "name": "personal_mobility", "supercategory": "none"},
    {"id": 15, "name": "pushable_pullable", "supercategory": "none"},
    {"id": 16, "name": "debris", "supercategory": "none"},
    {"id": 17, "name": "traffic_cone", "supercategory": "none"},
    {"id": 18, "name": "barrier", "supercategory": "none"},
]


NUIMAGES_TO_LT3D = {
    "vehicle.car": 1,
    "vehicle.truck": 2,
    "vehicle.trailer": 3,
    "vehicle.bus.rigid": 4,
    "vehicle.bus.bendy": 4,
    "vehicle.construction": 5,
    "vehicle.bicycle": 6,
    "vehicle.motorcycle": 7,
    "vehicle.emergency.police": 8,
    "vehicle.emergency.ambulance": 8,
    "human.pedestrian.adult": 9,
    "human.pedestrian.child": 10,
    "human.pedestrian.police_officer": 11,
    "human.pedestrian.construction_worker": 12,
    "human.pedestrian.stroller": 13,
    "human.pedestrian.personal_mobility": 14,
    "movable_object.pushable_pullable": 15,
    "movable_object.debris": 16,
    "movable_object.trafficcone": 17,
    "movable_object.barrier": 18,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuimages-root", required=True)
    parser.add_argument("--version", default="v1.0-train")
    parser.add_argument("--out-nuimages-json", required=True)
    parser.add_argument("--nusc-coco")
    parser.add_argument("--nusc-image-dir")
    parser.add_argument(
        "--nusc-samples-root",
        help="Optional nuScenes samples root; used to resolve flattened samples_CAM_* names.",
    )
    parser.add_argument("--merged-json")
    parser.add_argument(
        "--nuimages-file-prefix",
        default="samples/CAM_ALL",
        help="Relative image prefix written in the standalone nuImages JSON.",
    )
    parser.add_argument("--min-area", type=float, default=1.0)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)
    print(f"wrote {path}")


def resolve_image_path(root, sample_filename):
    basename = Path(sample_filename).name
    cam_all = root / "samples" / "CAM_ALL" / basename
    if cam_all.exists():
        return cam_all
    return root / sample_filename


def convert_nuimages(args):
    root = Path(args.nuimages_root).resolve()
    table_dir = root / args.version
    category_by_token = {
        row["token"]: row["name"]
        for row in load_json(table_dir / "category.json")
    }
    sample_data = load_json(table_dir / "sample_data.json")
    sample_data_by_token = {row["token"]: row for row in sample_data}

    image_id_by_sample_data = {}
    images = []
    missing_images = []
    for row in sample_data:
        filename = row["filename"]
        if not filename.startswith("samples/"):
            continue
        basename = Path(filename).name
        disk_path = resolve_image_path(root, filename)
        if not disk_path.exists():
            missing_images.append(str(disk_path))
            continue
        image_id = len(images) + 1
        image_id_by_sample_data[row["token"]] = image_id
        images.append({
            "file_name": f"{args.nuimages_file_prefix.rstrip('/')}/{basename}",
            "id": image_id,
            "width": int(row["width"]),
            "height": int(row["height"]),
        })

    annotations = []
    skipped_by_category = Counter()
    category_counts = Counter()
    invalid_bbox = 0
    ann_by_image = defaultdict(int)
    for ann in load_json(table_dir / "object_ann.json"):
        image_id = image_id_by_sample_data.get(ann["sample_data_token"])
        if image_id is None:
            continue
        category_name = category_by_token[ann["category_token"]]
        category_id = NUIMAGES_TO_LT3D.get(category_name)
        if category_id is None:
            skipped_by_category[category_name] += 1
            continue
        sample = sample_data_by_token[ann["sample_data_token"]]
        width = float(sample["width"])
        height = float(sample["height"])
        x1, y1, x2, y2 = (float(value) for value in ann["bbox"])
        x1 = min(max(x1, 0.0), width)
        x2 = min(max(x2, 0.0), width)
        y1 = min(max(y1, 0.0), height)
        y2 = min(max(y2, 0.0), height)
        box_width = x2 - x1
        box_height = y2 - y1
        area = box_width * box_height
        if box_width <= 0 or box_height <= 0 or area < args.min_area:
            invalid_bbox += 1
            continue
        annotations.append({
            "id": len(annotations) + 1,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x1, y1, box_width, box_height],
            "area": area,
            "iscrowd": 0,
            "segmentation": [],
        })
        category_counts[category_id] += 1
        ann_by_image[image_id] += 1

    dataset = {
        "images": images,
        "annotations": annotations,
        "categories": LT3D_CATEGORIES,
    }
    report = {
        "nuimages_root": str(root),
        "version": args.version,
        "images": len(images),
        "annotations": len(annotations),
        "images_with_annotations": len(ann_by_image),
        "missing_images": len(missing_images),
        "invalid_bbox": invalid_bbox,
        "category_counts": {str(k): v for k, v in sorted(category_counts.items())},
        "skipped_by_category": dict(skipped_by_category),
    }
    return dataset, report


def make_abs(path):
    return str(Path(path).resolve())


NUSC_CAMERAS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_FRONT",
    "CAM_BACK",
]


def resolve_nusc_image_path(file_name, nusc_image_dir=None, nusc_samples_root=None):
    raw_path = Path(file_name)
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path

    if nusc_image_dir:
        candidate = Path(nusc_image_dir).resolve() / raw_path.name
        if candidate.exists():
            return candidate

    if nusc_samples_root:
        samples_root = Path(nusc_samples_root).resolve()
        parts = raw_path.as_posix().split("/")
        if len(parts) >= 3 and parts[-3] == "samples":
            return samples_root / parts[-2] / parts[-1]

        name = raw_path.name
        if name.startswith("samples_"):
            rest = name[len("samples_"):]
            for camera in NUSC_CAMERAS:
                prefix = f"{camera}_"
                if rest.startswith(prefix):
                    return samples_root / camera / rest[len(prefix):]

    if nusc_image_dir:
        return Path(nusc_image_dir).resolve() / raw_path.name
    return raw_path


def build_merged_dataset(
    nusc_coco, nusc_image_dir, nusc_samples_root, nuimages_dataset, nuimages_root
):
    nusc = load_json(nusc_coco)
    nusc_image_dir = Path(nusc_image_dir).resolve() if nusc_image_dir else None
    nusc_samples_root = Path(nusc_samples_root).resolve() if nusc_samples_root else None
    nuimages_root = Path(nuimages_root).resolve()

    merged = {"images": [], "annotations": [], "categories": nusc["categories"]}
    image_id_map = {}
    missing_nusc_images = 0
    for image in nusc["images"]:
        new_id = len(merged["images"]) + 1
        image_id_map[("nusc", int(image["id"]))] = new_id
        image_path = resolve_nusc_image_path(
            image["file_name"], nusc_image_dir, nusc_samples_root)
        if not image_path.exists():
            missing_nusc_images += 1
        new_image = dict(image)
        new_image["id"] = new_id
        new_image["file_name"] = make_abs(image_path)
        merged["images"].append(new_image)

    for image in nuimages_dataset["images"]:
        new_id = len(merged["images"]) + 1
        image_id_map[("nuimages", int(image["id"]))] = new_id
        new_image = dict(image)
        new_image["id"] = new_id
        new_image["file_name"] = make_abs(nuimages_root / image["file_name"])
        merged["images"].append(new_image)

    for ann in nusc["annotations"]:
        new_ann = dict(ann)
        new_ann["id"] = len(merged["annotations"]) + 1
        new_ann["image_id"] = image_id_map[("nusc", int(ann["image_id"]))]
        merged["annotations"].append(new_ann)

    for ann in nuimages_dataset["annotations"]:
        new_ann = dict(ann)
        new_ann["id"] = len(merged["annotations"]) + 1
        new_ann["image_id"] = image_id_map[("nuimages", int(ann["image_id"]))]
        merged["annotations"].append(new_ann)

    report = {
        "nusc_images": len(nusc["images"]),
        "nusc_annotations": len(nusc["annotations"]),
        "missing_nusc_images": missing_nusc_images,
        "nuimages_images": len(nuimages_dataset["images"]),
        "nuimages_annotations": len(nuimages_dataset["annotations"]),
        "merged_images": len(merged["images"]),
        "merged_annotations": len(merged["annotations"]),
    }
    return merged, report


def main():
    args = parse_args()
    nuimages_dataset, report = convert_nuimages(args)
    dump_json(nuimages_dataset, args.out_nuimages_json)
    dump_json(report, str(Path(args.out_nuimages_json).with_suffix(".report.json")))

    if args.merged_json:
        if not args.nusc_coco or not (args.nusc_image_dir or args.nusc_samples_root):
            raise ValueError(
                "--merged-json requires --nusc-coco and either "
                "--nusc-image-dir or --nusc-samples-root")
        merged, merge_report = build_merged_dataset(
            args.nusc_coco,
            args.nusc_image_dir,
            args.nusc_samples_root,
            nuimages_dataset,
            args.nuimages_root,
        )
        dump_json(merged, args.merged_json)
        dump_json(merge_report, str(Path(args.merged_json).with_suffix(".report.json")))
        report["merge_report"] = merge_report

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
