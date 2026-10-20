#!/usr/bin/env python3
"""Convert COCO detector results to the exact per-camera MMLF TXT interface."""

import argparse
from collections import OrderedDict
import json
import math
from pathlib import Path
import pickle


CLASSES = (
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'emergency_vehicle', 'adult', 'child', 'police_officer',
    'construction_worker', 'stroller', 'personal_mobility',
    'pushable_pullable', 'debris', 'traffic_cone', 'barrier')
VIEWS = ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
         'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT')


def mappings(coco, infos):
    categories = {c['id']: CLASSES.index(c['name']) for c in coco['categories']}
    names = [c['name'] for c in coco['categories']]
    if len(names) != len(CLASSES) or set(names) != set(CLASSES):
        raise ValueError('Expected the complete, unique LT3D 18-class taxonomy')
    if len(categories) != len(CLASSES):
        raise ValueError('Duplicate category IDs')
    files = {}
    for info in infos:
        token = info['token']
        if not token or not all(c.isalnum() or c in '_-' for c in token):
            raise ValueError('Invalid sample token')
        for view in VIEWS:
            basename = Path(info['cams'][view]['data_path']).name
            target = '{}@{}.txt'.format(token, view)
            if basename in files and files[basename] != target:
                raise ValueError('Ambiguous image basename: ' + basename)
            files[basename] = target
    images = {}
    for image in coco['images']:
        images[image['id']] = files[Path(image['file_name']).name]
    if len(images) != len(coco['images']):
        raise ValueError('Duplicate image IDs')
    if len(images) != len(files) or len(set(images.values())) != len(files):
        raise ValueError('COCO annotations must cover every validation camera image')
    return categories, images


def export(coco, infos, detections, output, score_threshold=0.15):
    if not 0 <= score_threshold <= 1:
        raise ValueError('Score threshold must be in [0, 1]')
    categories, images = mappings(coco, infos)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output directory must be empty: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    for target in images.values():
        (output / target).touch()
    handles = OrderedDict()
    count = 0
    try:
        for detection in detections:
            target = images[detection['image_id']]
            class_id = categories[detection['category_id']]
            score = float(detection['score'])
            x, y, w, h = map(float, detection['bbox'])
            if not all(math.isfinite(v) for v in (score, x, y, w, h)):
                raise ValueError('Non-finite detection')
            if not 0 <= score <= 1 or w <= 0 or h <= 0:
                raise ValueError('Invalid score or box dimensions')
            if score < score_threshold:
                continue
            if target not in handles:
                if len(handles) == 64:
                    handles.popitem(last=False)[1].close()
                handles[target] = (output / target).open('a', encoding='utf-8')
            handles.move_to_end(target)
            handles[target].write('{} {} {} {} {} {}\n'.format(
                class_id, x, y, x + w, y + h, score))
            count += 1
    finally:
        for handle in handles.values():
            handle.close()
    return {'camera_files': len(images), 'detections': count,
            'score_threshold': score_threshold}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coco-annotations', type=Path, required=True)
    parser.add_argument('--bbox-results', type=Path, required=True)
    parser.add_argument('--infos', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--score-threshold', type=float, default=0.15)
    args = parser.parse_args()
    import ijson
    coco = json.loads(args.coco_annotations.read_text(encoding='utf-8'))
    # Pickles can execute code; use only locally generated or trusted release inputs.
    with args.infos.open('rb') as handle:
        infos = pickle.load(handle)['infos']
    with args.bbox_results.open('rb') as handle:
        report = export(coco, infos, ijson.items(handle, 'item'),
                        args.output_dir, args.score_threshold)
    (args.output_dir / 'conversion_report.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
