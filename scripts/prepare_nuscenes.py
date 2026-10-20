#!/usr/bin/env python3
"""Build LT3D infos, projected 2D COCO annotations, and optional DB samples."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--framework-root', type=Path, required=True)
    parser.add_argument('--create-infos', action='store_true')
    parser.add_argument('--create-database', action='store_true')
    args = parser.parse_args()
    root = args.data_root.resolve()
    sys.path.insert(0, str(args.framework_root.resolve()))
    from tools.data_converter.nuscenes_converter import (
        create_nuscenes_infos, export_2d_annotation, nus_categories)
    import mmcv
    if args.create_infos:
        create_nuscenes_infos(str(root), 'nuscenes', str(root),
                              version='v1.0-trainval', max_sweeps=10)
    for split in ('train', 'val'):
        info_path = root / ('nuscenes_infos_' + split + '.pkl')
        export_2d_annotation(str(root), str(info_path), 'v1.0-trainval', mono3d=False)
        path = root / ('nuscenes_infos_' + split + '.coco.json')
        coco = json.loads(path.read_text())
        category_ids = {c['id']: nus_categories.index(c['name']) + 1
                        for c in coco['categories']}
        image_paths = {}
        for info in mmcv.load(str(info_path))['infos']:
            for camera in info['cams'].values():
                p = Path(camera['data_path']).resolve()
                image_paths[camera['sample_data_token']] = str(p)
        image_ids = {}
        for index, image in enumerate(coco['images'], 1):
            image_ids[image['id']] = index
            image['file_name'] = image_paths[image['id']]
            image['id'] = index
        coco['categories'] = [dict(id=i + 1, name=name)
                              for i, name in enumerate(nus_categories)]
        for annotation in coco['annotations']:
            annotation['image_id'] = image_ids[annotation['image_id']]
            annotation['category_id'] = category_ids[annotation['category_id']]
        target = root / ('test_2D_' + split + '.json')
        if target.exists():
            raise FileExistsError('Refusing to overwrite annotations: ' + str(target))
        target.write_text(json.dumps(coco), encoding='utf-8')
        print('Wrote', target)
    if args.create_database:
        from tools.data_converter.create_gt_database import create_groundtruth_database
        create_groundtruth_database('NuScenesDataset', str(root), 'nuscenes',
                                    info_path=str(root / 'nuscenes_infos_train.pkl'))


if __name__ == '__main__':
    main()
