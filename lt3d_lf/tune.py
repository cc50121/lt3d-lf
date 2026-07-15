"""Evaluate one uniform GroundingDINO calibration candidate on LCA0."""

import argparse
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import mmcv

from . import cli


CLASSES = (
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'emergency_vehicle', 'adult', 'child', 'police_officer',
    'construction_worker', 'stroller', 'personal_mobility',
    'pushable_pullable', 'debris', 'traffic_cone', 'barrier',
)
LCA0_THRESHOLDS = ((0.5, 0), (1.0, 0), (2.0, 0), (4.0, 0))


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factor', type=float, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--framework-root', type=Path, default=cli.DEFAULT_FRAMEWORK_ROOT)
    parser.add_argument('--nuscenes-sdk', type=Path, default=cli.DEFAULT_NUSCENES_SDK)
    parser.add_argument('--data-root', type=Path, default=cli.DEFAULT_DATA_ROOT)
    parser.add_argument(
        '--config', type=Path,
        default=cli.DEFAULT_FRAMEWORK_ROOT / 'configs/centerpoint/lt3d/'
        'centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_50m_wide_hierarchy_tta_20e_nus.py')
    parser.add_argument('--lidar-predictions', type=Path, default=cli.DEFAULT_LIDAR_PREDICTIONS)
    parser.add_argument('--camera-results', type=Path, required=True)
    parser.add_argument('--keep-json', action='store_true')
    return parser.parse_args()


def runtime_args(args, output_dir):
    return SimpleNamespace(
        framework_root=args.framework_root,
        nuscenes_sdk=args.nuscenes_sdk,
        data_root=args.data_root,
        config=args.config,
        lidar_predictions=args.lidar_predictions,
        camera_results=args.camera_results,
        output_dir=output_dir,
    )


def uniform_calibration(factor):
    return {
        'bay': {
            class_name: {'c': factor, 'p': 0.1}
            for class_name in CLASSES
        }
    }


def main():
    args = parse_args()
    candidate_dir = args.work_dir.expanduser().resolve()
    manifest = candidate_dir / 'candidate_metrics.json'
    if manifest.exists():
        print(f'Already complete: {manifest}')
        return
    candidate_dir.mkdir(parents=True, exist_ok=True)
    cli.configure_runtime(runtime_args(args, candidate_dir))

    from .fusion import make_fine_fusion
    from . import evaluation

    result_json = candidate_dir / 'pts_bbox' / 'results_nusc.json'
    if result_json.exists():
        print(f'Reusing existing fusion JSON: {result_json}')
    else:
        make_fine_fusion(0.5, uniform_calibration(args.factor), 0.3)

    config = evaluation.config_factory(evaluation.eval_version)
    config.dist_ths = [list(value) for value in LCA0_THRESHOLDS]
    evaluator = evaluation.NuScenesEval(
        evaluation.nusc,
        config=config,
        result_path=str(result_json),
        eval_set=evaluation.eval_set_map[evaluation.version],
        output_dir=str(candidate_dir),
        metric_type='hierarchy',
        verbose=False,
    )
    evaluator.main(render_curves=False)
    summary = mmcv.load(candidate_dir / 'metrics_summary.json')
    lca0 = {
        class_name: sum(
            summary['label_aps'][class_name][f'{distance}/0']
            for distance in (0.5, 1.0, 2.0, 4.0)
        ) / 4.0
        for class_name in CLASSES
    }
    manifest.write_text(json.dumps({
        'factor': args.factor,
        'lca0_per_class': lca0,
        'mean_lca0': sum(lca0.values()) / len(lca0),
        'camera_results': str(args.camera_results.resolve()),
        'lidar_predictions': str(args.lidar_predictions.resolve()),
    }, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    if not args.keep_json:
        shutil.rmtree(candidate_dir / 'pts_bbox')
    print(manifest)


if __name__ == '__main__':
    main()
