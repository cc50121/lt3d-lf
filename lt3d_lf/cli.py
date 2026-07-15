"""Command-line entry point for reproducible LT3D late-fusion runs."""

import argparse
import os
import sys
from pathlib import Path

from .calibration import calibration_from_json, centerpoint_dino_calibration


DEFAULT_FRAMEWORK_ROOT = Path(
    os.environ.get('LT3D_FRAMEWORK_ROOT', '/path/to/mmdet3d-lt3d'))
DEFAULT_NUSCENES_SDK = Path(
    os.environ.get('LT3D_NUSCENES_SDK', '/path/to/nuscenes-lt3d/python-sdk'))
DEFAULT_DATA_ROOT = Path(os.environ.get('LT3D_DATA_ROOT', '/path/to/nuscenes'))
DEFAULT_LIDAR_PREDICTIONS = (
    Path(os.environ['LT3D_LIDAR_PREDICTIONS'])
    if 'LT3D_LIDAR_PREDICTIONS' in os.environ
    else DEFAULT_FRAMEWORK_ROOT / 'tools/fusion_open/lidar_results/'
    'prediction_filter_by_dis.pkl')
DEFAULT_CAMERA_RESULTS = (
    Path(os.environ['LT3D_CAMERA_RESULTS'])
    if 'LT3D_CAMERA_RESULTS' in os.environ
    else DEFAULT_FRAMEWORK_ROOT / 'tools/fusion_open/camera_results/Co-DINO/'
    'nuscenes_nuimages')

DEFAULT_RARE_SAFE_CLASSES = (
    'child,stroller,personal_mobility,emergency_vehicle,police_officer,'
    'debris,construction_worker,bicycle,motorcycle,pushable_pullable,'
    'construction_vehicle,trailer')
DEFAULT_RESCUE_CLASSES = DEFAULT_RARE_SAFE_CLASSES

def parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    cli = argparse.ArgumentParser(
        description='LiDAR + camera late fusion on nuScenes-LT3D.')
    cli.add_argument('command', choices=('fuse', 'evaluate'), nargs='?', default='fuse')
    cli.add_argument('--framework-root', type=Path, default=DEFAULT_FRAMEWORK_ROOT)
    cli.add_argument('--nuscenes-sdk', type=Path, default=DEFAULT_NUSCENES_SDK)
    cli.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    cli.add_argument(
        '--config', type=Path,
        default=DEFAULT_FRAMEWORK_ROOT / 'configs/centerpoint/lt3d/'
        'centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_50m_wide_hierarchy_tta_20e_nus.py')
    cli.add_argument(
        '--lidar-predictions', type=Path,
        default=DEFAULT_LIDAR_PREDICTIONS)
    cli.add_argument(
        '--camera-results', type=Path,
        default=DEFAULT_CAMERA_RESULTS)
    cli.add_argument('--output-dir', type=Path, default=project_root / 'results')
    cli.add_argument('--result-json', type=Path)
    cli.add_argument('--calibration-json', type=Path,
                     help='Per-class calibration manifest produced by the tuner.')
    cli.add_argument(
        '--evaluation-config', default='detection_lt3d_hierarchy',
        choices=('detection_lt3d_hierarchy', 'detection_lt3d'),
        help=('nuScenes-LT3D evaluation config. Use detection_lt3d for '
              'LCA0-only evaluation.'))
    cli.add_argument('--iou-threshold', type=float, default=0.5)
    cli.add_argument('--unmatched-lidar-factor', type=float, default=0.3)
    cli.add_argument(
        '--score-3d-threshold', type=float, default=0.1,
        help='Base LiDAR candidate threshold. Legacy default is 0.1.')
    cli.add_argument(
        '--score-3d-threshold-json',
        help=('Path or inline JSON mapping class name to base LiDAR '
              'candidate threshold. Overrides --score-3d-threshold for listed classes.'))
    cli.add_argument(
        '--score-2d-threshold', type=float,
        help='2D detector score threshold before fusion. Default keeps legacy 0.2.')
    cli.add_argument(
        '--match-mode', default='iou',
        choices=('iou', 'class_iou', 'class_relaxed', 'visible_amodal'),
        help=('2D/3D matching rule. iou is the legacy class-agnostic IoU '
              'matcher; class_iou gates by fine class; class_relaxed uses '
              'fine-class gate plus IoU/IoA/center-inside matching; '
              'visible_amodal also allows projected-box coverage and projected '
              'center-inside-detector matching.'))
    cli.add_argument('--relaxed-iou-threshold', type=float, default=0.3)
    cli.add_argument('--ioa-det-threshold', type=float, default=0.7)
    cli.add_argument('--ioa-proj-threshold', type=float, default=1.1)
    center_group = cli.add_mutually_exclusive_group()
    center_group.add_argument('--center-inside', dest='center_inside',
                              action='store_true',
                              help='Enable 2D-center-inside-3D-projection matching.')
    center_group.add_argument('--no-center-inside', dest='center_inside',
                              action='store_false',
                              help='Disable center-inside relaxed matching.')
    cli.set_defaults(center_inside=None)
    cli.add_argument('--center-inside-score', type=float, default=0.51)
    proj_center_group = cli.add_mutually_exclusive_group()
    proj_center_group.add_argument(
        '--proj-center-inside', dest='proj_center_inside',
        action='store_true',
        help='Enable projected-3D-box-center-inside-2D-detector matching.')
    proj_center_group.add_argument(
        '--no-proj-center-inside', dest='proj_center_inside',
        action='store_false',
        help='Disable projected-center-inside matching.')
    cli.set_defaults(proj_center_inside=None)
    cli.add_argument('--proj-center-inside-score', type=float, default=0.51)
    cli.add_argument(
        '--mismatch-policy', default='relabel',
        choices=('relabel', 'keep_3d', 'ignore'),
        help=('Policy for legacy class-agnostic matches whose 2D and 3D classes '
              'differ. relabel keeps the old behavior.'))
    cli.add_argument(
        '--adult-child-relaxed', action='store_true',
        help=('Allow 3D adult proposals to match 2D child detections under the '
              'relaxed visible/amodal matcher. This is off by default.'))
    cli.add_argument(
        '--adult-child-bidirectional', action='store_true',
        help='Also allow 3D child proposals to match 2D adult detections.')
    cli.add_argument(
        '--adult-child-min-2d-score', type=float, default=0.0,
        help='Minimum 2D score required for adult/child relaxed relabeling.')
    cli.add_argument(
        '--adult-child-min-match-score', type=float, default=0.0,
        help='Minimum 2D/3D match quality for adult/child relaxed relabeling.')
    cli.add_argument(
        '--rare-safe-unmatched', action='store_true',
        help=('For selected long-tail classes, do not apply the global unmatched '
              'LiDAR score penalty.'))
    cli.add_argument(
        '--rare-safe-classes', default=DEFAULT_RARE_SAFE_CLASSES,
        help='Comma-separated classes protected by --rare-safe-unmatched.')
    cli.add_argument('--rare-safe-factor', type=float, default=1.0)
    cli.add_argument(
        '--unmatched-factor-json',
        help=('Path to a JSON dict or inline JSON mapping class name to unmatched '
              'LiDAR factor. Overrides --rare-safe-unmatched for listed classes.'))
    cli.add_argument(
        '--rescue-classes', default='',
        help=('Comma-separated classes allowed to enter the low-score LiDAR '
              'rescue pool. Empty disables proposal rescue.'))
    cli.add_argument(
        '--default-rescue-classes', action='store_true',
        help='Use the default long-tail class list for --rescue-classes.')
    cli.add_argument(
        '--rescue-score-threshold', type=float, default=0.05,
        help='Minimum LiDAR score for rescue-pool candidates.')
    cli.add_argument(
        '--rescue-score-json',
        help='Path or inline JSON mapping class name to rescue score threshold.')
    cli.add_argument(
        '--rescue-min-2d-score', type=float, default=0.0,
        help='Extra 2D score requirement before a low-score LiDAR candidate is rescued.')
    cli.add_argument(
        '--rescue-min-match-score', type=float, default=0.0,
        help='Extra match-quality requirement before a low-score LiDAR candidate is rescued.')
    cli.add_argument(
        '--rescue-unmatched-factor', type=float, default=0.0,
        help='Score factor for low-score rescue candidates that do not get 2D support.')
    cli.add_argument(
        '--skip-evaluate', action='store_true',
        help='Only run fusion and skip nuScenes-LT3D evaluation.')
    return cli


def _require(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f'{description} does not exist: {resolved}')
    return resolved


def configure_runtime(args: argparse.Namespace) -> None:
    framework_root = _require(args.framework_root, 'Framework root')
    sdk_root = _require(args.nuscenes_sdk, 'nuScenes-LT3D SDK')
    data_root = _require(args.data_root, 'nuScenes data root')
    config = _require(args.config, 'CenterPoint config')
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (str(sdk_root), str(framework_root)):
        if path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, str(framework_root))
    sys.path.insert(0, str(sdk_root))

    import nuscenes
    imported_from = Path(nuscenes.__file__).resolve()
    if not str(imported_from).startswith(str(sdk_root)):
        raise RuntimeError(
            f'Expected nuScenes-LT3D SDK at {sdk_root}, but imported {imported_from}')

    os.environ.update({
        'LT3D_CONFIG_FILE': str(config),
        'LT3D_DATA_ROOT': str(data_root),
        'LT3D_INFO_PATH': str(data_root / 'nuscenes_infos_val.pkl'),
        'LT3D_LIDAR_PREDICTIONS': str(args.lidar_predictions.expanduser().resolve()),
        'LT3D_2D_RESULTS_DIR': str(args.camera_results.expanduser().resolve()),
        'LT3D_OUTPUT_DIR': str(output_dir),
        'LT3D_SCORE_3D_THRESH': str(args.score_3d_threshold),
        'LT3D_MATCH_MODE': args.match_mode,
        'LT3D_RELAXED_IOU_THRESH': str(args.relaxed_iou_threshold),
        'LT3D_IOA_DET_THRESH': str(args.ioa_det_threshold),
        'LT3D_IOA_PROJ_THRESH': str(args.ioa_proj_threshold),
        'LT3D_CENTER_INSIDE_SCORE': str(args.center_inside_score),
        'LT3D_PROJ_CENTER_INSIDE_SCORE': str(args.proj_center_inside_score),
        'LT3D_MISMATCH_POLICY': args.mismatch_policy,
        'LT3D_ADULT_CHILD_RELAXED': '1' if args.adult_child_relaxed else '0',
        'LT3D_ADULT_CHILD_BIDIRECTIONAL': (
            '1' if args.adult_child_bidirectional else '0'),
        'LT3D_ADULT_CHILD_MIN_2D_SCORE': str(args.adult_child_min_2d_score),
        'LT3D_ADULT_CHILD_MIN_MATCH_SCORE': (
            str(args.adult_child_min_match_score)),
        'LT3D_RESCUE_CLASSES': (
            DEFAULT_RESCUE_CLASSES if args.default_rescue_classes
            else args.rescue_classes),
        'LT3D_RESCUE_SCORE_THRESH': str(args.rescue_score_threshold),
        'LT3D_RESCUE_MIN_2D_SCORE': str(args.rescue_min_2d_score),
        'LT3D_RESCUE_MIN_MATCH_SCORE': str(args.rescue_min_match_score),
        'LT3D_RESCUE_UNMATCHED_FACTOR': str(args.rescue_unmatched_factor),
    })
    if args.score_3d_threshold_json:
        os.environ['LT3D_SCORE_3D_THRESH_JSON'] = args.score_3d_threshold_json
    if args.score_2d_threshold is not None:
        os.environ['LT3D_SCORE_2D_THRESH'] = str(args.score_2d_threshold)
    if args.center_inside is not None:
        os.environ['LT3D_USE_CENTER_INSIDE'] = '1' if args.center_inside else '0'
    if args.proj_center_inside is not None:
        os.environ['LT3D_USE_PROJ_CENTER_INSIDE'] = (
            '1' if args.proj_center_inside else '0')
    if args.rare_safe_unmatched:
        os.environ['LT3D_RARE_SAFE_UNMATCHED_CLASSES'] = args.rare_safe_classes
        os.environ['LT3D_RARE_SAFE_UNMATCHED_FACTOR'] = str(args.rare_safe_factor)
    if args.unmatched_factor_json:
        os.environ['LT3D_UNMATCHED_FACTOR_JSON'] = args.unmatched_factor_json
    if args.rescue_score_json:
        os.environ['LT3D_RESCUE_SCORE_JSON'] = args.rescue_score_json
    print(f'Using nuScenes-LT3D SDK: {imported_from}')


def run(args: argparse.Namespace) -> None:
    configure_runtime(args)
    from . import evaluation, metrics

    output_dir = Path(os.environ['LT3D_OUTPUT_DIR'])
    result_json = args.result_json
    if result_json is None:
        result_json = output_dir / 'pts_bbox/results_nusc.json'

    if args.command == 'fuse':
        _require(Path(os.environ['LT3D_LIDAR_PREDICTIONS']), 'LiDAR predictions')
        _require(Path(os.environ['LT3D_2D_RESULTS_DIR']), '2D detector results')
        from .fusion import make_fine_fusion
        calibration = (calibration_from_json(args.calibration_json)
                       if args.calibration_json else centerpoint_dino_calibration())
        make_fine_fusion(
            args.iou_threshold,
            calibration,
            args.unmatched_lidar_factor,
        )
        if args.skip_evaluate:
            return

    _require(result_json, 'Fusion result JSON')
    evaluation.evaluate_fusion(
        json_path=str(result_json), evaluation_config=args.evaluation_config)
    metrics.lca0()


def main() -> None:
    run(parser().parse_args())


if __name__ == '__main__':
    main()
