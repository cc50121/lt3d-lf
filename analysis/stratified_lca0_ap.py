#!/usr/bin/env python3
"""Compute per-class, strata-aware LCA0 AP for an existing LT3D result.

The script reuses the nuScenes evaluator's loaded and filtered boxes, its
distance function and its AP calculation.  For a stratum, boxes from other
strata are treated as ignored ground truth: a prediction close to one is not
counted as a false positive.  This prevents valid detections of, for example,
dense objects from depressing sparse-object AP.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Set LT3D_NUSCENES_SDK when the extended LT3D SDK is not installed globally.
LT3D_SDK = os.environ.get('LT3D_NUSCENES_SDK')
if LT3D_SDK and LT3D_SDK not in sys.path:
    sys.path.insert(0, LT3D_SDK)

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.algo import calc_ap
from nuscenes.eval.detection.data_classes import DetectionMetricData
from nuscenes.eval.detection.evaluate import DetectionEval


CLASSES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'emergency_vehicle', 'adult', 'child', 'police_officer',
    'construction_worker', 'stroller', 'personal_mobility',
    'pushable_pullable', 'debris', 'traffic_cone', 'barrier',
]
DIST_THRESHOLDS = ((0.5, 0), (1.0, 0), (2.0, 0), (4.0, 0))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', required=True, type=Path)
    parser.add_argument('--dataroot', default='./data/nuscenes/')
    parser.add_argument('--gt-cache', required=True, type=Path,
                        help='GT records from analyze_distance_sparsity.py')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--config', default='detection_lt3d')
    return parser.parse_args()


def ego_distance(box):
    return float(np.linalg.norm(box.ego_translation[:2]))


def cache_lookup(cache_records):
    by_key = defaultdict(list)
    for record in cache_records:
        by_key[(record['sample_token'], record['class_name'])].append(record)
    return by_key


def lidar_count_for_gt(box, cache_by_key):
    candidates = cache_by_key[(box.sample_token, box.detection_name)]
    point = np.asarray(box.translation[:2], dtype=np.float64)
    best = min(candidates, key=lambda row: np.sum(
        (np.asarray(row['translation'][:2], dtype=np.float64) - point) ** 2))
    error = float(np.linalg.norm(np.asarray(best['translation'][:2]) - point))
    if error > 0.05:
        raise RuntimeError('GT cache mismatch for {} {} ({} m)'.format(
            box.sample_token, box.detection_name, error))
    return int(best['lidar_point_count'])


def accumulate_with_ignored(gt_selected, gt_ignored, pred_boxes, class_name,
                            dist_fcn, dist_th):
    """nuScenes AP accumulation, with excluded-stratum matches ignored."""
    distance_threshold = dist_th[0]
    selected = [box for boxes in gt_selected.values() for box in boxes
                if box.detection_name == class_name]
    npos = len(selected)
    if npos == 0:
        return DetectionMetricData.no_predictions()

    predictions = [box for box in pred_boxes.all
                   if box.detection_name == class_name]
    predictions.sort(key=lambda box: box.detection_score, reverse=True)
    tp, fp, conf = [], [], []
    taken = set()
    for pred in predictions:
        candidates = gt_selected[pred.sample_token]
        best_distance, best_index = np.inf, None
        for index, gt in enumerate(candidates):
            if gt.detection_name != class_name or (pred.sample_token, index) in taken:
                continue
            distance = dist_fcn(gt, pred)
            if distance < best_distance:
                best_distance, best_index = distance, index
        if best_distance < distance_threshold:
            taken.add((pred.sample_token, best_index))
            tp.append(1)
            fp.append(0)
            conf.append(pred.detection_score)
            continue

        # A prediction that localizes an excluded GT is neither a TP nor FP.
        ignored_match = any(
            gt.detection_name == class_name and dist_fcn(gt, pred) < distance_threshold
            for gt in gt_ignored[pred.sample_token])
        if ignored_match:
            continue
        tp.append(0)
        fp.append(1)
        conf.append(pred.detection_score)

    if not tp or not any(tp):
        return DetectionMetricData.no_predictions()
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    recall = tp / float(npos)
    precision = tp / (tp + fp)
    recall_interp = np.linspace(0, 1, DetectionMetricData.nelem)
    precision = np.interp(recall_interp, recall, precision, right=0)
    confidence = np.interp(recall_interp, recall, np.asarray(conf), right=0)
    zeros = np.zeros(DetectionMetricData.nelem)
    return DetectionMetricData(recall=recall_interp, precision=precision,
                               confidence=confidence, trans_err=zeros,
                               vel_err=zeros, scale_err=zeros,
                               orient_err=zeros, attr_err=zeros)


def evaluate_stratum(evaluator, selectors, name, sparse_thresholds=None):
    rows = []
    for class_name in CLASSES:
        selected, ignored = defaultdict(list), defaultdict(list)
        for token in evaluator.sample_tokens:
            for gt in evaluator.gt_boxes[token]:
                if gt.detection_name != class_name:
                    continue
                if selectors[class_name](gt):
                    selected[token].append(gt)
                else:
                    ignored[token].append(gt)
        aps = []
        for threshold in DIST_THRESHOLDS:
            metrics = accumulate_with_ignored(
                selected, ignored, evaluator.pred_boxes, class_name,
                evaluator.cfg.dist_fcn_callable, threshold)
            aps.append(calc_ap(metrics, evaluator.cfg.min_recall,
                               evaluator.cfg.min_precision))
        row = {'stratum': name, 'class_name': class_name,
               'gt_count': sum(len(v) for v in selected.values()),
               'ap_0.5m': aps[0], 'ap_1m': aps[1], 'ap_2m': aps[2],
               'ap_4m': aps[3], 'lca0_ap': float(np.mean(aps))}
        if sparse_thresholds is not None:
            row['lidar_point_median'] = sparse_thresholds[class_name]
        rows.append(row)
    return rows


def write_csv(path, rows):
    fields = ['stratum', 'class_name', 'gt_count', 'lidar_point_median',
              'ap_0.5m', 'ap_1m', 'ap_2m', 'ap_4m', 'lca0_ap']
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)
    evaluator = DetectionEval(nusc, config_factory(args.config), str(args.result),
                              'val', str(args.output_dir / '_evaluator_tmp'),
                              metric_type='hierarchy', verbose=False)
    with args.gt_cache.open() as handle:
        cache = cache_lookup(json.load(handle))

    counts = {}
    for class_name in CLASSES:
        values = []
        for gt in evaluator.gt_boxes.all:
            if gt.detection_name == class_name:
                values.append(lidar_count_for_gt(gt, cache))
        counts[class_name] = float(np.median(values))

    all_rows = []
    all_rows += evaluate_stratum(
        evaluator,
        {name: (lambda box, high=high: ego_distance(box) < high)
         for name, high in [(name, 30.0) for name in CLASSES]},
        '0-30m')
    all_rows += evaluate_stratum(
        evaluator,
        {name: (lambda box, low=low: ego_distance(box) >= low)
         for name, low in [(name, 30.0) for name in CLASSES]},
        '30m+')
    all_rows += evaluate_stratum(
        evaluator,
        {name: (lambda box, high=high: ego_distance(box) < high)
         for name, high in [(name, 50.0) for name in CLASSES]},
        '0-50m')
    all_rows += evaluate_stratum(
        evaluator,
        {name: (lambda box, low=low: ego_distance(box) >= low)
         for name, low in [(name, 50.0) for name in CLASSES]},
        '50m+')
    all_rows += evaluate_stratum(
        evaluator,
        {name: (lambda box, name=name: lidar_count_for_gt(box, cache) <= counts[name])
         for name in CLASSES}, 'sparse (<= class median)', counts)
    all_rows += evaluate_stratum(
        evaluator,
        {name: (lambda box, name=name: lidar_count_for_gt(box, cache) > counts[name])
         for name in CLASSES}, 'dense (> class median)', counts)
    write_csv(args.output_dir / 'per_class_stratified_lca0.csv', all_rows)
    with (args.output_dir / 'class_lidar_point_medians.json').open('w') as handle:
        json.dump(counts, handle, indent=2, sort_keys=True)
    print('Wrote', args.output_dir / 'per_class_stratified_lca0.csv')
    print('Wrote', args.output_dir / 'class_lidar_point_medians.json')


if __name__ == '__main__':
    main()
