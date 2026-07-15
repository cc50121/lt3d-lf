import os
import numpy as np
from os import path as osp
import mmcv
import pandas as pd

from nuscenes.eval.detection.constants import DETECTION_NAMES


CLASSES = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
           'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility', 
           'pushable_pullable', 'debris', 'traffic_cone', 'barrier']
ErrNameMapping = {
    'trans_err': 'mATE',
    'scale_err': 'mASE',
    'orient_err': 'mAOE',
    'vel_err': 'mAVE',
    'attr_err': 'mAAE'
}
from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.config import config_factory

version = 'v1.0-trainval'
data_root = os.environ.get('LT3D_DATA_ROOT', './data/nuscenes/')
nusc = NuScenes(
    version=version, dataroot=data_root, verbose=False)
eval_set_map = {
    'v1.0-mini': 'mini_val',
    'v1.0-trainval': 'val',
}

def evaluate_single(result_path,
                    out_path,
                    metric_type='standard',
                    eval_cls='all',
                    evaluation_config='detection_lt3d_hierarchy'):
    """Evaluation for a single model in nuScenes protocol.

    Args:
        result_path (str): Path of the result file.
        logger (logging.Logger | str, optional): Logger used for printing
            related information during evaluation. Default: None.
        metric (str, optional): Metric name used for evaluation.
            Default: 'bbox'.
        result_name (str, optional): Result name in the metric prefix.
            Default: 'pts_bbox'.

    Returns:
        dict: Dictionary of evaluation details.
    """
    global DETECTION_NAMES
    global CLASSES
    if eval_cls != 'all':    
        DETECTION_NAMES = [eval_cls]
        CLASSES = [eval_cls]
    nusc_eval = NuScenesEval(
        nusc,
        config=config_factory(evaluation_config),
        result_path=result_path,
        eval_set=eval_set_map[version],
        output_dir=out_path,
        metric_type=metric_type,
        verbose=False)
    nusc_eval.main(render_curves=False)
    
    # record metrics
    metrics = mmcv.load(osp.join(out_path, 'metrics_summary.json'))

    if metric_type == "standard":
        detection_metrics = {"trans_err" : "ATE",
                    "scale_err" : "ASE",
                    "orient_err" : "AOE", 
                    "vel_err" : "AVE",
                    "attr_err" : "AAE"}

        detection_dataFrame = { "CLASS" : [],
                            "mAP" : [],
                            "ATE" : [],
                            "ASE" : [],
                            "AOE" : [],
                            "AVE" : [],
                            "AAE" : [],
                            "NDS" : [],
                        }

        detection_dataFrame["CLASS"] = detection_dataFrame["CLASS"] + DETECTION_NAMES

        for classname in detection_dataFrame["CLASS"]:
            detection_dataFrame["mAP"].append(metrics["mean_dist_aps"][classname])
            
        classMetrics = metrics["label_tp_errors"]
        for metric in detection_metrics.keys():
            for classname in detection_dataFrame["CLASS"]:
                detection_dataFrame[detection_metrics[metric]].append(classMetrics[classname][metric])

        for classname in detection_dataFrame["CLASS"]:
            ap = metrics["mean_dist_aps"][classname]
            tp, tp_cnt = 0, 0
            for metric in detection_metrics.keys():
                val = metrics["label_tp_errors"][classname][metric]
                if not np.isnan(val):
                    tp = tp + (1 - min(val, 1))
                    tp_cnt = tp_cnt + 1

            detection_dataFrame["NDS"].append((1 / (5 + tp_cnt)) * (5 * ap + tp))


    elif metric_type in ('hierarchy', 'lca0'):
        detection_dataFrame = {"CLASS": [], "LCA0": []}
        if metric_type == 'hierarchy':
            detection_dataFrame.update({"LCA1": [], "LCA2": []})

        detection_dataFrame["CLASS"] = detection_dataFrame["CLASS"] + DETECTION_NAMES

        for classname in detection_dataFrame["CLASS"]:
            aps = metrics["label_aps"][classname]

            l0 = [aps["0.5/0"], aps["1.0/0"], aps["2.0/0"], aps["4.0/0"]]
            
            detection_dataFrame["LCA0"].append(np.mean(l0))
            if metric_type == 'hierarchy':
                l1 = [aps["1.0/1"], aps["0.5/1"], aps["2.0/1"], aps["4.0/1"]]
                l2 = [aps["0.5/2"], aps["1.0/2"], aps["2.0/2"], aps["4.0/2"]]
                detection_dataFrame["LCA1"].append(np.mean(l1))
                detection_dataFrame["LCA2"].append(np.mean(l2))
    else:
        assert False, "{} is Invalid".format(metric_type)

    detection_dataFrame = pd.DataFrame.from_dict(detection_dataFrame)
    detection_dataFrame.to_csv(out_path + "/" + "nus_" + metric_type + ".csv", index=False)

    detail = dict()
    metric_prefix = 'pts_bbox_NuScenes'
    for name in CLASSES:
        for k, v in metrics['label_aps'][name].items():
            val = float('{:.4f}'.format(v))
            detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
        for k, v in metrics['label_tp_errors'][name].items():
            val = float('{:.4f}'.format(v))
            detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
        for k, v in metrics['tp_errors'].items():
            val = float('{:.4f}'.format(v))
            detail['{}/{}'.format(metric_prefix,
                                  ErrNameMapping[k])] = val

    detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
    detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
    return detail

def evaluate_fusion(eval_cls='all', json_path=None,
                    evaluation_config='detection_lt3d_hierarchy'):
    output_dir = os.environ.get('LT3D_OUTPUT_DIR', './results')
    if json_path is None:
        json_path = osp.join(output_dir, 'pts_bbox', 'results_nusc.json')
    metric_type = ('lca0' if evaluation_config == 'detection_lt3d'
                   else 'hierarchy')
    return evaluate_single(
        json_path, out_path=output_dir, metric_type=metric_type,
        eval_cls=eval_cls, evaluation_config=evaluation_config)
