import os
import json
import pickle
import numpy as np
import cv2
import lap
from tqdm import tqdm 
import mmcv
from mmcv import Config
try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

from .utils import *

font = cv2.FONT_HERSHEY_SIMPLEX

# Read cfg file
config_file = os.environ.get('LT3D_CONFIG_FILE', 'configs/centerpoint/lt3d/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_50m_wide_hierarchy_tta_20e_nus.py')
cfg_lidar_3d = Config.fromfile(config_file)
cfg_lidar_3d = compat_cfg(cfg_lidar_3d); cfg_lidar_3d.data_root = os.environ.get('LT3D_DATA_ROOT', cfg_lidar_3d.data_root)

# Some flags
write_res = int(os.environ.get('LT3D_WRITE_RES', '1'))
write_format = 'json'
# write_format = 'pkl'
draw_gt = 0
get_gt = 0
get_2d = 1
show = 0

'''********* Forward off line *********'''
classes = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
        'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility', 
        'pushable_pullable', 'debris', 'traffic_cone', 'barrier']
classes_coarse = ["vehicle", "vehicle", "vehicle", "vehicle", "vehicle", "vehicle", "vehicle", "vehicle", 
                "pedestrian", "pedestrian", "pedestrian", "pedestrian", "pedestrian", "pedestrian", 
                "movable", "movable", "movable", "movable"]
views = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
draw_boxes_indexes_img_view = [(0, 1), (1, 2), (2, 3), (3, 0),
                                (4, 5), (5, 6), (6, 7), (7, 4),
                                (0, 4), (1, 5), (2, 6), (3, 7)]            
color_map = {0: (255, 255, 0),
            1: (0, 255, 255)}
scale_factor = 4
score_3d_thresh = float(os.environ.get('LT3D_SCORE_3D_THRESH', '0.1'))
score_2d_thresh = float(os.environ.get('LT3D_SCORE_2D_THRESH', '0.2'))
DEFAULT_IMAGE_HEIGHT = int(os.environ.get('LT3D_IMAGE_HEIGHT', '900'))
DEFAULT_IMAGE_WIDTH = int(os.environ.get('LT3D_IMAGE_WIDTH', '1600'))
MATCH_MODE = os.environ.get('LT3D_MATCH_MODE', 'iou')
RELAXED_IOU_THRESH = float(os.environ.get('LT3D_RELAXED_IOU_THRESH', '0.3'))
IOA_DET_THRESH = float(os.environ.get('LT3D_IOA_DET_THRESH', '0.7'))
IOA_PROJ_THRESH = float(os.environ.get('LT3D_IOA_PROJ_THRESH', '1.1'))
USE_CENTER_INSIDE = int(os.environ.get('LT3D_USE_CENTER_INSIDE', '1')) == 1
CENTER_INSIDE_SCORE = float(os.environ.get('LT3D_CENTER_INSIDE_SCORE', '0.51'))
USE_PROJ_CENTER_INSIDE = int(os.environ.get('LT3D_USE_PROJ_CENTER_INSIDE', '0')) == 1
PROJ_CENTER_INSIDE_SCORE = float(
    os.environ.get('LT3D_PROJ_CENTER_INSIDE_SCORE', '0.51'))
MATCH_MIN_SCORE = float(os.environ.get('LT3D_MATCH_MIN_SCORE', '1e-6'))
MISMATCH_POLICY = os.environ.get('LT3D_MISMATCH_POLICY', 'relabel')
RARE_SAFE_UNMATCHED_CLASSES = {
    item.strip()
    for item in os.environ.get('LT3D_RARE_SAFE_UNMATCHED_CLASSES', '').split(',')
    if item.strip()
}
RARE_SAFE_UNMATCHED_FACTOR = float(
    os.environ.get('LT3D_RARE_SAFE_UNMATCHED_FACTOR', '1.0'))
RESCUE_CLASSES = {
    item.strip()
    for item in os.environ.get('LT3D_RESCUE_CLASSES', '').split(',')
    if item.strip()
}
RESCUE_SCORE_THRESH = float(
    os.environ.get('LT3D_RESCUE_SCORE_THRESH', str(score_3d_thresh)))
RESCUE_MIN_2D_SCORE = float(os.environ.get('LT3D_RESCUE_MIN_2D_SCORE', '0.0'))
RESCUE_MIN_MATCH_SCORE = float(
    os.environ.get('LT3D_RESCUE_MIN_MATCH_SCORE', '0.0'))
RESCUE_UNMATCHED_FACTOR = float(
    os.environ.get('LT3D_RESCUE_UNMATCHED_FACTOR', '0.0'))
ADULT_CHILD_RELAXED = int(os.environ.get('LT3D_ADULT_CHILD_RELAXED', '0')) == 1
ADULT_CHILD_BIDIRECTIONAL = (
    int(os.environ.get('LT3D_ADULT_CHILD_BIDIRECTIONAL', '0')) == 1)
ADULT_CHILD_MIN_2D_SCORE = float(
    os.environ.get('LT3D_ADULT_CHILD_MIN_2D_SCORE', '0.0'))
ADULT_CHILD_MIN_MATCH_SCORE = float(
    os.environ.get('LT3D_ADULT_CHILD_MIN_MATCH_SCORE', '0.0'))

'''Load val info(image, img_metas ...)'''
info_path = os.environ.get(
    'LT3D_INFO_PATH',
    os.path.join(cfg_lidar_3d.data_root, 'nuscenes_infos_val.pkl'))
info_data = pickle.load(open(info_path, 'rb'))
data = mmcv.load(info_path, file_format='pkl')
info_data = list(sorted(data['infos'], key=lambda e: e['timestamp']))

'''Load results from lidar detections'''
prediction_path = os.environ['LT3D_LIDAR_PREDICTIONS']
res3d_fusion = load(prediction_path)

'''Load 2D detections from YOLOV7 or DINO'''
res2d_dir = os.environ['LT3D_2D_RESULTS_DIR'].rstrip('/') + '/'

def resolve_image_path(data_path):
    """Resolve historical ./data/nuscenes paths without project symlinks."""
    if os.path.isabs(data_path):
        return data_path
    normalized = data_path.replace('\\', '/')
    marker = 'data/nuscenes/'
    relative = normalized.split(marker, 1)[1] if marker in normalized else normalized.lstrip('./')
    return os.path.join(cfg_lidar_3d.data_root, relative)


def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(valid,
                           np.logical_and(points[:, 0] < width,
                                          points[:, 1] < height))
    return valid

def lidar2img(points_lidar, camrera_info):
    points_lidar_homogeneous = \
        np.concatenate([points_lidar,
                        np.ones((points_lidar.shape[0], 1),
                                dtype=points_lidar.dtype)], axis=1)
    camera2lidar = np.eye(4, dtype=np.float32)
    camera2lidar[:3, :3] = camrera_info['sensor2lidar_rotation']
    camera2lidar[:3, 3] = camrera_info['sensor2lidar_translation']
    lidar2camera = np.linalg.inv(camera2lidar)
    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    points_camera = points_camera_homogeneous[:, :3]   
        
    valid = np.ones((points_camera.shape[0]),dtype=bool)
    valid = np.logical_and(points_camera[:, -1] > 0.5, valid)
    points_camera = points_camera / points_camera[:, 2:3]
    camera2img = camrera_info['cam_intrinsic']
    points_img = points_camera @ camera2img.T
    points_img = points_img[:, :2]
    return points_img, valid

def compute_iou(rec1, rec2):
    rec1 = (rec1[1], rec1[0], rec1[3], rec1[2])
    rec2 = (rec2[1], rec2[0], rec2[3], rec2[2])        
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
    
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
    
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
    return (intersect / (sum_area - intersect)) * 1.0


def compute_iou_matrix(boxes1, boxes2):
    """Vectorized IoU for boxes in [y1, x1, y2, x2] format."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)

    top = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    left = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    bottom = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    right = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_h = np.maximum(bottom - top, 0.0)
    inter_w = np.maximum(right - left, 0.0)
    inter = inter_h * inter_w

    area1 = np.maximum(boxes1[:, 2] - boxes1[:, 0], 0.0) * \
        np.maximum(boxes1[:, 3] - boxes1[:, 1], 0.0)
    area2 = np.maximum(boxes2[:, 2] - boxes2[:, 0], 0.0) * \
        np.maximum(boxes2[:, 3] - boxes2[:, 1], 0.0)
    union = area1[:, None] + area2[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def compute_overlap_matrices(boxes1, boxes2):
    """Return IoU, det-box IoA, and projected-box IoA for [y1, x1, y2, x2]."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        zeros = np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
        return zeros, zeros, zeros

    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)

    top = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    left = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    bottom = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    right = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_h = np.maximum(bottom - top, 0.0)
    inter_w = np.maximum(right - left, 0.0)
    inter = inter_h * inter_w

    area1 = np.maximum(boxes1[:, 2] - boxes1[:, 0], 0.0) * \
        np.maximum(boxes1[:, 3] - boxes1[:, 1], 0.0)
    area2 = np.maximum(boxes2[:, 2] - boxes2[:, 0], 0.0) * \
        np.maximum(boxes2[:, 3] - boxes2[:, 1], 0.0)
    union = area1[:, None] + area2[None, :] - inter

    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    ioa_det = np.divide(
        inter, area2[None, :], out=np.zeros_like(inter), where=area2[None, :] > 0)
    ioa_proj = np.divide(
        inter, area1[:, None], out=np.zeros_like(inter), where=area1[:, None] > 0)
    return iou, ioa_det, ioa_proj


def compute_center_inside_matrix(proj_boxes, det_boxes):
    """Whether the 2D detector box center lies inside the 3D projected box."""
    if len(proj_boxes) == 0 or len(det_boxes) == 0:
        return np.zeros((len(proj_boxes), len(det_boxes)), dtype=bool)
    proj_boxes = np.asarray(proj_boxes, dtype=np.float32)
    det_boxes = np.asarray(det_boxes, dtype=np.float32)
    center_y = 0.5 * (det_boxes[:, 0] + det_boxes[:, 2])
    center_x = 0.5 * (det_boxes[:, 1] + det_boxes[:, 3])
    return (
        (center_y[None, :] >= proj_boxes[:, None, 0]) &
        (center_y[None, :] <= proj_boxes[:, None, 2]) &
        (center_x[None, :] >= proj_boxes[:, None, 1]) &
        (center_x[None, :] <= proj_boxes[:, None, 3])
    )


def compute_proj_center_inside_matrix(proj_boxes, det_boxes):
    """Whether the projected 3D box center lies inside the 2D detector box."""
    if len(proj_boxes) == 0 or len(det_boxes) == 0:
        return np.zeros((len(proj_boxes), len(det_boxes)), dtype=bool)
    proj_boxes = np.asarray(proj_boxes, dtype=np.float32)
    det_boxes = np.asarray(det_boxes, dtype=np.float32)
    center_y = 0.5 * (proj_boxes[:, 0] + proj_boxes[:, 2])
    center_x = 0.5 * (proj_boxes[:, 1] + proj_boxes[:, 3])
    return (
        (center_y[:, None] >= det_boxes[None, :, 0]) &
        (center_y[:, None] <= det_boxes[None, :, 2]) &
        (center_x[:, None] >= det_boxes[None, :, 1]) &
        (center_x[:, None] <= det_boxes[None, :, 3])
    )


def compute_match_matrix(pre_boxes, det_boxes, pre_names, det_names, iou_thresh):
    """Build the LAP score matrix for the selected fusion ablation."""
    iou, ioa_det, ioa_proj = compute_overlap_matrices(pre_boxes, det_boxes)
    if len(pre_boxes) == 0 or len(det_boxes) == 0:
        return iou

    pre_names = np.asarray(pre_names)
    det_names = np.asarray(det_names)
    same_class = pre_names[:, None] == det_names[None, :]
    adult_child_match = (
        (pre_names[:, None] == 'adult') & (det_names[None, :] == 'child'))
    if ADULT_CHILD_RELAXED and ADULT_CHILD_BIDIRECTIONAL:
        adult_child_match |= (
            (pre_names[:, None] == 'child') & (det_names[None, :] == 'adult'))
    compatible_class = same_class | (adult_child_match if ADULT_CHILD_RELAXED else False)

    if MATCH_MODE == 'iou':
        score_mat = iou.copy()
        score_mat[score_mat <= iou_thresh] = 0
        return score_mat

    if MATCH_MODE == 'class_iou':
        score_mat = iou.copy()
        score_mat[~compatible_class] = 0
        score_mat[score_mat <= iou_thresh] = 0
        return score_mat

    if MATCH_MODE == 'class_relaxed':
        center_inside = compute_center_inside_matrix(pre_boxes, det_boxes)
        valid = compatible_class & (
            (iou >= RELAXED_IOU_THRESH) |
            (ioa_det >= IOA_DET_THRESH) |
            (center_inside if USE_CENTER_INSIDE else False)
        )
        center_score = np.where(center_inside, CENTER_INSIDE_SCORE, 0.0)
        score_mat = np.maximum(iou, np.maximum(ioa_det, center_score))
        score_mat[~valid] = 0
        return score_mat

    if MATCH_MODE == 'visible_amodal':
        det_center_inside = compute_center_inside_matrix(pre_boxes, det_boxes)
        proj_center_inside = compute_proj_center_inside_matrix(pre_boxes, det_boxes)
        valid = compatible_class & (
            (iou >= RELAXED_IOU_THRESH) |
            (ioa_det >= IOA_DET_THRESH) |
            (ioa_proj >= IOA_PROJ_THRESH) |
            (det_center_inside if USE_CENTER_INSIDE else False) |
            (proj_center_inside if USE_PROJ_CENTER_INSIDE else False)
        )
        det_center_score = np.where(det_center_inside, CENTER_INSIDE_SCORE, 0.0)
        proj_center_score = np.where(
            proj_center_inside, PROJ_CENTER_INSIDE_SCORE, 0.0)
        score_mat = np.maximum.reduce([
            iou, ioa_det, ioa_proj, det_center_score, proj_center_score])
        score_mat[~valid] = 0
        return score_mat

    raise ValueError(
        'LT3D_MATCH_MODE must be one of: iou, class_iou, class_relaxed, '
        'visible_amodal. '
        f'Got {MATCH_MODE!r}.')


def match_cost_limit(iou_thresh):
    if MATCH_MODE in ('iou', 'class_iou'):
        return 1 - iou_thresh
    return 1 - MATCH_MIN_SCORE


def load_float_map(env_name):
    raw = os.environ.get(env_name, '').strip()
    if not raw:
        return {}
    if os.path.exists(raw):
        with open(raw, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    else:
        payload = json.loads(raw)
    return {str(k): float(v) for k, v in payload.items()}


UNMATCHED_FACTOR_MAP = load_float_map('LT3D_UNMATCHED_FACTOR_JSON')
RESCUE_SCORE_MAP = load_float_map('LT3D_RESCUE_SCORE_JSON')
SCORE_3D_THRESH_MAP = load_float_map('LT3D_SCORE_3D_THRESH_JSON')


def unmatched_lidar_factor(class_name, default_factor):
    if class_name in UNMATCHED_FACTOR_MAP:
        return UNMATCHED_FACTOR_MAP[class_name]
    if class_name in RARE_SAFE_UNMATCHED_CLASSES:
        return RARE_SAFE_UNMATCHED_FACTOR
    return default_factor


def rescue_score_threshold(class_name):
    return RESCUE_SCORE_MAP.get(class_name, RESCUE_SCORE_THRESH)


def score_3d_threshold(class_name):
    return SCORE_3D_THRESH_MAP.get(class_name, score_3d_thresh)


def is_adult_child_relaxed_match(name_3d, name_2d):
    if not ADULT_CHILD_RELAXED:
        return False
    if name_3d == 'adult' and name_2d == 'child':
        return True
    return (
        ADULT_CHILD_BIDIRECTIONAL and
        name_3d == 'child' and
        name_2d == 'adult')


def build_lidar_candidate_mask(scores, labels):
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    base_mask = np.zeros_like(scores, dtype=bool)
    for cls_idx, class_name in enumerate(classes):
        class_mask = labels == cls_idx
        base_mask |= class_mask & (scores > score_3d_threshold(class_name))
    if not RESCUE_CLASSES:
        return base_mask, np.zeros_like(base_mask, dtype=bool)

    rescue_mask = np.zeros_like(base_mask, dtype=bool)
    for cls_idx, class_name in enumerate(classes):
        if class_name not in RESCUE_CLASSES:
            continue
        class_mask = labels == cls_idx
        rescue_mask |= class_mask & (scores > rescue_score_threshold(class_name))
    rescue_mask &= ~base_mask
    return base_mask | rescue_mask, rescue_mask


def get_image_for_view(data_path):
    """Avoid reading images when visualization is disabled."""
    if show:
        img = cv2.imread(resolve_image_path(data_path))
        if img is None:
            raise FileNotFoundError(resolve_image_path(data_path))
        return img, img.shape[0], img.shape[1]
    return None, DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH


def build_projected_boxes(corners_img, valid_z, height, width):
    """Build projected 2D boxes from projected 3D corners."""
    num_boxes = valid_z.shape[0]
    if num_boxes == 0:
        return []

    x = corners_img[:, :, 0].astype(np.float32)
    y = corners_img[:, :, 1].astype(np.float32)
    valid = valid_z.astype(bool)
    has_visible_corner = valid.sum(axis=1) >= 1
    if not has_visible_corner.any():
        return []

    min_col = np.clip(np.min(np.where(valid, x, np.inf), axis=1), 0, width)
    max_col = np.clip(np.max(np.where(valid, x, -np.inf), axis=1), 0, width)
    min_row = np.clip(np.min(np.where(valid, y, np.inf), axis=1), 0, height)
    max_row = np.clip(np.max(np.where(valid, y, -np.inf), axis=1), 0, height)

    keep = has_visible_corner & ((max_col - min_col) > 0) & ((max_row - min_row) > 0)
    indices = np.where(keep)[0]
    return [
        [int(aid), float(min_row[aid]), float(min_col[aid]),
         float(max_row[aid]), float(max_col[aid])]
        for aid in indices
    ]


def load_2d_detections(path):
    result_2d_det = []
    if not os.path.exists(path):
        return result_2d_det
    with open(path, 'r') as f:
        for objs in f:
            temp_list = objs.split()
            if len(temp_list) != 6:
                continue
            cls_id, x1, y1, x2, y2, score = temp_list
            score = float(score)
            if score < score_2d_thresh:
                continue
            name_2d = classes[int(cls_id)]
            result_2d_det.append(
                [name_2d, score, float(y1), float(x1), float(y2), float(x2)])
    return result_2d_det

def make_fine_fusion(iou_thresh, score_cal_times_method_dict, lidar_no_match_times):
    print('fusion_ablation_config:', {
        'score_2d_thresh': score_2d_thresh,
        'score_3d_thresh': score_3d_thresh,
        'score_3d_thresh_map': SCORE_3D_THRESH_MAP,
        'match_mode': MATCH_MODE,
        'iou_thresh': iou_thresh,
        'relaxed_iou_thresh': RELAXED_IOU_THRESH,
        'ioa_det_thresh': IOA_DET_THRESH,
        'ioa_proj_thresh': IOA_PROJ_THRESH,
        'use_center_inside': USE_CENTER_INSIDE,
        'center_inside_score': CENTER_INSIDE_SCORE,
        'use_proj_center_inside': USE_PROJ_CENTER_INSIDE,
        'proj_center_inside_score': PROJ_CENTER_INSIDE_SCORE,
        'mismatch_policy': MISMATCH_POLICY,
        'default_unmatched_lidar_factor': lidar_no_match_times,
        'rare_safe_unmatched_classes': sorted(RARE_SAFE_UNMATCHED_CLASSES),
        'rare_safe_unmatched_factor': RARE_SAFE_UNMATCHED_FACTOR,
        'unmatched_factor_map': UNMATCHED_FACTOR_MAP,
        'rescue_classes': sorted(RESCUE_CLASSES),
        'rescue_score_thresh': RESCUE_SCORE_THRESH,
        'rescue_score_map': RESCUE_SCORE_MAP,
        'rescue_min_2d_score': RESCUE_MIN_2D_SCORE,
        'rescue_min_match_score': RESCUE_MIN_MATCH_SCORE,
        'rescue_unmatched_factor': RESCUE_UNMATCHED_FACTOR,
        'adult_child_relaxed': ADULT_CHILD_RELAXED,
        'adult_child_bidirectional': ADULT_CHILD_BIDIRECTIONAL,
        'adult_child_min_2d_score': ADULT_CHILD_MIN_2D_SCORE,
        'adult_child_min_match_score': ADULT_CHILD_MIN_MATCH_SCORE,
    })

    all_num = 0
    not_project_num = 0
    info_index = 0
    max_frames = int(os.environ.get('LT3D_MAX_FRAMES', '0') or 0)
    iterable_info_data = info_data[:max_frames] if max_frames > 0 else info_data
    for infos in tqdm(iterable_info_data):                   
        raw_scores = res3d_fusion[info_index]['pts_bbox']['scores_3d']
        raw_labels = res3d_fusion[info_index]['pts_bbox']['labels_3d']
        mask_score, rescue_mask_raw = build_lidar_candidate_mask(
            raw_scores, raw_labels)
        rescue_candidate_flags = rescue_mask_raw[mask_score]
        res3d_fusion[info_index]['pts_bbox']['boxes_3d'] = res3d_fusion[info_index]['pts_bbox']['boxes_3d'][mask_score]
        res3d_fusion[info_index]['pts_bbox']['scores_3d'] = res3d_fusion[info_index]['pts_bbox']['scores_3d'][mask_score]
        res3d_fusion[info_index]['pts_bbox']['labels_3d'] = res3d_fusion[info_index]['pts_bbox']['labels_3d'][mask_score]
        pred_boxes = res3d_fusion[info_index]['pts_bbox']['boxes_3d'].tensor.numpy()

        '''Del pre 3d Boxes'''
        if draw_gt == 1:
            pred_boxes =  np.zeros((0, 3), dtype=np.float32)
            scores = np.zeros((0, 1), dtype=np.float32)

        if pred_boxes.shape[0] == 0:
            corners_lidar = np.zeros((0, 3), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            names = np.zeros((0,), dtype=object)
            rescue_flags = np.zeros((0,), dtype=bool)
        else:
            corners_lidar = res3d_fusion[info_index]['pts_bbox']['boxes_3d'].corners.numpy().reshape(-1, 3)
            scores = res3d_fusion[info_index]['pts_bbox']['scores_3d']
            scores = np.array(scores, dtype=np.float32)
            names = res3d_fusion[info_index]['pts_bbox']['labels_3d']
            names = np.array(names)
            names = np.array([classes[name] for name in names])      
            rescue_flags = np.asarray(rescue_candidate_flags, dtype=bool)

        pred_flags = np.ones((corners_lidar.shape[0]//8,), dtype=bool)
        fusion_flag = np.zeros((names.shape[0]), dtype=bool)
        
        if pred_boxes.shape[0] == 0:
            info_index += 1
            continue

        projection_flag = np.zeros((pred_boxes.shape[0]), dtype=np.bool_)

        for view_index, view in enumerate(views):
            result_2d_pre = []
            img, img_height, img_width = get_image_for_view(infos['cams'][view]['data_path'])
            corners_img, valid_z = lidar2img(corners_lidar, infos['cams'][view])
            valid_shape = check_point_in_img(corners_img, img_height, img_width)
            valid_all = np.logical_and(valid_z, valid_shape)
            valid_z = valid_z.reshape(-1, 8)
            valid_shape = valid_shape.reshape(-1, 8)
            valid_all = valid_all.reshape(-1, 8)        
            corners_img = corners_img.reshape(-1, 8, 2).astype(np.int32)

            '''Generate 3D results'''
            result_2d_pre = build_projected_boxes(corners_img, valid_z, img_height, img_width)
            if show:
                for aid, min_row, min_col, max_row, max_col in result_2d_pre:
                    cv2.rectangle(img, (int(min_col), int(min_row)), (int(max_col), int(max_row)), (255, 0, 0), 2)
                    cv2.putText(img, names[aid], (int(min_col), int(min_row)), font, 1, (0, 0, 255), 1)
                    cv2.putText(img, str(scores[aid])[:4], (int(min_col), int(min_row)+30), font, 1, (255, 0, 255), 1)

            '''Generate 2D results from offline'''
            if get_2d:
                res_2d_file_path = res2d_dir+infos['token']+"@" +view+'.txt'
                result_2d_det = load_2d_detections(res_2d_file_path)
                if show:
                    for name_2d, score, y1, x1, y2, x2 in result_2d_det:
                        cv2.rectangle(img, (int(float(x1)), int(float(y1))), (int(float(x2)), int(float(y2))), (0, 255, 0), 2)
                        cv2.putText(img, name_2d, (int(float(x1)), int(float(y1))-35), font, 1, (0, 0, 255), 2)
                        cv2.putText(img, str(score)[:4], (int(float(x1)), int(float(y1))-10), font, 1, (0, 0, 255), 2)  
            '''3D boxes show'''
            if show:
                for aid in range(valid_all.shape[0]):
                    score = scores[aid]
                    name = names[aid]                  
                    if valid_z[aid].sum() >= 4: 
                        min_col = max(min(corners_img[aid, valid_z[aid], 0].min(), img_width), 0)
                        max_col = max(min(corners_img[aid, valid_z[aid], 0].max(), img_width), 0)
                        min_row = max(min(corners_img[aid, valid_z[aid], 1].min(), img_height), 0)
                        max_row = max(min(corners_img[aid, valid_z[aid], 1].max(), img_height), 0) 
                        if (max_col - min_col) == 0 or (max_row - min_row) == 0:
                            continue                                          
                        cv2.putText(img, name, (int(min_col), int(min_row)-35), font, 1, (0, 0, 255), 2)
                        cv2.putText(img, str(score)[:4],  (int(min_col), int(min_row)-10), font, 1, (0, 0, 255), 2)
                        for index in draw_boxes_indexes_img_view:
                                corners_img[aid, index[0]][0] = min(max(corners_img[aid, index[0]][0], 0), img_width)
                                corners_img[aid, index[0]][1] = min(max(corners_img[aid, index[0]][1], 0), img_height)  
                                corners_img[aid, index[1]][0] = min(max(corners_img[aid, index[1]][0], 0), img_width)  
                                corners_img[aid, index[1]][1] = min(max(corners_img[aid, index[1]][1], 0), img_height)                                               
                                cv2.line(img,
                                            corners_img[aid, index[0]],
                                            corners_img[aid, index[1]],
                                            color=[255, 255, 0],
                                            thickness=scale_factor) 

            if show:
                cv2.imwrite("./results/"+infos['token']+"&"+view+".jpg", img)
            
            '''Math and cal IOU'''
            if get_2d:
                result = []  
                if len(result_2d_pre) > 0 and len(result_2d_det) > 0:
                    pre_boxes = [item[1:] for item in result_2d_pre]
                    det_boxes = [item[2:] for item in result_2d_det]
                    pre_names = [names[item[0]] for item in result_2d_pre]
                    det_names = [item[0] for item in result_2d_det]
                    score_mat = compute_match_matrix(
                        pre_boxes, det_boxes, pre_names, det_names, iou_thresh)
                    _, x, y = lap.lapjv(
                        1-score_mat, extend_cost=True,
                        cost_limit=match_cost_limit(iou_thresh))
                    for i, j in enumerate(y):
                        if j != -1:
                            result.append([j, i, 1.0]) 
            
            '''spatio and semantic fusion'''
            if get_2d:
                index_3d_list = []
                if len(result) > 0:
                    for aid in range(len(result)):
                        result[aid][-1] = score_mat[result[aid][0], result[aid][1]]
                        index_3d = result_2d_pre[result[aid][0]][0]
                        name_2d, score_2d = result_2d_det[result[aid][1]][0], result_2d_det[result[aid][1]][1]
                        name_3d, score_3d = names[index_3d], scores[index_3d] 
                        if rescue_flags[index_3d]:
                            if (score_2d < RESCUE_MIN_2D_SCORE or
                                    result[aid][-1] < RESCUE_MIN_MATCH_SCORE):
                                continue
                        '''You can choose coarse or fine-frained classes matching for semantic fusion'''
                        # if classes_coarse[classes.index(name_3d)] == classes_coarse[classes.index(name_2d)]:
                        if classes.index(name_3d) == classes.index(name_2d):
                            fusion_name = name_2d 
                            score_cal_times_dict = score_cal_times_method_dict['bay']
                            score_cal_times = score_cal_times_dict[fusion_name]['c']
                            p = score_cal_times_dict[fusion_name]['p']

                            fusion_2d = score_cal_times*score_2d
                            fusion_2d_no = 1 - fusion_2d
                            fusion_3d = score_3d
                            fusion_3d_no = 1 - fusion_3d                            
                            fusion_2d_3d = fusion_2d*fusion_3d/p
                            fusion_2d_3d_no = fusion_2d_no*fusion_3d_no/(1-p)
                            fusion_score = fusion_2d_3d / ((fusion_2d_3d+fusion_2d_3d_no)*1.0)                                             
                        else:
                            adult_child_relaxed = is_adult_child_relaxed_match(
                                name_3d, name_2d)
                            if adult_child_relaxed:
                                if (score_2d < ADULT_CHILD_MIN_2D_SCORE or
                                        result[aid][-1] < ADULT_CHILD_MIN_MATCH_SCORE):
                                    continue
                            if MISMATCH_POLICY == 'relabel':
                                fusion_score = float(score_2d)
                                fusion_name = name_2d
                            elif MISMATCH_POLICY == 'keep_3d':
                                fusion_score = float(score_3d)
                                fusion_name = name_3d
                            elif MISMATCH_POLICY == 'ignore':
                                continue
                            else:
                                raise ValueError(
                                    'LT3D_MISMATCH_POLICY must be relabel, '
                                    f'keep_3d, or ignore. Got {MISMATCH_POLICY!r}.')
                        projection_flag[index_3d] = True
                        index_3d_list.append(index_3d)  

                        '''Overlapping fusion'''
                        if fusion_flag[index_3d] == True:
                            if fusion_score > res3d_fusion[info_index]['pts_bbox']['scores_3d'][index_3d]:
                                '''if calibrating, you should comment out them'''
                                res3d_fusion[info_index]['pts_bbox']['scores_3d'][index_3d] = fusion_score
                                res3d_fusion[info_index]['pts_bbox']['labels_3d'][index_3d] = classes.index(fusion_name)
                        else:                           
                            res3d_fusion[info_index]['pts_bbox']['scores_3d'][index_3d] = fusion_score
                            res3d_fusion[info_index]['pts_bbox']['labels_3d'][index_3d] = classes.index(fusion_name)
                            fusion_flag[index_3d] = True 
            
                '''For no Matched obj, reducing the lidar detection socres'''
                for temp in result_2d_pre:
                    index_3d = temp[0]
                    if index_3d not in index_3d_list:
                        score_3d = scores[index_3d]
                        name_3d = names[index_3d]
                        if fusion_flag[index_3d] == True:
                            fusion_score = res3d_fusion[info_index]['pts_bbox']['scores_3d'][index_3d]
                            fusion_name = classes[res3d_fusion[info_index]['pts_bbox']['labels_3d'][index_3d]]
                        else:
                            fusion_score = score_3d
                            fusion_name = name_3d
                            if rescue_flags[index_3d]:
                                factor = RESCUE_UNMATCHED_FACTOR
                            else:
                                factor = unmatched_lidar_factor(
                                    fusion_name, lidar_no_match_times)
                            res3d_fusion[info_index]['pts_bbox']['scores_3d'][index_3d] = float(score_3d * factor)

        if rescue_flags.any():
            unmatched_rescue = rescue_flags & (~fusion_flag)
            if unmatched_rescue.any():
                for index_3d in np.where(unmatched_rescue)[0]:
                    res3d_fusion[info_index]['pts_bbox']['scores_3d'][index_3d] = float(
                        scores[index_3d] * RESCUE_UNMATCHED_FACTOR)
        
        all_num = all_num + pred_boxes.shape[0]
        if int(projection_flag.sum()) < pred_boxes.shape[0]:
            not_project_num = not_project_num + (pred_boxes.shape[0] - projection_flag.sum())
        info_index += 1
    not_project_scale = not_project_num * 1.0 / all_num    

    print('not_project_num:', not_project_num)
    print('all_num:', all_num)
    print('not_project_scale:', not_project_scale)

    if write_res:
        if write_format == 'pkl':
            res3d_fusion_path = os.path.join(os.environ.get('LT3D_OUTPUT_DIR', './results'), 'prediction_fine_fusion.pkl')
            save(res3d_fusion, res3d_fusion_path)
        else:
            '''Trans format to json results'''
            from mmcv import Config
            try:
                from mmdet.utils import compat_cfg
            except ImportError:
                from mmdet3d.utils import compat_cfg
            from mmdet3d.datasets import build_dataset
            config_file = os.environ.get('LT3D_CONFIG_FILE', 'configs/centerpoint/lt3d/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_50m_wide_hierarchy_tta_20e_nus.py')
            out_path = os.environ.get('LT3D_OUTPUT_DIR', './results')
            cfg_lidar_3d = Config.fromfile(config_file)
            cfg_lidar_3d = compat_cfg(cfg_lidar_3d); cfg_lidar_3d.data_root = os.environ.get('LT3D_DATA_ROOT', cfg_lidar_3d.data_root)
            dataset = build_dataset(cfg_lidar_3d.data.test)
            result_files, tmp_dir = dataset.format_results(res3d_fusion, out_path, None)
