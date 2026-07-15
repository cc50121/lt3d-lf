import json
import os
classes = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
        'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility', 
        'pushable_pullable', 'debris', 'traffic_cone', 'barrier']

def lca0(eval_cls='all'):
    global classes
    if eval_cls != 'all':
        classes = [eval_cls]

    dis_list = ['0.5/0', '1.0/0', '2.0/0', '4.0/0']
    result_dict = {}
    result_ap = 0
    output_dir = os.environ.get('LT3D_OUTPUT_DIR', './results')
    result_json_path_list = [os.path.join(output_dir, 'metrics_summary.json')]
    for result_json_path in result_json_path_list:
        result_ap = 0
        result = json.load(open(result_json_path, 'r'))
        for cls in classes:
            result_dict[cls] = 0
            for dis in dis_list:
                result_dict[cls] = result_dict[cls] + result['label_aps'][cls][dis]
            result_dict[cls] = result_dict[cls] / 4.0
            result_ap += result_dict[cls]
        result_ap = result_ap / (len(classes)*1.0)
        print('result_dict:', result_dict)
        print('result_ap:', result_ap)
    return result_ap
    
