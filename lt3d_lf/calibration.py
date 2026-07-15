"""Detector-specific score calibration used by the original CenterPoint+DINO run."""

from copy import deepcopy
import json
from pathlib import Path


_BAYES_COEFFICIENTS = {
    'car': 0.6,
    'truck': 1.2,
    'trailer': 0.6,
    'bus': 0.5,
    'construction_vehicle': 1.0,
    'bicycle': 0.8,
    'motorcycle': 0.7,
    'emergency_vehicle': 1.9,
    'adult': 0.3,
    'child': 4.1,
    'police_officer': 1.6,
    'construction_worker': 0.4,
    'stroller': 1.4,
    'personal_mobility': 1.9,
    'pushable_pullable': 1.2,
    'debris': 0.5,
    'traffic_cone': 1.1,
    'barrier': 1.1,
}


def centerpoint_dino_calibration():
    """Return the original GitHub CenterPoint+DINO Bayes calibration table."""
    calibration = {
        class_name: {'c': coefficient, 'p': 0.1}
        for class_name, coefficient in _BAYES_COEFFICIENTS.items()
    }
    return deepcopy({'bay': calibration})


def calibration_from_json(path):
    """Load a detector-specific per-class calibration manifest."""
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    coefficients = payload['coefficients']
    return {
        'bay': {
            class_name: {'c': float(value), 'p': 0.1}
            for class_name, value in coefficients.items()
        }
    }
