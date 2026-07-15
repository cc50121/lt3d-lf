"""Directional per-class calibration search, following legacy/main_tune.py."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .calibration import _BAYES_COEFFICIENTS


CLASSES = tuple(_BAYES_COEFFICIENTS)


def tag(factor):
    return f'c_{factor:.2f}'.replace('.', '_')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid-dir', type=Path, required=True)
    parser.add_argument('--camera-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--state-output', type=Path, required=True)
    parser.add_argument('--step', type=float, default=0.1)
    parser.add_argument('--min-factor', type=float, default=0.1)
    parser.add_argument('--max-factor', type=float, default=5.0)
    parser.add_argument('--jobs', type=int, default=5)
    return parser.parse_args()


def rounded(value):
    return round(value + 1e-9, 2)


def manifest_path(grid_dir, factor):
    return grid_dir / tag(factor) / 'candidate_metrics.json'


def load_candidates(grid_dir):
    records = {}
    for path in grid_dir.glob('*/candidate_metrics.json'):
        payload = json.loads(path.read_text(encoding='utf-8'))
        records[rounded(float(payload['factor']))] = payload
    return records


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def run_missing(args, factors):
    factors = sorted(set(factors))
    for offset in range(0, len(factors), args.jobs):
        batch = factors[offset:offset + args.jobs]
        processes = []
        for factor in batch:
            work_dir = args.grid_dir / tag(factor)
            work_dir.mkdir(parents=True, exist_ok=True)
            log = args.grid_dir / f'{tag(factor)}.log'
            command = [
                sys.executable, '-m', 'lt3d_lf.tune',
                '--factor', f'{factor:.2f}',
                '--camera-results', str(args.camera_results),
                '--work-dir', str(work_dir),
            ]
            with log.open('ab') as handle:
                processes.append((factor, subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)))
        for factor, process in processes:
            code = process.wait()
            if code:
                raise RuntimeError(f'Candidate {factor:.2f} failed; inspect {args.grid_dir / (tag(factor) + ".log")}')


def metric(candidates, factor, class_name):
    return candidates[rounded(factor)]['lca0_per_class'][class_name]


def advance(state, candidates, step, minimum, maximum):
    """Advance one legacy-style directional search state, or return required factors."""
    base = state['base']
    phase = state['phase']
    class_name = state['class_name']
    required = []

    if phase == 'initial':
        up = rounded(base + step)
        for factor in (base, up):
            if factor not in candidates:
                required.append(factor)
        if required:
            return required
        base_score = metric(candidates, base, class_name)
        up_score = metric(candidates, up, class_name)
        state['trajectory'].extend([
            {'factor': base, 'lca0': base_score},
            {'factor': up, 'lca0': up_score},
        ])
        state['best_factor'] = base
        state['best_lca0'] = base_score
        if up_score > base_score:
            state['direction'] = 1
            state['current'] = up
            state['best_factor'] = up
            state['best_lca0'] = up_score
            state['phase'] = 'continue'
        else:
            state['direction'] = -1
            state['current'] = base
            state['phase'] = 'continue'
        return []

    if phase != 'continue':
        return []

    next_factor = rounded(state['current'] + state['direction'] * step)
    if next_factor < minimum or next_factor > maximum:
        state['phase'] = 'done'
        return []
    if next_factor not in candidates:
        return [next_factor]

    next_score = metric(candidates, next_factor, class_name)
    state['trajectory'].append({'factor': next_factor, 'lca0': next_score})
    if next_score > state['best_lca0']:
        state['current'] = next_factor
        state['best_factor'] = next_factor
        state['best_lca0'] = next_score
    else:
        state['phase'] = 'done'
    return []


def main():
    args = parse_args()
    args.grid_dir = args.grid_dir.expanduser().resolve()
    args.camera_results = args.camera_results.expanduser().resolve()
    states = {
        class_name: {
            'class_name': class_name,
            'base': rounded(value),
            'phase': 'initial',
            'direction': None,
            'current': None,
            'best_factor': None,
            'best_lca0': None,
            'trajectory': [],
        }
        for class_name, value in _BAYES_COEFFICIENTS.items()
    }

    while True:
        candidates = load_candidates(args.grid_dir)
        needed = []
        for state in states.values():
            needed.extend(advance(state, candidates, args.step, args.min_factor, args.max_factor))
        write_json(args.state_output, {'states': states, 'available_factors': sorted(candidates)})
        missing = [factor for factor in sorted(set(needed)) if factor not in candidates]
        if missing:
            run_missing(args, missing)
            continue
        if all(state['phase'] == 'done' for state in states.values()):
            break

    result = {
        'detector': 'GroundingDINO original category-name prompts',
        'objective': 'validation LCA0 directional search from DINO coefficients',
        'step': args.step,
        'coefficients': {name: state['best_factor'] for name, state in states.items()},
        'per_class': states,
    }
    write_json(args.output, result)
    print(args.output)


if __name__ == '__main__':
    main()
