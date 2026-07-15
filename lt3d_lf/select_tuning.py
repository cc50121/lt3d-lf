"""Select the best per-class LCA0 coefficient from completed grid candidates."""

import argparse
import json
from pathlib import Path

from .tune import CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    candidates = []
    for path in sorted(args.grid_dir.glob('*/candidate_metrics.json')):
        candidates.append(json.loads(path.read_text(encoding='utf-8')))
    if not candidates:
        raise SystemExit('No completed candidate_metrics.json files found.')

    best = {}
    table = {}
    for class_name in CLASSES:
        ranked = sorted(
            ((item['lca0_per_class'][class_name], item['factor']) for item in candidates),
            key=lambda item: (-item[0], item[1]),
        )
        score, factor = ranked[0]
        best[class_name] = factor
        table[class_name] = {
            'factor': factor,
            'lca0': score,
            'tested': len(ranked),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        'detector': 'GroundingDINO original category-name prompts',
        'objective': 'validation LCA0 over 0.5/0, 1.0/0, 2.0/0, 4.0/0',
        'coefficients': best,
        'per_class': table,
        'grid_candidates': [item['factor'] for item in candidates],
    }, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
