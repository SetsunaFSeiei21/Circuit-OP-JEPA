#!/usr/bin/env python3
"""Export an explicitly labeled best/worst-seed view of a completed pin run."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from final_exam.summarize_pin import export_results_pin


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='Saved run root, training_pin directory, or results_pin.json')
    parser.add_argument('--k', type=int, required=True, help='Number of complete seeds to keep per listed model')
    parser.add_argument('--models-better', nargs='*', default=[], help='Keep the k highest test macro R2 seeds')
    parser.add_argument('--models-worse', nargs='*', default=[], help='Keep the k lowest test macro R2 seeds')
    parser.add_argument('--output', help='New filename ending in _pin.xlsx; never overwrites source summary')
    parser.add_argument('--decimals', type=int, default=4)
    parser.add_argument('--recover-export', action='store_true', help='Accept failed_export after strict validation')
    args = parser.parse_args(argv)
    try:
        output = export_results_pin(
            args.source, args.output, decimals=args.decimals, recover_export=args.recover_export,
            seed_selection=dict(k=args.k, models_better=args.models_better, models_worse=args.models_worse))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print('Ranking: test macro R2; retain complete seeds. See SeedSelection for every candidate.')
    print(f'Excel: {output}')


if __name__ == '__main__':
    main()
