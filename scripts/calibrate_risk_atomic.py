#!/usr/bin/env python3
"""Freeze risk-aware atomic thresholds before the predeclared heldout run."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from kinesync.external_gs.risk_atomic import RiskRecord, calibrate
from kinesync.guard.schema import UpdateEvidence


def evidence(row):
    return UpdateEvidence(int(row['view_count']), float(row['visual_gain_ratio']),
                          float(row['gradient_cosine']), float(row['correction_cosine']),
                          float(row['relative_correction_disagreement']),
                          float(row['candidate_bound_fraction']), row['finite'].lower() == 'true')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--development', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--heldout-output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.heldout_output.exists():
        raise ValueError('Refusing to overwrite a freeze or calibrate after heldout output exists')
    source = args.development / 'candidate_evidence.csv'
    with source.open() as handle:
        rows = list(csv.DictReader(handle))
    records = [RiskRecord(r['state_id'], r['split'], evidence(r),
                          float(np.mean(json.loads(r['initial_abs_error_rad']))),
                          float(np.mean(json.loads(r['candidate_abs_error_rad'])))) for r in rows]
    result = calibrate(records)
    result['created_utc'] = datetime.now(timezone.utc).isoformat()
    result['protocol'] = 'One heldout evaluation; fixed 12 poses x 4 conditions; no test-time tuning'
    result['config'] = yaml.safe_load(args.config.read_text())
    result['sources'] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [source, args.config, Path(__file__),
                                   Path('src/kinesync/external_gs/risk_atomic.py'),
                                   Path('scripts/run_external_gs_guard.py')]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('config', 'sources')}, indent=2))


if __name__ == '__main__':
    main()
