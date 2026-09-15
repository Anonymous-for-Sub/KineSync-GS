#!/usr/bin/env python3
"""Native high-resolution re-render of fixed heldout states, not new rollouts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import yaml

from calibrate_risk_atomic import evidence
from kinesync.external_abc import require_gpu_allocation
from kinesync.external_gs.franka import load_franka_external_gs
from kinesync.external_gs.risk_atomic import commit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    job = require_gpu_allocation()
    frozen = json.loads(args.freeze.read_text())
    for source, expected in frozen['sources'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen source changed: {source}')
    config = yaml.safe_load((args.run / 'config_source.yaml').read_text())
    with (args.run / 'candidate_evidence.csv').open() as handle:
        candidates = list(csv.DictReader(handle))
    if len(candidates) != 48 or any(r['split'] != 'heldout' for r in candidates):
        raise ValueError('Expected complete 48-condition heldout matrix')
    with (args.run / 'results.csv').open() as handle:
        old = {(r['state_id'], r['method']): r for r in csv.DictReader(handle)}
    args.output.mkdir(parents=True, exist_ok=False)
    asset = load_franka_external_gs(config['assets']['root'])
    states = np.asarray([r['qpos_rad'] for split in ('development_cases', 'heldout_cases')
                         for r in config[split]], dtype=float)
    height, width = 1080, 1440
    cameras = asset.framed_analysis_cameras(states, camera_specs=config['cameras']['specs'],
                                           width=width, height=height)
    # Use the complete source PLY for display; estimates remain the original 512/link run.
    renderer = asset.cuda_backend(output_size=(height, width), per_link=100000,
                                  device=torch.device('cuda'), cameras=cameras)
    renderer.background = tuple(config['render']['background'])
    receipts = []
    rendered = {}
    for index, row in enumerate(candidates):
        measured, candidate, truth = [json.loads(row[k]) for k in
                                      ('measured_qpos_rad', 'candidate_qpos_rad', 'true_qpos_rad')]
        final, accepted = commit(measured, candidate, evidence(row), frozen['thresholds'])
        values = {'reference': truth, 'baseline': measured, 'unguarded': candidate, 'ours': final,
                  'legacy': json.loads(old[(row['state_id'], 'component_guard')]['final_qpos_rad'])}
        for role, q in values.items():
            key = tuple(q)
            if key not in rendered:
                with torch.no_grad():
                    buffers = renderer.render_buffers(torch.tensor(q, device='cuda', dtype=torch.float32))
                rendered[key] = {camera: np.clip(np.rint(buffer.rgb.cpu().numpy() * 255), 0, 255).astype('uint8')
                                 for camera, buffer in buffers.items()}
            for camera, image in rendered[key].items():
                filename = f"NATIVE5__{row['state_id']}__{role}__{camera}__1440x1080.png"
                Image.fromarray(image).save(args.output / filename, optimize=True)
                receipts.append({'state_id': row['state_id'], 'case_id': row['case_id'],
                                 'condition_id': row['condition_id'], 'role': role, 'camera': camera,
                                 'qpos_rad': q, 'qmae_deg': float(np.rad2deg(np.mean(np.abs(np.array(q) - truth)))),
                                 'atomic_gate_accepted': accepted, 'filename': filename,
                                 'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                                 'file_sha256': hashlib.sha256((args.output / filename).read_bytes()).hexdigest()})
        print(f"Rendered {index + 1}/{len(candidates)}: {row['state_id']}", flush=True)
    payload = {'schema': 'kinesync.paper.heldout-native-rerender.v1', 'slurm_job_id': job,
               'source_run': str(args.run.resolve()), 'source_freeze': str(args.freeze.resolve()),
               'source_csv_sha256': hashlib.sha256((args.run / 'candidate_evidence.csv').read_bytes()).hexdigest(),
               'source_asset_sha256': asset.provenance.asset_sha256,
               'renderer': 'native Gaussian rasterizer; complete PLY for visualization only',
               'evaluation_gaussians_per_link': config['render']['gaussians_per_link'],
               'display_gaussians': int(len(asset.local_xyz)), 'dimensions': [width, height],
               'reference_kind': 'controlled same-asset render; not a real camera image',
               'cameras': {name: {'intrinsic': camera.intrinsic.tolist(),
                                   'world_to_camera': camera.world_to_camera.tolist()}
                           for name, camera in cameras.items()}, 'frames': receipts}
    (args.output / 'native_manifest.json').write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
