# Configuration guide

The committed files are portable templates and frozen, non-identifying protocol
definitions. Keep local data paths and camera serial numbers in untracked files.

| File | Purpose |
|---|---|
| `rt10_piper_d455_live.yaml` | One-camera PiPER/D455 read-only capture |
| `rt10_piper_dual_d455_live.yaml` | Two-camera PiPER/D455 read-only capture |
| `rt10_camera_calibration_template.yaml` | Camera intrinsics/extrinsics template |
| `rt10_capture_manifest_template.yaml` | Session provenance template |
| `rt10_capture_schedule.yaml` | Calibration and held-out capture schedule |
| `rt10_frozen_temporal_guard.json` | Development-calibrated temporal verifier |
| `rt3_piper_guard_dev.yaml` | Spatial verifier development matrix |
| `rt3_piper_guard_test.yaml` | Held-out spatial evaluation matrix |
| `rt4_piper_temporal_dev.yaml` | Controlled temporal development protocol |
| `rt5_piper_observation_audit.yaml` | Observation-resolution audit |
| `rt6_piper_fresh_matrix.yaml` | Frozen held-out whole-state protocol |

Copy a template to a local filename before adding asset paths or hardware
identifiers. Local configs, datasets, assets, and run directories are ignored by
Git.
