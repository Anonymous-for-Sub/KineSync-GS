# KineSync-GS

**Verified Visual State Synchronization for Articulated Gaussian Twins**

[Project page](https://anonymous-for-sub.github.io/KineSync-GS/) | [Paper](https://anonymous-for-sub.github.io/KineSync-GS/static/paper/kinesync-gs.pdf) | [Documentation](#usage)

KineSync-GS synchronizes articulated Gaussian robot twins with physical robot
state. Its central design separates state estimation from state-changing
authority: a Gaussian renderer or motion estimator proposes a correction,
independent evidence verifies the proposal, and the system either commits the
complete state update or retains the incoming measurement. Spatial and temporal
synchronization therefore share one interface:

```text
proposal -> independent evidence -> verified update / measurement fallback
```

## Highlights

- Kinematically bound 3D Gaussian observation models for articulated robots.
- Bounded whole-state correction rather than independent per-joint edits.
- Cross-view evidence and development-calibrated atomic commit/fallback.
- Motion-based camera/telemetry alignment with verified resampling.
- Recorded-real Component-GS and Hybrid-GS replay adapters.
- Read-only PiPER + RealSense D455 capture and synchronization integration.
- Robot-independent state/action adapters, including OpenVLA-OFT compatibility.

The evaluation includes controlled Franka view conflicts, recorded PiPER
trajectories, two Gaussian backends, temporal jitter/dropout, an ABC bimanual
simulation interface, and online PiPER manipulation. Whole-state verification
reduces aggregate Franka qMAE from 0.590 to 0.322 degrees without a harmful
update across 48 held-out conditions. In online PiPER evaluation, the complete
system reaches 3.07 mrad qMAE with 0.36% harmful updates; supplying the verified
state to the same policy raises task-macro success from 58.33% to 71.67%.

## Repository layout

```text
configs/       Portable evaluation and hardware configuration templates
docs/          Static GitHub Pages project website
scripts/       Analysis and external-stack experiment entry points
src/kinesync/  Core library, CLIs, hardware adapters, and visualizers
tests/         Unit and asset-independent integration tests
```

Large robot recordings, Gaussian assets, model checkpoints, and generated run
directories are intentionally kept outside the repository. Configure their
locations locally; no machine-specific paths are embedded in this release.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For a PiPER arm with one or two RealSense D455 cameras:

```bash
python -m pip install -e '.[hardware]'
```

## Usage

### 1. Run the asset-independent test suite

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The public suite is asset-independent. GPU renderer checks and evaluations that
consume downloaded robot assets are launched separately with their experiment
configuration.

### 2. Discover PiPER and D455 devices

The discovery path is read-only and does not send robot commands.

```bash
kinesync-rt10-discover
kinesync-rt10-discover \
  --write-config configs/local-piper-d455.yaml \
  --run-root hardware-runs/rt10
```

Review the generated file before capture. The supplied single- and dual-camera
templates keep `command_mode: disabled`.

```bash
kinesync-rt10-live --config configs/rt10_piper_d455_live.yaml
# or
kinesync-rt10-live --config configs/rt10_piper_dual_d455_live.yaml
```

### 3. Evaluate temporal synchronization

Place the recorded Take-Pens files under `data/take_pens/`, or set the dataset
root in a local YAML derived from the public template.

```bash
kinesync-rt9t \
  --config configs/local-temporal.yaml \
  --run-id rt9-evaluation
```

### 4. Reproduce controlled verification

The spatial experiments expect a robot URDF, link-bound Gaussian assets,
camera calibration, and frozen development artifacts. Create a local config
from the templates in `configs/` and invoke the corresponding CLI:

```bash
kinesync-rt6f --config configs/local-spatial.yaml --run-id rt6-evaluation
```

Every run writes a manifest, resolved configuration, source fingerprints,
case-level results, aggregate metrics, and selected visualizations.

## Hardware boundary

KineSync-GS reads PiPER telemetry and D455 images. Robot motion remains under
the robot's existing controller; this repository does not replace that control
path. The live integration validates SDK compatibility, timestamps observations,
estimates synchronized state, applies the frozen evidence guard, and publishes
auditable state artifacts.

## Website

The project page is a zero-build static site under `docs/`. GitHub Pages can be
configured to deploy from the `main` branch and `/docs` directory.

For local preview:

```bash
python -m http.server 8000 --directory docs
```

Then open `http://localhost:8000`.

## Citation

```bibtex
@inproceedings{kinesyncgs2027,
  title     = {KineSync-GS: Verified Visual State Synchronization for Articulated Gaussian Twins},
  author    = {Anonymous Authors},
  booktitle = {Under Review},
  year      = {2027}
}
```

## License

Code is released under the [MIT License](LICENSE). External datasets, robot
assets, simulators, checkpoints, and SDKs retain their original licenses.
