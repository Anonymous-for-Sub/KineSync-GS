# KineSync-GS narrated method

The presentation at `explainer.html` follows a correction from its physical
observation through proposal generation, evidence and publication of the
synchronized state. It uses the project palette and the existing static site.

## Story and computations

1. Real2Sim2Real: physical articulation, image capture and a link-bound Gaussian twin.
2. Motivation: inconsistent observations make image fit and state error diverge.
3. The contract: a bounded vector reaches a whole-state update or measurement fallback.
4. Spatial evidence: view-specific correction vectors produce normalized disagreement.
5. Temporal evidence: sampled motion signals, normalized correlation and verified resampling.
6. Controlled Franka: fixed reported qMAE endpoints and aggregate useful/harmful counts.
7. Mechanism: shared joint calibration, view-specific residuals and reported retention.
8. Real-world PiPER: state-indexed Gaussian recovery and a recorded physical execution.
9. Online operation: asynchronous streams enter temporal and spatial verification.
10. Manipulation: raw and synchronized state feed the same policy; reported task success.
11. A shared state interface for Gaussian twins, simulation and robot policies.

Mechanism geometry, correction vectors, motion signals and packet paths are
deterministic teaching examples. Recorded media integrated into the main stage preserve their source frames
and normal playback speed. Result labels are the paper's reported aggregates;
animated reveals do not interpolate invented intermediate statistics. The
48-condition count grid does not encode individual case identities.

The spatial example implements the scalar 0.18-rad bound and normalized
cross-view disagreement, with the reported Franka threshold 0.42671. Its two
branches commit the entire proposed vector or retain the entire measurement.
The temporal example computes Pearson correlation over sampled motion signals
and searches lag at 10-ms spacing. These are inspectable explanations of the
method, not additional research evaluations.

## Recorded narration

`NARRATION.md` records the current audio status. The manifest stores actual MP3
durations, SHA-256 identities, script-matched sentence boundaries and visual
keyframes. All eleven supplied recordings are published unchanged. Chapter 03
uses the corrected 16.296-second recording. The total duration is 174.192 seconds.

Audio time drives chapter progress and the centered, highlighted captions. Mute changes only
audibility; pause stops narration, diagrams and video. Chapter selection and
inspection pause playback. Resuming restores the scripted visual state.
Short audio files are buffered to support seeking on static servers without
byte-range support. Generation tokens cancel outdated asynchronous starts.

## Sources

- `static/js/tour-math.mjs`: kinematics, bounded corrections, evidence, correlation and time mapping.
- `static/js/tour-composition.mjs`: eleven scene compositions and integrated media positions.
- `static/js/tour-robot.mjs`: URDF articulation, mesh rendering and link-bound elliptical Gaussian samples.
- `static/js/tour-player.mjs`: audio, seeking, media and interaction.
- `static/data/explainer-timeline.json`: recorded timing, captions and evidence identity.
- `static/css/explainer.css`, `tour-stage.css`, `tour-fit.css`: presentation controls and stage layout.
- `static/models/panda/`: original Panda joint hierarchy and detailed link meshes.
- `static/vendor/`: locally bundled Three.js and URDFLoader with licenses.
- `static/icons/`: Lucide icons, with their ISC license.

Checks: `node --test tests/explainer-model.test.mjs` and
`node scripts/validate_explainer.mjs`. Serve `docs/` with any static HTTP server.

## Authoring lineage

The presentation follows the reference explainer workflow: deterministic state,
shared audio clock, inspectable transformations and recorded evidence windows.
Its method computations and scene implementation are specific to KineSync-GS.
The robot view uses Three.js and URDFLoader. The reference robot webpage uses
MuJoCo in the browser; the shared design lesson is to animate actual articulated
mesh geometry with joint state and a consistent camera. This presentation uses
the smaller visual-only URDF path. No model service or inference backend is required.

The Gaussian view is a deterministic surface-sample illustration with a Gaussian
opacity kernel and two oriented tangent axes. Samples are children of their
respective link visuals, so centers and axes follow the same URDF transformations.
It is not a learned reconstruction. Original experimental Gaussian results remain
in the recorded media. Mesh, Gaussians and Overlay modes allow direct inspection;
dragging rotates the view, and image selection enlarges the original result.

The presentation has no evidence sidebar or under-canvas microcopy. All media
use contain sizing, and recorded videos retain their original speed. Method
results are shown as fixed reported values, including the readable online table.

## Presentation refinement (4.2)

Spatial scenes now distinguish the published robot from a translucent gold
proposal. Fitting leaves the published state unchanged; cross-view assessment
then permits a whole-state commit or keeps the measurement. A single seekable
phase function drives the vectors, decision and articulated models. Display
easing connects the before/after poses; it does not introduce an intermediate
update rule. The motivation scene retains a measured-pose reference so that
fitting-induced displacement is visible in geometry as well as the recorded plot.

Temporal scenes connect corresponding motion events across the two streams,
reveal the correlation search, and bring the events into alignment when the
offset is applied. These signals are the existing deterministic teaching example.
Mesh overlays use restrained Gaussian opacity to retain link contours, and
both robot views use identical surface samples. Recorded-media borders and
chapter-entry transitions follow the existing palette; no experimental outcomes,
recordings or narration timings have changed.

Validation includes publication order, fallback invariance, representation
switching, every caption at desktop/mobile widths, full narrated playback and
pixel-level checks for articulated motion and display modes.
