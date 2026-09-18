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
deterministic teaching examples. Recorded insets preserve their source frames
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
keyframes. The supplied recordings are published unchanged. Chapter 03 is a
silent interlude pending a corrected recording: the supplied 03 was identical
to 02. It does not replay the wrong narration.

Audio time drives chapter progress and captions. During the silent interlude a
monotonic presentation clock advances the same visual state. Mute changes only
audibility; pause stops narration, diagrams and video. Chapter selection and
inspection pause playback. Resuming restores the scripted visual state.
Short audio files are buffered to support seeking on static servers without
byte-range support. Generation tokens cancel outdated asynchronous starts.

## Sources

- `static/js/tour-math.mjs`: kinematics, bounded corrections, evidence, correlation and time mapping.
- `static/js/tour-scenes.mjs`: deterministic scene drawing.
- `static/js/tour-player.mjs`: audio, seeking, media and interaction.
- `static/data/explainer-timeline.json`: recorded timing, captions and evidence identity.
- `static/css/explainer.css`: responsive presentation layout.
- `static/icons/`: Lucide icons, with their ISC license.

Checks: `node --test tests/explainer-model.test.mjs` and
`node scripts/validate_explainer.mjs`. Serve `docs/` with any static HTTP server.

## Authoring lineage

The presentation follows the reference explainer workflow: deterministic state,
shared audio clock, inspectable transformations and recorded evidence windows.
Its method computations and scene implementation are specific to KineSync-GS.
No external model service or GPU is required by the webpage.
