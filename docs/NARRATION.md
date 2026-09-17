# KineSync-GS Interactive Animation Narration

This document is the recording handoff for the 150-second interactive animation at `explainer.html`. The visual tour is complete and currently runs without narration. Scene boundaries will be aligned to the measured audio duration after the final recording is supplied; no synthetic narration or estimated waveform timing is used.

## Audio handoff

- Preferred master filename: `static/audio/narration/kinesync-gs-narration.wav`
- Accepted delivery: lossless WAV, mono or stereo, 44.1/48 kHz.
- Keep the supplied master unchanged. A web delivery copy may be generated next to it after timing verification.
- The scene manifest currently records `timingSource: visual-plan-pre-audio` and `audio: null`. These fields change only after the real file is measured and synchronized.

## Timed subtitles

| Time | Scene | English subtitle |
|---|---|---|
| 00:00–00:14 | State, not appearance alone | KineSync-GS addresses a missing link in Real2Sim2Real. Gaussian Splatting can reproduce a robot visually, but a useful digital twin must also represent the physical robot's articulated state at the correct time. |
| 00:14–00:29 | Observed failure mode | The difficulty appears when cameras are delayed or views disagree. Direct inverse rendering can keep lowering image error while joint error rises. A visually improved estimate therefore cannot receive automatic authority to modify the twin. |
| 00:29–00:43 | Proposal–evidence contract | KineSync-GS separates two decisions. Rendering or motion first proposes a bounded correction. Cross-view or motion evidence then determines whether the proposal is committed as a verified update or the incoming measurement is preserved. |
| 00:43–01:00 | Spatial synchronization | For spatial synchronization, fixed kinematics bind Gaussians to robot links and inverse rendering estimates one coordinated joint correction. View-specific estimates provide evidence for the complete proposal, which is committed atomically or rejected as a whole. |
| 01:00–01:14 | Temporal synchronization | The temporal path instantiates the same contract. Image and telemetry motion propose a residual lag after native-delay correction. Timing evidence then selects corrected resampling or preserves the original camera-to-state association. |
| 01:14–01:29 | Controlled Franka | On twelve held-out Franka poses and forty-eight view-offset conditions, KineSync-GS reduces whole-state joint error from zero point five nine zero to zero point three two two degrees. It retains all twelve helpful proposals and makes no harmful update. |
| 01:29–01:43 | Mechanism ablations | The mechanism ablations explain this selectivity. Factorized calibration raises dual-view recovery from forty-two point eight six to eighty point nine five percent, while component re-verification retains eighty-eight point eight nine percent of successful proposals. |
| 01:43–01:58 | Real-world PiPER | Real-world PiPER recordings evaluate the same interface under physical motion and matched head and wrist cameras. Cross-view evidence aligns the rendered gripper while the verified state remains available throughout corn-to-plate execution. |
| 01:58–02:12 | Complete online chain | In the complete online chain, spatial and temporal evidence are complementary. KineSync-GS reaches three point zero seven milliradians qMAE with eighty-three point zero five percent recovery and only zero point three six percent harmful updates. |
| 02:12–02:23 | Same-policy manipulation | Most importantly, synchronization improves downstream manipulation without changing the policy. Across two hundred forty paired real-robot episodes, task-macro success rises from fifty-eight point three three to seventy-one point six seven percent. |
| 02:23–02:30 | Reusable interface | KineSync-GS turns visual evidence into a verified state-changing interface across Gaussian twins, simulators, and downstream robot policies. |

## Clean recording script

KineSync-GS addresses a missing link in Real2Sim2Real. Gaussian Splatting can reproduce a robot visually, but a useful digital twin must also represent the physical robot's articulated state at the correct time.

The difficulty appears when cameras are delayed or views disagree. Direct inverse rendering can keep lowering image error while joint error rises. A visually improved estimate therefore cannot receive automatic authority to modify the twin.

KineSync-GS separates two decisions. Rendering or motion first proposes a bounded correction. Cross-view or motion evidence then determines whether the proposal is committed as a verified update or the incoming measurement is preserved.

For spatial synchronization, fixed kinematics bind Gaussians to robot links and inverse rendering estimates one coordinated joint correction. View-specific estimates provide evidence for the complete proposal, which is committed atomically or rejected as a whole.

The temporal path instantiates the same contract. Image and telemetry motion propose a residual lag after native-delay correction. Timing evidence then selects corrected resampling or preserves the original camera-to-state association.

On twelve held-out Franka poses and forty-eight view-offset conditions, KineSync-GS reduces whole-state joint error from zero point five nine zero to zero point three two two degrees. It retains all twelve helpful proposals and makes no harmful update.

The mechanism ablations explain this selectivity. Factorized calibration raises dual-view recovery from forty-two point eight six to eighty point nine five percent, while component re-verification retains eighty-eight point eight nine percent of successful proposals.

Real-world PiPER recordings evaluate the same interface under physical motion and matched head and wrist cameras. Cross-view evidence aligns the rendered gripper while the verified state remains available throughout corn-to-plate execution.

In the complete online chain, spatial and temporal evidence are complementary. KineSync-GS reaches three point zero seven milliradians qMAE with eighty-three point zero five percent recovery and only zero point three six percent harmful updates.

Most importantly, synchronization improves downstream manipulation without changing the policy. Across two hundred forty paired real-robot episodes, task-macro success rises from fifty-eight point three three to seventy-one point six seven percent.

KineSync-GS turns visual evidence into a verified state-changing interface across Gaussian twins, simulators, and downstream robot policies.
