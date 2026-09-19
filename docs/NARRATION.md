# KineSync-GS Narration

The complete presentation runs for 174.192 seconds at the original recording speed. All eleven chapters have unique recordings. Audio is the master clock.

## Timing

The third recording has been replaced with the supplied corrected core-principle narration. Sentence starts at 0.00, 3.42 and 7.36 seconds follow its measured speech. Its duration is 16.296 seconds. The other recordings retain their established timing.

## Timeline

| Chapter | Audio | Duration | Script |
|---|---|---:|---|
| 01 · State, not appearance alone | 01.mp3 | 15.600 s | KineSync-GS addresses a missing link in Real2Sim2Real. Gaussian Splatting can reproduce a robot visually, but a useful digital twin must also represent the physical robot's articulated state at the correct time. |
| 02 · Observed failure mode | 02.mp3 | 15.984 s | The difficulty appears when cameras are delayed or views disagree. Direct inverse rendering can keep lowering image error while joint error rises. A visually improved estimate therefore cannot receive automatic authority to modify the twin. |
| 03 · Proposal–evidence contract | 03.mp3 | 16.296 s | KineSync-GS separates two decisions. Rendering or motion first proposes a bounded correction. Cross-view or motion evidence then determines whether the proposal is committed as a verified update or the incoming measurement is preserved. |
| 04 · Spatial synchronization | 04.mp3 | 16.728 s | For spatial synchronization, fixed kinematics bind Gaussians to robot links and inverse rendering estimates one coordinated joint correction. View-specific estimates provide evidence for the complete proposal, which is committed atomically or rejected as a whole. |
| 05 · Temporal synchronization | 05.mp3 | 16.512 s | The temporal path instantiates the same contract. Image and telemetry motion propose a residual lag after native-delay correction. Timing evidence then selects corrected resampling or preserves the original camera-to-state association. |
| 06 · Controlled Franka | 06.mp3 | 17.136 s | On twelve held-out Franka poses and forty-eight view-offset conditions, KineSync-GS reduces whole-state joint error from zero point five nine zero to zero point three two two degrees. It retains all twelve helpful proposals and makes no harmful update. |
| 07 · Mechanism ablations | 07.mp3 | 17.376 s | The mechanism ablations explain this selectivity. Factorized calibration raises dual-view recovery from forty-two point eight six to eighty point nine five percent, while component re-verification retains eighty-eight point eight nine percent of successful proposals. |
| 08 · Real-world PiPER | 08.mp3 | 15.360 s | Real-world PiPER recordings evaluate the same interface under physical motion and matched head and wrist cameras. Cross-view evidence aligns the rendered gripper while the verified state remains available throughout corn-to-plate execution. |
| 09 · Complete online chain | 09.mp3 | 16.440 s | In the complete online chain, spatial and temporal evidence are complementary. KineSync-GS reaches three point zero seven milliradians qMAE with eighty-three point zero five percent recovery and only zero point three six percent harmful updates. |
| 10 · Same-policy manipulation | 10.mp3 | 16.512 s | Most importantly, synchronization improves downstream manipulation without changing the policy. Across two hundred forty paired real-robot episodes, task-macro success rises from fifty-eight point three three to seventy-one point six seven percent. |
| 11 · Reusable interface | 11.mp3 | 10.248 s | KineSync-GS turns visual evidence into a verified state-changing interface across Gaussian twins, simulators, and downstream robot policies. |

## Visual evidence

The interactive Panda uses the original articulated URDF chain and detailed link meshes. Gaussian surface samples are a deterministic teaching model attached to those link frames, not a learned reconstruction or a new experiment. Vectors and correlation signals illustrate the method computation. Experimental media and aggregate numbers come from the existing paper results. Recordings retain their original playback speed; the presentation shows excerpts and holds the final frame when a clip ends.

The stage integrates recorded media with method transformations. Images can be enlarged. Captions are centered and highlighted; no peripheral evidence labels or under-canvas notes are displayed.
