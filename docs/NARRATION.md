# KineSync-GS Interactive Demonstration Narration

This document is the recording handoff for the 150-second interactive project demonstration at `explainer.html`. The current timeline is a complete silent visual tour. Scene boundaries will be aligned to the measured audio duration after the final recording is supplied; no synthetic narration or estimated waveform timing is used.

## Audio handoff

- Preferred master filename: `static/audio/narration/kinesync-gs-narration.wav`
- Accepted delivery: lossless WAV, mono or stereo, 44.1/48 kHz.
- Keep the supplied master unchanged. A web delivery copy may be generated next to it after timing verification.
- The scene manifest currently records `timingSource: visual-plan-pre-audio` and `audio: null`. These fields must change only after the real file is measured and synchronized.

## Timed bilingual subtitles

| Time | Scene | English subtitle | 中文字幕 |
|---|---|---|---|
| 00:00–00:14 | The synchronization problem | KineSync-GS addresses a missing link in Real2Sim2Real. Gaussian Splatting can reproduce a robot visually, but a useful digital twin must also represent the physical robot's articulated state at the correct time. | KineSync-GS 关注 Real2Sim2Real 中一个容易被忽略的环节。Gaussian Splatting 可以逼真重现机器人外观，但有用的数字孪生还必须在正确时刻表示真实机器人的关节状态。 |
| 00:14–00:29 | Observed failure mode | The difficulty appears when cameras are delayed or views disagree. Direct inverse rendering can keep lowering image error while joint error rises. A visually improved estimate therefore cannot receive automatic authority to modify the twin. | 当相机存在延迟或多个视角互相冲突时，问题就会出现。直接逆渲染可能持续降低图像误差，却让关节误差不断上升。因此，视觉上更好的估计不能自动获得修改数字孪生的权限。 |
| 00:29–00:43 | Proposal–evidence contract | KineSync-GS separates two decisions. Rendering or motion first proposes a bounded correction. Cross-view or motion evidence then determines whether the proposal is committed as a verified update or the incoming measurement is preserved. | KineSync-GS 将两个决定分开。渲染或运动首先提出受约束的修正，再由跨视角或运动证据决定提交验证更新，还是保留输入测量。 |
| 00:43–01:00 | Spatial path | For spatial synchronization, fixed kinematics bind Gaussians to robot links and inverse rendering estimates one coordinated joint correction. View-specific estimates provide evidence for the complete proposal, which is committed atomically or rejected as a whole. | 在空间同步中，固定运动学将高斯绑定到机械臂连杆，逆渲染估计一个协调一致的关节修正。各视角估计为完整提议提供证据，使其整体提交或整体拒绝。 |
| 01:00–01:14 | Temporal path | The temporal path instantiates the same contract. Image and telemetry motion propose a residual lag after native-delay correction. Timing evidence then selects corrected resampling or preserves the original camera-to-state association. | 时间路径遵循同一契约。图像与遥测运动在校正固有延迟后提出剩余时差，再由时间证据选择校正重采样，或保留原始相机与状态关联。 |
| 01:14–01:29 | Controlled Franka results | On twelve held-out Franka poses and forty-eight view-offset conditions, KineSync-GS reduces whole-state joint error from zero point five nine zero to zero point three two two degrees. It retains all twelve helpful proposals and makes no harmful update. | 在十二个未见 Franka 姿态和四十八个视角偏移条件上，KineSync-GS 将全状态关节误差从 0.590 度降至 0.322 度，保留全部十二个有益提议，并且没有产生有害更新。 |
| 01:29–01:43 | Mechanism ablations | The mechanism ablations explain this selectivity. Factorized calibration raises dual-view recovery from forty-two point eight six to eighty point nine five percent, while component re-verification retains eighty-eight point eight nine percent of successful proposals. | 机制消融解释了这种选择性。因子化标定将双视角恢复率从 42.86% 提升到 80.95%，组件重验证则保留了 88.89% 的成功提议。 |
| 01:43–01:58 | Real-world PiPER | Real-world PiPER recordings evaluate the same interface under physical motion and matched head and wrist cameras. Cross-view evidence aligns the rendered gripper while the verified state remains available throughout corn-to-plate execution. | 真实 PiPER 记录在物理运动以及匹配的头部和腕部相机下评估同一接口。跨视角证据对齐渲染夹爪，并在玉米放盘任务全过程中持续发布验证状态。 |
| 01:58–02:12 | Complete online chain | In the complete online chain, spatial and temporal evidence are complementary. KineSync-GS reaches three point zero seven milliradians qMAE with eighty-three point zero five percent recovery and only zero point three six percent harmful updates. | 在完整在线链路中，空间与时间证据形成互补。KineSync-GS 达到 3.07 毫弧度 qMAE、83.05% 恢复率，同时有害更新仅为 0.36%。 |
| 02:12–02:23 | Same-policy manipulation | Most importantly, synchronization improves downstream manipulation without changing the policy. Across two hundred forty paired real-robot episodes, task-macro success rises from fifty-eight point three three to seventy-one point six seven percent. | 更重要的是，同步在不改变策略的情况下改善了下游操控。在二百四十个配对真机 episode 上，任务宏平均成功率从 58.33% 提升到 71.67%。 |
| 02:23–02:30 | Reusable interface | KineSync-GS turns visual evidence into a verified state-changing interface across Gaussian twins, simulators, and downstream robot policies. | KineSync-GS 将视觉证据转化为经过验证的状态修改接口，连接 Gaussian 数字孪生、仿真器与下游机器人策略。 |

## Clean English recording script

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

## 中文录音净稿

我们这个项目叫 KineSync-GS，研究的是机器人数字孪生中的状态同步问题。先从 Real2Sim2Real 这个背景讲起：我们把真实机器人和环境搬到仿真中进行训练、评估或数据生成，再将结果用于真实机器人。3D Gaussian Splatting 已经能让数字孪生看起来非常真实，但数字孪生不能只是长得像，它还必须在正确时刻表示真实机器人的关节状态。

当相机存在延迟，或者多个相机看到的并不是同一时刻，直接让渲染器拟合图像可能会找到一个看起来更像、却不是真实状态的机械臂姿态。我们的实验确实观察到：图像误差持续下降时，关节状态误差反而可能上升。

因此，KineSync-GS 的核心不是再做一个更强的渲染器，而是把状态估计与修改状态的权限分开。渲染或运动首先提出一个受约束的修正，跨视角或运动证据再决定这个修正是否可以作为验证更新提交；证据不足时，系统保留输入测量。

在空间同步路径中，固定运动学把高斯绑定到机器人连杆，逆渲染估计一个协调一致的全关节修正。各个视角单独恢复的状态提供跨视角证据，使完整提议被整体提交或整体拒绝。

时间同步路径采用同一套契约。图像运动与遥测运动在校正固有延迟后提出剩余时差，再由时间证据决定是否进行校正重采样，或者保留原始的相机与状态关联。

在十二个未见 Franka 姿态和四十八个视角偏移条件上，KineSync-GS 将全状态关节误差从 0.590 度降至 0.322 度，保留全部十二个有益提议，同时没有产生有害更新。

机制消融进一步说明证据如何改善更新决策。因子化标定将双视角恢复率从 42.86% 提升到 80.95%，组件重验证保留了 88.89% 的成功提议。

在真实 PiPER 数据上，匹配的头部和腕部相机为物理运动提供跨视角证据。系统在玉米放盘任务全过程中对齐渲染夹爪，并持续发布验证后的任务状态。

在完整在线链路中，空间与时间证据形成互补。KineSync-GS 达到 3.07 毫弧度的关节误差、83.05% 的恢复率，同时有害更新仅为 0.36%。

更重要的是，同步后的状态在不修改策略的情况下改善了下游操控。在二百四十个配对真机实验中，任务宏平均成功率从 58.33% 提升到 71.67%。

KineSync-GS 最终提供的是一个统一的验证状态接口：由渲染或运动提出修正，由证据决定是否改变数字孪生，并将同步后的状态提供给仿真器和下游机器人策略。
