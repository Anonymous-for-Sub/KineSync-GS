export const COLORS = {
  seal: "#6b2717",
  gold: "#cc9e4c",
  dun: "#e0d0b6",
  cadet: "#8b9ea5",
  bistre: "#442c1b",
  success: "#3f7658",
  danger: "#b14945",
  measurement: "#708c98",
  paper: "#ffffff",
  soft: "#f8f5f1",
  line: "#dfd5cc",
};

export const clamp = (value, min = 0, max = 1) => Math.min(max, Math.max(min, value));
export const lerp = (a, b, t) => a + (b - a) * t;
export const ease = (value) => { const t = clamp(value); return t * t * (3 - 2 * t); };
export const phase = (progress, start, end) => ease((progress - start) / (end - start));

export function interpolatePose(a, b, progress) {
  return a.map((value, index) => lerp(value, b[index], ease(progress)));
}

export function armPoints(pose, base = [180, 430], lengths = [94, 84, 72, 58]) {
  const points = [base];
  let angle = -Math.PI / 2;
  for (let index = 0; index < lengths.length; index += 1) {
    angle += pose[index] || 0;
    const previous = points[points.length - 1];
    points.push([
      previous[0] + Math.cos(angle) * lengths[index],
      previous[1] + Math.sin(angle) * lengths[index],
    ]);
  }
  return points;
}

export function gaussianPoints(points, density = 5) {
  const splats = [];
  for (let segment = 0; segment < points.length - 1; segment += 1) {
    const [x0, y0] = points[segment];
    const [x1, y1] = points[segment + 1];
    for (let index = 0; index < density; index += 1) {
      const t = (index + 0.5) / density;
      const wobble = Math.sin((segment + 1) * 7.1 + index * 2.3) * 5;
      const dx = x1 - x0, dy = y1 - y0;
      const length = Math.hypot(dx, dy) || 1;
      splats.push([
        lerp(x0, x1, t) - (dy / length) * wobble,
        lerp(y0, y1, t) + (dx / length) * wobble,
        4 + ((segment + index) % 3),
      ]);
    }
  }
  return splats;
}

export function sampleCurve(fn, x, y, width, height, count = 50) {
  return Array.from({ length: count }, (_, index) => {
    const t = index / (count - 1);
    return [x + t * width, y + (1 - clamp(fn(t), -0.05, 1.05)) * height];
  });
}

export function pathFromPoints(points) {
  return points.map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(2)} ${y.toFixed(2)}`).join(" ");
}

export function motionSignal(t, offset = 0) {
  const shifted = t + offset;
  return 0.5 + 0.28 * Math.sin(shifted * Math.PI * 4.2) + 0.14 * Math.sin(shifted * Math.PI * 9.4 + 0.7);
}

export function evidenceScores(progress) {
  const t = ease(progress);
  return {
    visualGain: lerp(0.08, 0.86, t),
    gradient: lerp(0.12, 0.78, t),
    direction: lerp(0.05, 0.82, t),
    agreement: lerp(0.18, 0.91, t),
  };
}

const measuredPose = [-0.54, 0.72, -0.48, 0.34];
const proposalPose = [-0.68, 0.91, -0.58, 0.46];
const stalePose = [-0.91, 0.42, -0.12, 0.58];

export const POSES = { measuredPose, proposalPose, stalePose };

export function sceneState(id, progress) {
  const p = clamp(progress);
  const common = { step: clamp(Math.floor(p * 4), 0, 3), decision: "Following the synchronization contract", readouts: [] };
  if (id === "real2sim2real") return { ...common, decision: "Binding observations to q(t)", readouts: [["State age", `${Math.round(48 - 34 * p)} ms`], ["Twin clock", `t + ${(p * 0.8).toFixed(2)} s`]] };
  if (id === "failure") return { ...common, step: p < .55 ? 1 : 2, decision: p < .55 ? "Fitting a stale observation" : "State error is diverging", readouts: [["Visual residual", (1 - .67 * p).toFixed(2)], ["Joint error", `${(0.31 + 4.763 * p).toFixed(2)}°`]] };
  if (id === "contract") return { ...common, decision: p < .68 ? "Evaluating separate evidence" : p < .88 ? "Committing supported proposal" : "Preserving the measurement", readouts: [["Proposal bound", "0.18 rad"], ["Gate", p < .68 ? "pending" : p < .88 ? "accept" : "fallback"]] };
  if (id === "spatial") {
    const scores = evidenceScores(p);
    return { ...common, decision: p < .72 ? "Evaluating cross-view evidence" : "Committing the whole-state proposal", readouts: [["Visual gain", scores.visualGain.toFixed(2)], ["Agreement", scores.agreement.toFixed(2)]] };
  }
  if (id === "temporal") return { ...common, decision: p < .73 ? "Searching residual lag" : "Publishing verified resampling", readouts: [["Residual lag", `${Math.round(56 * (1 - ease(p)))} ms`], ["Peak support", `${Math.round(54 + 43 * ease(p))}%`]] };
  if (id === "controlled") return { ...common, step: 3, decision: "Useful proposals retained; harmful updates rejected", readouts: [["Whole-state qMAE", `${lerp(.590, .322, ease(p)).toFixed(3)}°`], ["Harmful", "0 / 48"]] };
  if (id === "mechanism") return { ...common, step: 3, decision: "Factorized evidence sharpens acceptance", readouts: [["Dual-view", `${lerp(42.86, 80.95, ease(p)).toFixed(2)}%`], ["Retained", `${lerp(61.11, 88.89, ease(p)).toFixed(2)}%`]] };
  if (id === "real-world") return { ...common, decision: p < .68 ? "Recovering PiPER state" : "Publishing synchronized state", readouts: [["qMAE", `${lerp(84.06, 2.26, ease(p)).toFixed(2)} mrad`], ["Views", "head + wrist"]] };
  if (id === "online") return { ...common, step: 3, decision: "Temporal and spatial evidence combined", readouts: [["qMAE", `${lerp(11.67, 3.07, ease(p)).toFixed(2)} mrad`], ["Harmful", `${lerp(6.68, .36, ease(p)).toFixed(2)}%`]] };
  if (id === "policy") return { ...common, step: 3, decision: "Same policy consumes synchronized state", readouts: [["Raw state", "58.33%"], ["KineSync state", `${lerp(58.33, 71.67, ease(p)).toFixed(2)}%`]] };
  return { ...common, step: 3, decision: "One synchronized state contract", readouts: [["Robot stacks", "3"], ["Published state", "qˢ(t)"]] };
}
