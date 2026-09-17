import test from "node:test";
import assert from "node:assert/strict";
import {
  POSES, armPoints, evidenceScores, interpolatePose, motionSignal, sceneState,
} from "../docs/static/js/explainer-model.mjs";
import { renderScene } from "../docs/static/js/explainer-scenes.mjs";

test("forward kinematics returns one point per link endpoint", () => {
  const points = armPoints(POSES.measuredPose);
  assert.equal(points.length, 5);
  for (const point of points) assert.equal(point.length, 2);
});

test("pose interpolation preserves exact endpoints", () => {
  assert.deepEqual(interpolatePose(POSES.measuredPose, POSES.proposalPose, 0), POSES.measuredPose);
  const end = interpolatePose(POSES.measuredPose, POSES.proposalPose, 1);
  end.forEach((value, index) => assert.ok(Math.abs(value - POSES.proposalPose[index]) < 1e-12));
});

test("cross-view evidence strengthens over the animation", () => {
  const start = evidenceScores(0);
  const end = evidenceScores(1);
  for (const key of Object.keys(start)) assert.ok(end[key] > start[key]);
});

test("motion signal is deterministic and bounded", () => {
  const a = motionSignal(.37, .08);
  const b = motionSignal(.37, .08);
  assert.equal(a, b);
  assert.ok(a >= 0 && a <= 1);
});

test("every scene reconstructs SVG at start, middle, and end", () => {
  const ids = ["real2sim2real", "failure", "contract", "spatial", "temporal", "controlled", "mechanism", "real-world", "online", "policy", "closing"];
  for (const id of ids) {
    const frames = [];
    for (const progress of [0, .5, 1]) {
      const markup = renderScene(id, progress);
      frames.push(markup);
      assert.match(markup, /<defs>/);
      assert.ok(markup.length > 1000, `${id} at ${progress} produced insufficient scene markup`);
      const live = sceneState(id, progress);
      assert.ok(live.step >= 0 && live.step <= 3);
      assert.ok(live.decision.length > 0);
    }
    assert.notEqual(frames[0], frames[1], `${id} does not animate between start and middle`);
    assert.notEqual(frames[1], frames[2], `${id} does not animate between middle and end`);
  }
});
