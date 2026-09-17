import { clamp } from "./explainer-model.mjs";
import { renderScene, sceneState } from "./explainer-scenes.mjs";

const manifestUrl = "static/data/explainer-timeline.json";
const state = {
  manifest: null,
  time: 0,
  playing: false,
  sceneIndex: -1,
  lastFrame: 0,
  lastUrlSecond: -1,
};

const el = {};
const ids = [
  "explainer", "scene-kicker", "scene-title", "scene-count", "mode-label",
  "chapter-list", "animation-stage", "animation-description", "evidence-window",
  "evidence-label", "evidence-counter", "evidence-media", "evidence-kind",
  "scene-metric", "caption-en", "inspector-title", "inspector-detail",
  "detail-formula", "inspector-flow", "tracked-object", "decision-readout",
  "live-readouts", "previous", "play", "next", "current-time", "timeline",
  "timeline-markers", "audio", "fullscreen", "restart", "demo-shell", "state-packet",
];

const sceneFlows = {
  real2sim2real: ["Observe physical execution", "Associate image, state, and time", "Drive an articulated Gaussian twin"],
  failure: ["Fit a delayed observation", "Reduce the visual residual", "Expose the rising state error"],
  contract: ["Measure the incoming state", "Propose a bounded correction", "Evaluate separate evidence", "Commit or preserve"],
  spatial: ["Bind Gaussians to kinematic links", "Recover fused and view-specific states", "Verify the complete proposal"],
  temporal: ["Extract image and joint motion", "Estimate residual lag", "Verify and resample the state stream"],
  controlled: ["Perturb held-out views", "Recover coordinated corrections", "Audit every state-changing decision"],
  mechanism: ["Factor shared and view-specific error", "Calibrate cross-view evidence", "Re-verify successful components"],
  "real-world": ["Record matched physical views", "Recover the Gaussian state", "Publish a task-ready state stream"],
  online: ["Associate asynchronous measurements", "Verify temporal and spatial proposals", "Publish one synchronized state"],
  policy: ["Hold the policy fixed", "Change only its state input", "Measure paired task success"],
  closing: ["Adapt robot-specific observations", "Share one verified state bus", "Serve twins, simulators, and policies"],
};

const trackedLabels = {
  real2sim2real: "Physical state q(t)",
  failure: "Fitted state q+",
  contract: "Measured state qᵐ",
  spatial: "Whole-state proposal q+",
  temporal: "Camera–telemetry offset Δt",
  controlled: "Verified update set",
  mechanism: "Cross-view evidence eˢᵖ",
  "real-world": "PiPER state stream",
  online: "Timestamped state qˢ(t)",
  policy: "Policy state input",
  closing: "Synchronized state qˢ(t)",
};

const evidenceKinds = {
  real2sim2real: "Animated state flow",
  failure: "Measured failure mode",
  contract: "Update contract",
  spatial: "Kinematic recovery",
  temporal: "Motion alignment",
  controlled: "Held-out evaluation",
  mechanism: "Matched ablation",
  "real-world": "Physical replay",
  online: "Online chain",
  policy: "Same-policy test",
  closing: "Cross-stack interface",
};

function cacheElements() {
  ids.forEach((id) => { el[id] = document.getElementById(id); });
  el.railNodes = Array.from(document.querySelectorAll(".rail-node"));
}

function formatTime(value) {
  const seconds = Math.max(0, Math.round(value));
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

function sceneAt(time) {
  const index = state.manifest.scenes.findIndex((scene) => time >= scene.start && time < scene.end);
  return index === -1 ? state.manifest.scenes.length - 1 : index;
}

function buildChapters() {
  el["chapter-list"].innerHTML = "";
  el["timeline-markers"].innerHTML = "";
  state.manifest.scenes.forEach((scene, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chapter-button";
    button.dataset.scene = String(index);
    button.innerHTML = `<span>${String(index + 1).padStart(2, "0")}</span><b>${scene.shortTitle}</b>`;
    button.addEventListener("click", () => seek(scene.start + 0.01));
    el["chapter-list"].appendChild(button);
    if (index > 0) {
      const marker = document.createElement("i");
      marker.style.left = `${(scene.start / state.manifest.duration) * 100}%`;
      el["timeline-markers"].appendChild(marker);
    }
  });
}

function createEvidence(scene) {
  const evidence = scene.evidence;
  el["evidence-media"].innerHTML = "";
  if (!evidence) {
    el["evidence-window"].hidden = true;
    el["evidence-window"].removeAttribute("data-position");
    return;
  }
  el["evidence-window"].hidden = false;
  el["evidence-window"].dataset.position = evidence.position || "top-right";
  el["evidence-label"].textContent = evidence.label;
  const isVideo = /\.mp4(?:$|\?)/i.test(evidence.media);
  el["evidence-counter"].textContent = isVideo ? "RECORDED VIDEO" : "MEASURED RESULT";
  const media = document.createElement(isVideo ? "video" : "img");
  media.src = evidence.media;
  if (isVideo) {
    media.muted = true;
    media.loop = true;
    media.playsInline = true;
    media.preload = "metadata";
    if (evidence.poster) media.poster = evidence.poster;
    media.addEventListener("loadedmetadata", () => syncEvidence(scene));
  } else {
    media.alt = evidence.label;
  }
  el["evidence-media"].appendChild(media);
}

function syncEvidence(scene) {
  const media = el["evidence-media"].querySelector("video");
  if (!media) return;
  const local = clamp(state.time - scene.start, 0, scene.end - scene.start);
  if (Number.isFinite(media.duration) && media.duration > .1) {
    const target = local % Math.max(media.duration - .08, .1);
    if (Math.abs(media.currentTime - target) > .45) media.currentTime = target;
  }
  if (state.playing) media.play().catch(() => {});
  else media.pause();
}

function renderFlow(scene) {
  const items = sceneFlows[scene.id] || sceneFlows.contract;
  el["inspector-flow"].innerHTML = items.map((item) => `<div class="flow-row">${item}</div>`).join("");
}

function activateScene(index) {
  state.sceneIndex = index;
  const scene = state.manifest.scenes[index];
  el["scene-kicker"].textContent = scene.kicker;
  el["scene-title"].textContent = scene.title;
  el["scene-count"].textContent = `${String(index + 1).padStart(2, "0")} / ${String(state.manifest.scenes.length).padStart(2, "0")}`;
  el["scene-metric"].textContent = scene.metric;
  el["evidence-kind"].textContent = evidenceKinds[scene.id] || "Method animation";
  el["caption-en"].textContent = scene.caption;
  el["inspector-title"].textContent = scene.shortTitle;
  el["inspector-detail"].textContent = scene.detail;
  el["detail-formula"].textContent = scene.formula;
  el["animation-description"].textContent = `${scene.title}. ${scene.detail}`;
  el["tracked-object"].textContent = trackedLabels[scene.id] || "Synchronized state";
  createEvidence(scene);
  renderFlow(scene);
  document.querySelectorAll(".chapter-button").forEach((button, chapterIndex) => {
    button.classList.toggle("active", chapterIndex === index);
    button.setAttribute("aria-current", chapterIndex === index ? "step" : "false");
  });
}

function updateRail(step) {
  el.railNodes.forEach((node, index) => node.classList.toggle("active", index <= step));
  el["state-packet"].style.left = `${9 + step * 27.1}%`;
}

function updateInspector(scene, progress) {
  const live = sceneState(scene.id, progress);
  updateRail(live.step);
  el["decision-readout"].textContent = live.decision;
  el["live-readouts"].innerHTML = live.readouts.map(([label, value]) => `<div class="live-readout"><span>${label}</span><b>${value}</b></div>`).join("");
}

function renderAt(time, forceScene = false) {
  state.time = clamp(time, 0, state.manifest.duration);
  const index = sceneAt(state.time);
  if (index !== state.sceneIndex || forceScene) activateScene(index);
  const scene = state.manifest.scenes[index];
  const progress = clamp((state.time - scene.start) / (scene.end - scene.start));

  el["animation-stage"].innerHTML = renderScene(scene.id, progress);
  updateInspector(scene, progress);
  syncEvidence(scene);
  el.timeline.value = String(state.time);
  el["current-time"].textContent = formatTime(state.time);

  const queryTime = Math.floor(state.time);
  if (history.replaceState && queryTime % 5 === 0 && queryTime !== state.lastUrlSecond) {
    history.replaceState(null, "", `${location.pathname}?t=${queryTime}`);
    state.lastUrlSecond = queryTime;
  }
}

function seek(time) {
  renderAt(time);
  state.lastFrame = performance.now();
}

function updatePlayButton() {
  el.play.innerHTML = state.playing ? '<span aria-hidden="true">Ⅱ</span><b>Pause</b>' : '<span aria-hidden="true">▶</span><b>Play tour</b>';
  el.play.setAttribute("aria-label", state.playing ? "Pause tour" : "Play tour");
  el["mode-label"].textContent = state.playing ? "Playing silent tour" : "Silent tour";
}

function setPlaying(playing) {
  if (playing && state.time >= state.manifest.duration - .01) seek(0);
  state.playing = playing;
  state.lastFrame = performance.now();
  updatePlayButton();
  const scene = state.manifest.scenes[state.sceneIndex];
  if (scene) syncEvidence(scene);
}

function tick(timestamp) {
  if (state.playing) {
    const delta = clamp((timestamp - state.lastFrame) / 1000, 0, .1);
    const next = state.time + delta;
    if (next >= state.manifest.duration) {
      renderAt(state.manifest.duration);
      setPlaying(false);
    } else renderAt(next);
  }
  state.lastFrame = timestamp;
  requestAnimationFrame(tick);
}

function changeChapter(direction) {
  const next = clamp(state.sceneIndex + direction, 0, state.manifest.scenes.length - 1);
  seek(state.manifest.scenes[next].start + .01);
}

function bindEvents() {
  el.play.addEventListener("click", () => setPlaying(!state.playing));
  el.previous.addEventListener("click", () => changeChapter(-1));
  el.next.addEventListener("click", () => changeChapter(1));
  el.restart.addEventListener("click", () => { setPlaying(false); seek(0); });
  el.timeline.addEventListener("input", (event) => seek(Number(event.target.value)));
  el.fullscreen.addEventListener("click", () => {
    if (!document.fullscreenElement) el["demo-shell"].requestFullscreen?.();
    else document.exitFullscreen?.();
  });
  document.addEventListener("visibilitychange", () => { if (document.hidden && state.playing) setPlaying(false); });
  document.addEventListener("keydown", (event) => {
    if (event.code === "Space" && !["INPUT", "BUTTON"].includes(document.activeElement?.tagName)) {
      event.preventDefault(); setPlaying(!state.playing);
    }
    if (event.code === "ArrowLeft") seek(state.time - 5);
    if (event.code === "ArrowRight") seek(state.time + 5);
  });
}

async function init() {
  cacheElements();
  try {
    const response = await fetch(manifestUrl);
    if (!response.ok) throw new Error(`Timeline request failed: ${response.status}`);
    state.manifest = await response.json();
    el.timeline.max = String(state.manifest.duration);
    buildChapters();
    bindEvents();
    const requested = Number(new URLSearchParams(location.search).get("t"));
    renderAt(Number.isFinite(requested) ? requested : 0, true);
    updatePlayButton();
    el.explainer.setAttribute("aria-busy", "false");
    requestAnimationFrame((timestamp) => { state.lastFrame = timestamp; requestAnimationFrame(tick); });
  } catch (error) {
    el["scene-title"].textContent = "The interactive timeline could not be loaded";
    el["inspector-detail"].textContent = error.message;
    el.explainer.setAttribute("aria-busy", "false");
  }
}

init();
