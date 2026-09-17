(() => {
  "use strict";

  const manifestUrl = "static/data/explainer-timeline.json";
  const state = {
    manifest: null,
    time: 0,
    playing: false,
    sceneIndex: -1,
    lastFrame: 0,
    captionMode: "both",
    lastUrlSecond: -1,
    reducedMotion: window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  };

  const el = {};
  const ids = [
    "explainer", "scene-kicker", "scene-title", "scene-count", "mode-label",
    "chapter-list", "media-frame", "computed-layer", "evidence-kind",
    "scene-metric", "caption-en", "caption-zh", "inspector-title",
    "inspector-detail", "inspector-flow", "tracked-object", "decision-readout",
    "previous", "play", "next", "current-time", "timeline",
    "timeline-markers", "language", "audio", "fullscreen", "restart",
    "demo-shell", "state-packet",
  ];

  const sceneFlows = {
    opening: ["Observe the physical task", "Bind appearance to articulated state", "Publish a synchronized twin"],
    failure: ["Fit the stale image", "Observe lower visual residual", "Detect rising state error"],
    pipeline: ["Measure the incoming state", "Propose a bounded correction", "Evaluate separate evidence", "Commit or preserve"],
    spatial: ["Bind Gaussians to kinematic links", "Recover fused and view-specific states", "Verify one whole-state proposal"],
    temporal: ["Extract image and joint motion", "Propose residual camera lag", "Verify and resample the state stream"],
    result: ["Aggregate held-out evaluation", "Compare matched update rules", "Report state and decision quality"],
    real: ["Record matched head and wrist views", "Recover Gaussian state", "Publish the verified task state"],
    closing: ["Reuse one state contract", "Adapt robot-specific backends", "Serve simulators and policies"],
  };

  const evidenceLabels = {
    opening: "Recorded execution",
    failure: "Observed failure",
    pipeline: "Method contract",
    spatial: "Cross-view evidence",
    temporal: "Motion evidence",
    result: "Measured aggregate",
    real: "Real-world evidence",
    closing: "External stack",
  };

  const trackedLabels = {
    opening: "Physical state q(t)",
    failure: "Fitted state q⁺",
    pipeline: "Measured state qᵐ",
    spatial: "Whole-state proposal q⁺",
    temporal: "Camera–telemetry offset Δt",
    result: "Verified update set",
    real: "PiPER state stream",
    closing: "Synchronized state qˢ",
  };

  function cacheElements() {
    ids.forEach((id) => { el[id] = document.getElementById(id); });
    el.captionPanel = document.querySelector(".caption-panel");
    el.railNodes = Array.from(document.querySelectorAll(".rail-node"));
  }

  function formatTime(value) {
    const seconds = Math.max(0, Math.round(value));
    return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function sceneAt(time) {
    const scenes = state.manifest.scenes;
    const index = scenes.findIndex((scene) => time >= scene.start && time < scene.end);
    return index === -1 ? scenes.length - 1 : index;
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

  function createMedia(scene) {
    el["media-frame"].className = `media-frame ${scene.secondary ? "dual" : "single"}`;
    el["media-frame"].innerHTML = "";

    const appendMedia = (src, poster, secondary = false) => {
      const isVideo = /\.mp4(?:$|\?)/i.test(src);
      const media = document.createElement(isVideo ? "video" : "img");
      media.className = secondary ? "secondary-media" : "primary-media";
      if (isVideo) {
        media.src = src;
        media.muted = true;
        media.loop = true;
        media.playsInline = true;
        media.preload = "metadata";
        if (poster) media.poster = poster;
        media.addEventListener("loadedmetadata", () => syncSceneVideos(scene));
      } else {
        media.src = src;
        media.alt = secondary ? `${scene.shortTitle} supporting view` : scene.title;
      }
      el["media-frame"].appendChild(media);
    };

    appendMedia(scene.media, scene.poster);
    if (scene.secondary) appendMedia(scene.secondary, null, true);
  }

  function overlayMarkup(scene) {
    switch (scene.type) {
      case "failure":
        return `<div class="failure-bars">
          <div><label><span>Visual residual</span><b id="loss-value">1.00</b></label><span class="bar-track"><i id="loss-bar" class="loss-bar"></i></span></div>
          <div><label><span>Joint-state error</span><b id="error-value">0.31°</b></label><span class="bar-track"><i id="error-bar" class="error-bar"></i></span></div>
        </div>`;
      case "pipeline":
        return `<div class="pipeline-overlay">
          <div class="pipeline-box">Measured<br>state</div><div class="pipeline-box">Bounded<br>proposal</div><div class="pipeline-box">Separate<br>evidence</div><div class="pipeline-box">Verified state<br>or fallback</div>
        </div>`;
      case "spatial":
        return `<div class="evidence-gauges">
          ${[["Visual gain", "r"], ["Gradient support", "g"], ["Direction support", "a"], ["State agreement", "d"]].map(([name, symbol]) => `<div class="gauge"><span>${name}</span><span class="gauge-track"><i data-gauge="${symbol}"></i></span><b>${symbol}</b></div>`).join("")}
        </div>`;
      case "temporal":
        return `<div class="lag-overlay"><span>Residual lag proposal</span><div class="lag-axis"><i id="lag-marker" class="lag-marker"></i></div></div>`;
      default:
        return "";
    }
  }

  function renderFlow(scene) {
    const items = sceneFlows[scene.type] || sceneFlows.result;
    el["inspector-flow"].innerHTML = items.map((item) => `<div class="flow-row">${item}</div>`).join("");
  }

  function activateScene(index) {
    state.sceneIndex = index;
    const scene = state.manifest.scenes[index];
    el["scene-kicker"].textContent = scene.kicker;
    el["scene-title"].textContent = scene.title;
    el["scene-count"].textContent = `${String(index + 1).padStart(2, "0")} / ${String(state.manifest.scenes.length).padStart(2, "0")}`;
    el["scene-metric"].textContent = scene.metric;
    el["evidence-kind"].textContent = evidenceLabels[scene.type] || "Evaluation evidence";
    el["caption-en"].textContent = scene.caption.en;
    el["caption-zh"].textContent = scene.caption.zh;
    el["inspector-title"].textContent = scene.shortTitle;
    el["inspector-detail"].textContent = scene.detail;
    el["tracked-object"].textContent = trackedLabels[scene.type] || "Synchronized state";
    createMedia(scene);
    el["computed-layer"].innerHTML = overlayMarkup(scene);
    renderFlow(scene);
    document.querySelectorAll(".chapter-button").forEach((button, chapterIndex) => {
      button.classList.toggle("active", chapterIndex === index);
      button.setAttribute("aria-current", chapterIndex === index ? "step" : "false");
    });
  }

  function syncSceneVideos(scene) {
    const local = clamp(state.time - scene.start, 0, scene.end - scene.start);
    el["media-frame"].querySelectorAll("video").forEach((video) => {
      if (Number.isFinite(video.duration) && video.duration > 0) {
        const target = local % Math.max(video.duration - 0.08, 0.1);
        if (Math.abs(video.currentTime - target) > 0.45) video.currentTime = target;
      }
      if (state.playing) video.play().catch(() => {});
      else video.pause();
    });
  }

  function renderFailure(progress) {
    const loss = 1 - 0.67 * progress;
    const error = 0.31 + 4.763 * progress;
    const lossBar = document.getElementById("loss-bar");
    const errorBar = document.getElementById("error-bar");
    if (!lossBar || !errorBar) return;
    lossBar.style.transform = `scaleX(${loss})`;
    errorBar.style.transform = `scaleX(${0.1 + 0.9 * progress})`;
    document.getElementById("loss-value").textContent = loss.toFixed(2);
    document.getElementById("error-value").textContent = `${error.toFixed(2)}°`;
  }

  function renderPipeline(progress) {
    const active = clamp(Math.floor(progress * 4), 0, 3);
    document.querySelectorAll(".pipeline-box").forEach((box, index) => box.classList.toggle("active", index <= active));
  }

  function renderSpatial(progress) {
    const offsets = { r: 0.03, g: 0.12, a: 0.22, d: 0.32 };
    document.querySelectorAll("[data-gauge]").forEach((gauge) => {
      const fill = clamp((progress - offsets[gauge.dataset.gauge]) / 0.58, 0.06, 1);
      gauge.style.transform = `scaleX(${fill})`;
    });
  }

  function renderTemporal(progress) {
    const marker = document.getElementById("lag-marker");
    if (marker) marker.style.left = `${75 - 25 * clamp(progress / 0.78, 0, 1)}%`;
  }

  function updateRail(scene, progress) {
    let step = 0;
    if (scene.type === "failure") step = Math.min(1, Math.floor(progress * 2));
    else if (["result", "real", "closing"].includes(scene.type)) step = 3;
    else step = clamp(Math.floor(progress * 4), 0, 3);
    el.railNodes.forEach((node, index) => node.classList.toggle("active", index <= step));
    el["state-packet"].style.left = `${9 + step * 27.1}%`;

    let decision = "Collecting observations";
    if (scene.type === "failure") decision = progress > 0.55 ? "Visual fit is not state authority" : "Fitting the observation";
    else if (scene.type === "pipeline") decision = progress > 0.72 ? "Verified update or fallback" : "Following the update contract";
    else if (scene.type === "spatial") decision = progress > 0.72 ? "Commit supported whole state" : "Evaluating cross-view evidence";
    else if (scene.type === "temporal") decision = progress > 0.72 ? "Publish verified resampling" : "Evaluating motion evidence";
    else if (scene.type === "result") decision = "Measured verified-update behavior";
    else if (scene.type === "real") decision = progress > 0.6 ? "Publish synchronized PiPER state" : "Recovering physical motion";
    else if (scene.type === "closing") decision = "One reusable state interface";
    el["decision-readout"].textContent = decision;
  }

  function renderAt(time, forceScene = false) {
    state.time = clamp(time, 0, state.manifest.duration);
    const index = sceneAt(state.time);
    if (index !== state.sceneIndex || forceScene) activateScene(index);
    const scene = state.manifest.scenes[index];
    const progress = clamp((state.time - scene.start) / (scene.end - scene.start), 0, 1);

    if (scene.type === "failure") renderFailure(progress);
    if (scene.type === "pipeline") renderPipeline(progress);
    if (scene.type === "spatial") renderSpatial(progress);
    if (scene.type === "temporal") renderTemporal(progress);
    updateRail(scene, progress);
    syncSceneVideos(scene);

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
    if (playing && state.time >= state.manifest.duration - 0.01) seek(0);
    state.playing = playing;
    state.lastFrame = performance.now();
    updatePlayButton();
    const scene = state.manifest.scenes[state.sceneIndex];
    if (scene) syncSceneVideos(scene);
  }

  function tick(timestamp) {
    if (state.playing) {
      const delta = clamp((timestamp - state.lastFrame) / 1000, 0, 0.1);
      const next = state.time + delta;
      if (next >= state.manifest.duration) {
        renderAt(state.manifest.duration);
        setPlaying(false);
      } else {
        renderAt(next);
      }
    }
    state.lastFrame = timestamp;
    requestAnimationFrame(tick);
  }

  function changeChapter(direction) {
    const next = clamp(state.sceneIndex + direction, 0, state.manifest.scenes.length - 1);
    seek(state.manifest.scenes[next].start + 0.01);
  }

  function cycleLanguage() {
    const modes = ["both", "en", "zh"];
    state.captionMode = modes[(modes.indexOf(state.captionMode) + 1) % modes.length];
    el.captionPanel.classList.toggle("en-only", state.captionMode === "en");
    el.captionPanel.classList.toggle("zh-only", state.captionMode === "zh");
    el.language.textContent = state.captionMode === "both" ? "EN + 中文" : state.captionMode === "en" ? "English" : "中文";
  }

  function bindEvents() {
    el.play.addEventListener("click", () => setPlaying(!state.playing));
    el.previous.addEventListener("click", () => changeChapter(-1));
    el.next.addEventListener("click", () => changeChapter(1));
    el.restart.addEventListener("click", () => { setPlaying(false); seek(0); });
    el.timeline.addEventListener("input", (event) => seek(Number(event.target.value)));
    el.language.addEventListener("click", cycleLanguage);
    el.fullscreen.addEventListener("click", () => {
      if (!document.fullscreenElement) el["demo-shell"].requestFullscreen?.();
      else document.exitFullscreen?.();
    });
    document.addEventListener("visibilitychange", () => {
      if (document.hidden && state.playing) setPlaying(false);
    });
    document.addEventListener("keydown", (event) => {
      if (["INPUT", "BUTTON"].includes(document.activeElement?.tagName) && event.code === "Space") return;
      if (event.code === "Space") { event.preventDefault(); setPlaying(!state.playing); }
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
})();
