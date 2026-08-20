const elements = {
  datasetSelect: document.querySelector("#datasetSelect"),
  methodASelect: document.querySelector("#methodASelect"),
  methodBSelect: document.querySelector("#methodBSelect"),
  swapMethods: document.querySelector("#swapMethods"),
  playPause: document.querySelector("#playPause"),
  previousFrame: document.querySelector("#previousFrame"),
  nextFrame: document.querySelector("#nextFrame"),
  timeline: document.querySelector("#timeline"),
  timecode: document.querySelector("#timecode"),
  framecode: document.querySelector("#framecode"),
  speedControl: document.querySelector("#speedControl"),
  datasetMeta: document.querySelector("#datasetMeta"),
  health: document.querySelector("#health"),
  healthText: document.querySelector("#healthText"),
  matrix: document.querySelector("#videoMatrix"),
  cellTemplate: document.querySelector("#videoCellTemplate"),
  methodARowLabel: document.querySelector("#methodARowLabel"),
  methodBRowLabel: document.querySelector("#methodBRowLabel"),
  legendTicks: document.querySelector("#legendTicks"),
  depthLegend: document.querySelector("#depthLegend"),
  attentionLegend: document.querySelector("#attentionLegend"),
  mobileCameraTabs: document.querySelector("#mobileCameraTabs"),
};

const state = {
  catalog: null,
  dataset: null,
  methodA: "lingbot_v05_sensor_fused",
  methodB: "ai_consensus_fused",
  playbackRate: 1,
  playing: false,
  scrubbing: false,
  resumeAfterScrub: false,
  currentFrame: 0,
  generation: 0,
  animationFrame: 0,
  starting: false,
  resyncing: false,
  driftFrames: 0,
  videos: new Map(),
};

const ROWS = ["rgb", "method-a", "method-b"];

function setHealth(status, text) {
  elements.health.dataset.state = status;
  elements.healthText.textContent = text;
}

function methodById(id) {
  return state.catalog.methods.find((method) => method.id === id);
}

function formatTime(seconds) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const minutes = Math.floor(safe / 60);
  const remainder = safe - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(3).padStart(6, "0")}`;
}

function updateReadouts(frame = state.currentFrame) {
  if (!state.dataset || !state.catalog) return;
  const totalFrames = state.dataset.frameCounts.cam_h;
  const clamped = Math.max(0, Math.min(totalFrames - 1, Math.round(frame)));
  state.currentFrame = clamped;
  elements.timeline.value = String(clamped);
  const progress = totalFrames > 1 ? (clamped / (totalFrames - 1)) * 100 : 0;
  elements.timeline.style.setProperty("--progress", `${progress}%`);
  const currentSeconds = clamped / state.catalog.fps;
  const durationSeconds = totalFrames / state.catalog.fps;
  elements.timecode.textContent = `${formatTime(currentSeconds)} / ${formatTime(durationSeconds)}`;
  elements.framecode.textContent = `帧 ${clamped.toLocaleString()} / ${(totalFrames - 1).toLocaleString()}`;
}

function createVideoCells() {
  const fragment = document.createDocumentFragment();
  for (const row of ROWS) {
    const cell = elements.cellTemplate.content.firstElementChild.cloneNode(true);
    cell.classList.add("video-strip");
    cell.dataset.row = row;
    cell.style.gridColumn = "2 / 5";
    cell.style.gridRow = String(ROWS.indexOf(row) + 2);
    cell.querySelector(".cell-camera").textContent = "头部 · 左腕 · 右腕";
    const video = cell.querySelector("video");
    video.dataset.row = row;
    video.disablePictureInPicture = true;
    video.controls = false;
    video.playbackRate = state.playbackRate;
    state.videos.set(row, video);
    fragment.appendChild(cell);
  }
  elements.matrix.appendChild(fragment);
}

function createMobileCameraTabs() {
  for (const camera of state.catalog.cameras) {
    const button = document.createElement("button");
    button.type = "button";
    button.role = "tab";
    button.dataset.camera = camera.id;
    button.textContent = camera.label;
    button.classList.toggle("is-active", camera.id === "cam_h");
    button.setAttribute("aria-selected", camera.id === "cam_h" ? "true" : "false");
    button.addEventListener("click", () => {
      document.body.dataset.mobileCamera = camera.id;
      elements.mobileCameraTabs.querySelectorAll("button").forEach((candidate) => {
        const active = candidate.dataset.camera === camera.id;
        candidate.classList.toggle("is-active", active);
        candidate.setAttribute("aria-selected", active ? "true" : "false");
      });
    });
    elements.mobileCameraTabs.appendChild(button);
  }
}

function populateSelect(select, items, value, labelBuilder) {
  select.replaceChildren();
  for (const item of items) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = labelBuilder(item);
    option.selected = item.id === value;
    select.appendChild(option);
  }
}

function videoSource(row) {
  const media = state.dataset.mosaic;
  if (row === "rgb") return media.rgb;
  const methodId = row === "method-a" ? state.methodA : state.methodB;
  return media.methods[methodId];
}

function waitForVideo(video, generation) {
  return new Promise((resolve, reject) => {
    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      resolve();
      return;
    }
    const timeout = window.setTimeout(() => finish(new Error("视频载入超时")), 20000);
    const finish = (error) => {
      window.clearTimeout(timeout);
      video.removeEventListener("loadeddata", onLoaded);
      video.removeEventListener("error", onError);
      if (generation !== state.generation) {
        resolve();
      } else if (error) {
        reject(error);
      } else {
        resolve();
      }
    };
    const onLoaded = () => finish();
    const onError = () => finish(new Error(video.error?.message || "视频无法载入"));
    video.addEventListener("loadeddata", onLoaded, { once: true });
    video.addEventListener("error", onError, { once: true });
  });
}

async function loadRows(rows, keepFrame = state.currentFrame) {
  const generation = ++state.generation;
  const wasPlaying = state.playing;
  pauseAll();
  setHealth("loading", "载入视频");
  const pending = [];

  for (const row of rows) {
    const video = state.videos.get(row);
    const cell = video.closest(".video-cell");
    const stateLabel = cell.querySelector(".cell-state");
    cell.classList.remove("is-ready", "is-error");
    stateLabel.textContent = "载入中";
    video.src = videoSource(row);
    video.load();
    pending.push(
      waitForVideo(video, generation)
        .then(() => {
          if (generation !== state.generation) return;
          cell.classList.add("is-ready");
          stateLabel.textContent = `${state.catalog.fps.toFixed(0)} FPS · 3 视角`;
        })
        .catch((error) => {
          cell.classList.add("is-error");
          stateLabel.textContent = "载入失败";
          throw error;
        }),
    );
  }

  const results = await Promise.allSettled(pending);
  if (generation !== state.generation) return;
  const failed = results.filter((result) => result.status === "rejected").length;
  seekAll(keepFrame, false);
  if (failed) {
    setHealth("error", `${failed} 路异常`);
  } else {
    setHealth("ready", "9 路同步");
    if (wasPlaying) await playAll();
  }
}

function updateDatasetMeta() {
  const counts = state.catalog.cameras
    .map((camera) => `${camera.label} ${state.dataset.frameCounts[camera.id].toLocaleString()} 帧`)
    .join(" · ");
  elements.datasetMeta.textContent = `${state.dataset.label} · ${counts} · ${formatTime(state.dataset.durationSeconds)}`;
}

async function selectDataset(datasetId) {
  state.dataset = state.catalog.datasets.find((dataset) => dataset.id === datasetId);
  state.currentFrame = 0;
  const totalFrames = state.dataset.frameCounts.cam_h;
  elements.timeline.max = String(Math.max(0, totalFrames - 1));
  updateDatasetMeta();
  updateReadouts(0);
  await loadRows(ROWS, 0);
}

function updateMethodLabels() {
  const methodA = methodById(state.methodA);
  const methodB = methodById(state.methodB);
  elements.methodARowLabel.textContent = methodA.label;
  elements.methodBRowLabel.textContent = methodB.label;
  const visualizations = [methodA.visualization || "depth", methodB.visualization || "depth"];
  elements.depthLegend.hidden = !visualizations.includes("depth");
  elements.attentionLegend.hidden = !visualizations.includes("attention");
}

function allVideos() {
  return [...state.videos.values()];
}

function masterVideo() {
  return state.videos.get("rgb");
}

function waitUntil(video, predicate, eventNames, generation, timeoutMs = 12000) {
  if (predicate()) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => finish(new Error("媒体同步超时")), timeoutMs);
    const generationCheck = window.setInterval(() => {
      if (generation !== state.generation) finish();
    }, 100);
    const finish = (error) => {
      window.clearTimeout(timeout);
      window.clearInterval(generationCheck);
      eventNames.forEach((eventName) => video.removeEventListener(eventName, onEvent));
      video.removeEventListener("error", onError);
      if (generation !== state.generation || !error) resolve();
      else reject(error);
    };
    const onEvent = () => {
      if (predicate()) finish();
    };
    const onError = () => finish(new Error(video.error?.message || "媒体同步失败"));
    eventNames.forEach((eventName) => video.addEventListener(eventName, onEvent));
    video.addEventListener("error", onError, { once: true });
  });
}

async function alignVideo(video, targetTime, generation) {
  await waitUntil(
    video,
    () => video.readyState >= HTMLMediaElement.HAVE_METADATA,
    ["loadedmetadata"],
    generation,
  );
  if (generation !== state.generation) return;
  const safeTarget = Math.min(targetTime, Math.max(0, video.duration - 0.001));
  const tolerance = 0.5 / state.catalog.fps;
  if (video.seeking || Math.abs(video.currentTime - safeTarget) > tolerance) {
    const seeked = waitUntil(
      video,
      () => !video.seeking && Math.abs(video.currentTime - safeTarget) <= tolerance,
      ["seeked", "timeupdate"],
      generation,
    );
    if (!video.seeking || Math.abs(video.currentTime - safeTarget) > tolerance) {
      video.currentTime = safeTarget;
    }
    await seeked;
  }
  await waitUntil(
    video,
    () => video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA,
    ["loadeddata", "canplay", "seeked"],
    generation,
  );
  video.playbackRate = state.playbackRate;
}

async function alignAll(targetTime, generation = state.generation) {
  await Promise.all(allVideos().map((video) => alignVideo(video, targetTime, generation)));
}

function setPlayButton(playing) {
  state.playing = playing;
  elements.playPause.classList.toggle("is-playing", playing);
  elements.playPause.title = playing ? "暂停" : "播放";
  elements.playPause.setAttribute("aria-label", playing ? "暂停" : "播放");
  elements.playPause.innerHTML = playing
    ? '<i data-lucide="pause"></i>'
    : '<i data-lucide="play"></i>';
  window.lucide?.createIcons({ attrs: { "stroke-width": 1.8 } });
}

async function playAll() {
  if (!state.dataset || state.starting || state.resyncing) return;
  state.starting = true;
  const generation = state.generation;
  const lastFrame = state.dataset.frameCounts.cam_h - 1;
  if (state.currentFrame >= lastFrame) seekAll(0, false);
  const targetTime = state.currentFrame / state.catalog.fps;
  setHealth("loading", "同步定位");
  try {
    await alignAll(targetTime, generation);
  } catch (error) {
    console.error(error);
    pauseAll();
    setHealth("error", "同步失败");
    state.starting = false;
    return;
  }
  if (generation !== state.generation) {
    state.starting = false;
    return;
  }
  const playable = allVideos();
  const results = await Promise.allSettled(playable.map((video) => video.play()));
  const masterResult = results[playable.indexOf(masterVideo())];
  if (!masterResult || masterResult.status === "rejected") {
    pauseAll();
    setHealth("error", "主视频无法播放");
    state.starting = false;
    return;
  }
  if (results.some((result) => result.status === "rejected")) {
    setHealth("error", "播放受限");
  } else {
    setHealth("ready", "9 路同步");
  }
  setPlayButton(true);
  state.starting = false;
  startAnimationLoop();
}

function pauseAll() {
  allVideos().forEach((video) => video.pause());
  setPlayButton(false);
  state.starting = false;
  state.driftFrames = 0;
  if (state.animationFrame) {
    cancelAnimationFrame(state.animationFrame);
    state.animationFrame = 0;
  }
}

function seekAll(frame, update = true) {
  if (!state.dataset) return;
  const totalFrames = state.dataset.frameCounts.cam_h;
  const clamped = Math.max(0, Math.min(totalFrames - 1, Math.round(frame)));
  const targetTime = clamped / state.catalog.fps;
  for (const video of allVideos()) {
    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
      video.currentTime = Math.min(targetTime, Math.max(0, video.duration - 0.001));
    }
  }
  if (update) updateReadouts(clamped);
  else state.currentFrame = clamped;
}

function synchronizeFollowers(masterTime) {
  const master = masterVideo();
  let maxDrift = 0;
  for (const video of allVideos()) {
    if (video === master || video.readyState < HTMLMediaElement.HAVE_METADATA) continue;
    const drift = masterTime - video.currentTime;
    maxDrift = Math.max(maxDrift, Math.abs(drift));
    if (Math.abs(drift) > 0.02) {
      const correction = Math.max(-0.2, Math.min(0.2, drift * 1.4));
      video.playbackRate = Math.max(0.25, state.playbackRate + correction);
    } else if (video.playbackRate !== state.playbackRate) {
      video.playbackRate = state.playbackRate;
    }
  }
  return { maxDrift };
}

async function resyncPlayback(targetTime) {
  if (state.resyncing || !state.playing) return;
  state.resyncing = true;
  const generation = state.generation;
  allVideos().forEach((video) => video.pause());
  setHealth("loading", "缓冲同步");
  try {
    await alignAll(targetTime, generation);
    if (generation !== state.generation || !state.playing) return;
    const results = await Promise.allSettled(allVideos().map((video) => video.play()));
    if (results.some((result) => result.status === "rejected")) {
      pauseAll();
      setHealth("error", "重新同步失败");
    } else {
      setHealth("ready", "9 路同步");
    }
  } catch (error) {
    console.error(error);
    pauseAll();
    setHealth("error", "重新同步失败");
  } finally {
    state.resyncing = false;
  }
}

function startAnimationLoop() {
  if (state.animationFrame) cancelAnimationFrame(state.animationFrame);
  const tick = () => {
    const master = masterVideo();
    if (!state.playing || !master) return;
    const frame = Math.min(
      state.dataset.frameCounts.cam_h - 1,
      Math.round(master.currentTime * state.catalog.fps),
    );
    updateReadouts(frame);
    if (!state.resyncing) {
      const playbackHealth = synchronizeFollowers(master.currentTime);
      state.driftFrames = playbackHealth.maxDrift > 0.14 ? state.driftFrames + 1 : 0;
      if (state.driftFrames >= 6) {
        state.driftFrames = 0;
        void resyncPlayback(master.currentTime);
      }
    }
    if (master.ended || frame >= state.dataset.frameCounts.cam_h - 1) {
      pauseAll();
      return;
    }
    state.animationFrame = requestAnimationFrame(tick);
  };
  state.animationFrame = requestAnimationFrame(tick);
}

function bindEvents() {
  elements.datasetSelect.addEventListener("change", () => selectDataset(elements.datasetSelect.value));
  elements.methodASelect.addEventListener("change", async () => {
    state.methodA = elements.methodASelect.value;
    updateMethodLabels();
    await loadRows(["method-a"]);
  });
  elements.methodBSelect.addEventListener("change", async () => {
    state.methodB = elements.methodBSelect.value;
    updateMethodLabels();
    await loadRows(["method-b"]);
  });
  elements.swapMethods.addEventListener("click", async () => {
    [state.methodA, state.methodB] = [state.methodB, state.methodA];
    elements.methodASelect.value = state.methodA;
    elements.methodBSelect.value = state.methodB;
    updateMethodLabels();
    await loadRows(["method-a", "method-b"]);
  });
  elements.playPause.addEventListener("click", () => (state.playing ? pauseAll() : playAll()));
  elements.previousFrame.addEventListener("click", () => {
    pauseAll();
    seekAll(state.currentFrame - 1);
  });
  elements.nextFrame.addEventListener("click", () => {
    pauseAll();
    seekAll(state.currentFrame + 1);
  });

  elements.timeline.addEventListener("pointerdown", () => {
    state.scrubbing = true;
    state.resumeAfterScrub = state.playing;
    pauseAll();
  });
  elements.timeline.addEventListener("input", () => seekAll(Number(elements.timeline.value)));
  const finishScrub = async () => {
    if (!state.scrubbing) return;
    state.scrubbing = false;
    if (state.resumeAfterScrub) await playAll();
    state.resumeAfterScrub = false;
  };
  elements.timeline.addEventListener("change", finishScrub);
  elements.timeline.addEventListener("pointerup", finishScrub);

  elements.speedControl.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-speed]");
    if (!button) return;
    state.playbackRate = Number(button.dataset.speed);
    allVideos().forEach((video) => {
      video.playbackRate = state.playbackRate;
    });
    elements.speedControl.querySelectorAll("button").forEach((candidate) => {
      candidate.classList.toggle("is-active", candidate === button);
    });
  });

  document.addEventListener("keydown", (event) => {
    if (["SELECT", "INPUT", "BUTTON"].includes(document.activeElement?.tagName)) return;
    if (event.code === "Space") {
      event.preventDefault();
      state.playing ? pauseAll() : playAll();
    } else if (event.code === "ArrowLeft") {
      event.preventDefault();
      pauseAll();
      seekAll(state.currentFrame - 1);
    } else if (event.code === "ArrowRight") {
      event.preventDefault();
      pauseAll();
      seekAll(state.currentFrame + 1);
    }
  });
}

async function initialize() {
  try {
    const response = await fetch("data/catalog.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`目录请求失败: HTTP ${response.status}`);
    state.catalog = await response.json();
    if (!state.catalog.datasets?.length) throw new Error("数据目录为空");

    if (!state.catalog.methods.some((method) => method.id === state.methodA)) {
      state.methodA = state.catalog.methods[0].id;
    }
    if (!state.catalog.methods.some((method) => method.id === state.methodB)) {
      state.methodB = state.catalog.methods[Math.min(1, state.catalog.methods.length - 1)].id;
    }

    populateSelect(
      elements.datasetSelect,
      state.catalog.datasets,
      state.catalog.datasets[0].id,
      (dataset) => `${dataset.label} · ${dataset.group}`,
    );
    populateSelect(elements.methodASelect, state.catalog.methods, state.methodA, (method) => method.label);
    populateSelect(elements.methodBSelect, state.catalog.methods, state.methodB, (method) => method.label);
    const [minimum, maximum] = state.catalog.depthRangeM;
    elements.legendTicks.textContent = `${minimum.toFixed(1)} m — ${maximum.toFixed(1)} m`;
    createVideoCells();
    createMobileCameraTabs();
    updateMethodLabels();
    bindEvents();
    window.lucide?.createIcons({ attrs: { "stroke-width": 1.8 } });
    await selectDataset(state.catalog.datasets[0].id);
  } catch (error) {
    console.error(error);
    elements.datasetMeta.textContent = error.message;
    setHealth("error", "载入失败");
  }
}

initialize();
