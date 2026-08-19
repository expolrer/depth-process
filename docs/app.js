const initialView = new URLSearchParams(window.location.search).get("view");

const state = {
  catalog: null,
  roiMetrics: null,
  depthQualityMetrics: null,
  view: initialView === "attention" ? "attention" : "depth",
  datasetIndex: 0,
  selections: {
    depth: { a: "lingbot_v05", b: "ai_consensus_fused" },
    attention: { a: "lingbot_v05", b: "raw_aligned" },
  },
  playing: false,
  playbackRate: 1,
  generation: 0,
  animationFrame: 0,
};

const $ = (id) => document.getElementById(id);
const videos = () => [$("rgb-video"), $("method-a-video"), $("method-b-video")];
const DEPTH_CAPTION = "Metric depth · 0.2–4.0 m";
const ATTENTION_CAPTION = "ACT Action Query → Depth Token";

function currentDataset() { return state.catalog.datasets[state.datasetIndex]; }
function currentSelection() { return state.selections[state.view]; }
function methodById(id) { return state.catalog.methods.find((method) => method.id === id); }
function availableMethods() {
  return state.view === "depth"
    ? state.catalog.methods.filter((method) => method.family !== "Negative controls")
    : state.catalog.methods;
}

function formatTime(seconds) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const minutes = Math.floor(safe / 60);
  const remainder = safe - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(3).padStart(6, "0")}`;
}

function setLoading(visible, text = "正在加载完整视频") {
  $("loading-text").textContent = text;
  $("loading-state").classList.toggle("hidden", !visible);
}

function optionGroups(select) {
  const methods = availableMethods();
  const families = [...new Set(methods.map((method) => method.family))];
  select.replaceChildren();
  for (const family of families) {
    const group = document.createElement("optgroup");
    group.label = family === "Negative controls" ? "负对照" : "深度处理方法";
    for (const method of methods.filter((item) => item.family === family)) {
      const option = document.createElement("option");
      option.value = method.id;
      option.textContent = method.label;
      group.append(option);
    }
    select.append(group);
  }
}

function refreshMethodControls() {
  optionGroups($("method-a-select"));
  optionGroups($("method-b-select"));
  $("method-a-select").value = currentSelection().a;
  $("method-b-select").value = currentSelection().b;
}

function mediaSources(dataset) {
  const selection = currentSelection();
  const group = state.view === "attention" ? dataset.attention : dataset.depth;
  return [dataset.rgb, group[selection.a], group[selection.b]];
}

function waitForMetadata(video, generation) {
  return new Promise((resolve, reject) => {
    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) return resolve();
    const timeout = window.setTimeout(() => finish(new Error("视频加载超时")), 20000);
    const finish = (error) => {
      window.clearTimeout(timeout);
      video.removeEventListener("loadedmetadata", onLoad);
      video.removeEventListener("error", onError);
      if (generation !== state.generation) resolve();
      else if (error) reject(error);
      else resolve();
    };
    const onLoad = () => finish();
    const onError = () => finish(new Error("视频文件无法加载"));
    video.addEventListener("loadedmetadata", onLoad, { once: true });
    video.addEventListener("error", onError, { once: true });
  });
}

function waitForCurrentFrame(video, generation) {
  return new Promise((resolve, reject) => {
    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && !video.seeking) return resolve();
    const timeout = window.setTimeout(() => finish(new Error("目标视频帧解码超时")), 20000);
    const finish = (error) => {
      window.clearTimeout(timeout);
      video.removeEventListener("loadeddata", onReady);
      video.removeEventListener("seeked", onReady);
      video.removeEventListener("error", onError);
      if (generation !== state.generation) resolve();
      else if (error) reject(error);
      else resolve();
    };
    const onReady = () => {
      if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && !video.seeking) finish();
    };
    const onError = () => finish(new Error("目标视频帧无法解码"));
    video.addEventListener("loadeddata", onReady);
    video.addEventListener("seeked", onReady);
    video.addEventListener("error", onError, { once: true });
  });
}

async function loadVideos(preserveTime = true) {
  const generation = ++state.generation;
  const wasPlaying = state.playing;
  const oldTime = preserveTime ? $("rgb-video").currentTime || 0 : 0;
  state.playing = false;
  cancelAnimationFrame(state.animationFrame);
  setLoading(true, "正在同步三路完整视频");
  const sources = mediaSources(currentDataset());
  videos().forEach((video, index) => {
    video.pause();
    video.src = sources[index];
    video.playbackRate = state.playbackRate;
    video.load();
  });
  try {
    await Promise.all(videos().map((video) => waitForMetadata(video, generation)));
    if (generation !== state.generation) return;
    seekTo(Math.min(oldTime, currentDataset().durationSeconds - 1 / state.catalog.fps));
    await Promise.all(videos().map((video) => waitForCurrentFrame(video, generation)));
    if (generation !== state.generation) return;
    setLoading(false);
    if (wasPlaying) await playVideos();
  } catch (error) {
    if (generation === state.generation) setLoading(true, error.message);
  }
}

function seekTo(seconds) {
  const dataset = currentDataset();
  const safe = Math.max(0, Math.min(seconds, dataset.durationSeconds - 1 / state.catalog.fps));
  videos().forEach((video) => { if (Number.isFinite(video.duration)) video.currentTime = safe; });
  updateReadouts(safe);
}

function updateReadouts(time = $("rgb-video").currentTime || 0) {
  const dataset = currentDataset();
  const frame = Math.min(dataset.frameCount - 1, Math.max(0, Math.round(time * state.catalog.fps)));
  $("timeline").max = dataset.durationSeconds;
  $("timeline").value = time;
  $("time-readout").textContent = `${formatTime(time)} / ${formatTime(dataset.durationSeconds)}`;
  $("frame-readout").textContent = `帧 ${frame.toLocaleString()} / ${(dataset.frameCount - 1).toLocaleString()}`;
  renderCameraRoles(dataset, frame);
}

function synchronize() {
  if (!state.playing) return;
  const master = $("rgb-video");
  for (const video of videos().slice(1)) {
    if (Math.abs(video.currentTime - master.currentTime) > 0.08) video.currentTime = master.currentTime;
  }
  updateReadouts(master.currentTime);
  state.animationFrame = requestAnimationFrame(synchronize);
}

async function playVideos() {
  videos().forEach((video) => { video.playbackRate = state.playbackRate; });
  try {
    await Promise.all(videos().map((video) => video.play()));
    state.playing = true;
    renderPlayIcon();
    synchronize();
  } catch (error) {
    state.playing = false;
    renderPlayIcon();
  }
}

function pauseVideos() {
  state.playing = false;
  videos().forEach((video) => video.pause());
  cancelAnimationFrame(state.animationFrame);
  renderPlayIcon();
  updateReadouts();
}

function togglePlayback() { state.playing ? pauseVideos() : playVideos(); }

function renderPlayIcon() {
  const button = $("play-pause");
  button.innerHTML = `<i data-lucide="${state.playing ? "pause" : "play"}"></i>`;
  button.title = state.playing ? "暂停" : "播放";
  lucide.createIcons({ attrs: { "stroke-width": 1.8 } });
}

function metricText(value) { return Number(value).toFixed(4); }

function percentText(value, digits = 2) {
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function renderProjectMetrics() {
  const report = state.depthQualityMetrics;
  if (!report) {
    $("project-metrics").classList.add("hidden");
    return;
  }

  const { scope, depth_distribution: distribution, methods, negative_controls: controls } = report;
  const raw = methods.find((method) => method.id === "raw_aligned");
  const bestCoverage = methods.reduce((best, method) => method.valid_fraction > best.valid_fraction ? method : best);
  const bestAct = methods.reduce((best, method) => method.act_chunk_mae_rad < best.act_chunk_mae_rad ? method : best);
  const bestRoi = methods.reduce((best, method) => method.roi_attention_lift > best.roi_attention_lift ? method : best);
  const zeroControl = controls.find((control) => control.id === "zero_depth");
  const shuffleControl = controls.find((control) => control.id === "spatially_shuffled_raw");

  $("metrics-updated").textContent = `报告日期 ${report.updated}`;
  $("scope-rgbd-frames").textContent = scope.rgbd_frames.toLocaleString("zh-CN");
  $("scope-streams").textContent = `${scope.datasets} 个数据集 · ${scope.camera_streams} 路相机流`;
  $("scope-pixels").textContent = scope.pixels.toLocaleString("zh-CN");
  $("scope-raw-valid").textContent = percentText(distribution.raw_valid_fraction);
  $("scope-depth-distribution").textContent = `中位 ${distribution.median_m.toFixed(3)} m · P95 ${distribution.p95_m.toFixed(3)} m`;
  $("scope-act-frames").textContent = scope.act_held_out_frames.toLocaleString("zh-CN");
  $("scope-roi-events").textContent = `${scope.roi_events} 次抓取`;
  $("scope-roi-records").textContent = `每方法 ${scope.roi_records_per_method.toLocaleString("zh-CN")} 个执行腕 ROI`;

  const actGain = (raw.act_chunk_mae_rad - bestAct.act_chunk_mae_rad) / raw.act_chunk_mae_rad * 100;
  $("metrics-highlights").innerHTML = `
    <div><i data-lucide="maximize-2"></i><span><strong>最高覆盖 ${percentText(bestCoverage.valid_fraction)}</strong><small>${bestCoverage.label}，新增填充 ${percentText(bestCoverage.filled_fraction)}</small></span></div>
    <div><i data-lucide="circle-gauge"></i><span><strong>最低 ACT MAE ${bestAct.act_chunk_mae_rad.toFixed(4)} rad</strong><small>${bestAct.label}，较原始降低 ${actGain.toFixed(2)}%</small></span></div>
    <div><i data-lucide="focus"></i><span><strong>普通方法最高 ROI lift ${bestRoi.roi_attention_lift.toFixed(3)}</strong><small>${bestRoi.label}；此项仅解释注意力分布</small></span></div>
    <div><i data-lucide="test-tube-2"></i><span><strong>负对照 ACT MAE ${zeroControl.act_chunk_mae_rad.toFixed(4)} / ${shuffleControl.act_chunk_mae_rad.toFixed(4)}</strong><small>全零 / 空间打乱，明显高于正常深度方法</small></span></div>
  `;

  $("depth-metrics-body").innerHTML = methods.map((method) => {
    const isCoverageBest = method.id === bestCoverage.id;
    const isActBest = method.id === bestAct.id;
    const isRoiBest = method.id === bestRoi.id;
    const actDelta = (method.act_chunk_mae_rad - raw.act_chunk_mae_rad) / raw.act_chunk_mae_rad * 100;
    let sensorError;
    if (method.sensor_behavior === "reference") {
      sensorError = `<strong>基准</strong><small>自比较不适用</small>`;
    } else if (method.sensor_behavior === "preserved") {
      sensorError = `<strong class="preserved-value">0（保留）</strong><small>有效像素原样复制</small>`;
    } else {
      sensorError = `<strong>${(method.sensor_mae_m * 100).toFixed(2)} cm</strong><small>RMSE ${(method.sensor_rmse_m * 100).toFixed(2)} cm</small>`;
    }

    let deltaText = `<span class="delta-baseline">基线</span>`;
    if (method.id !== raw.id) {
      const direction = actDelta < 0 ? "↓" : "↑";
      const deltaClass = actDelta < 0 ? "delta-better" : "delta-worse";
      deltaText = `<span class="${deltaClass}">${direction} ${Math.abs(actDelta).toFixed(2)}%</span>`;
    }

    return `
      <tr>
        <td><strong>${method.label}</strong><small>${method.kind}</small></td>
        <td><strong>${percentText(method.valid_fraction)}</strong>${isCoverageBest ? '<span class="best-tag">最高</span>' : ""}</td>
        <td><strong>${percentText(method.filled_fraction)}</strong></td>
        <td>${sensorError}</td>
        <td><strong>${method.act_chunk_mae_rad.toFixed(4)} rad</strong><small>首步 ${method.act_first_step_mae_rad.toFixed(4)}</small>${isActBest ? '<span class="best-tag">最低</span>' : ""}</td>
        <td>${deltaText}</td>
        <td><strong>${method.roi_attention_lift.toFixed(3)}</strong>${isRoiBest ? '<span class="best-tag">最高</span>' : ""}</td>
      </tr>
    `;
  }).join("");

  lucide.createIcons({ attrs: { "stroke-width": 1.8 } });
}

function signedMetric(value, digits = 4) {
  const numeric = Number(value);
  return `${numeric >= 0 ? "+" : ""}${numeric.toFixed(digits)}`;
}

function activeWristAnalysis() {
  return state.roiMetrics?.active_wrist_analysis;
}

function roiMethod(methodId) {
  return activeWristAnalysis()?.by_method.find((row) => row.method === methodId);
}

function roiSequence(methodId, dataset) {
  return activeWristAnalysis()?.by_sequence.find(
    (row) => row.method === methodId && row.sequence === dataset.sequence,
  );
}

function actionMae(methodId) {
  return state.roiMetrics?.by_method.find((row) => row.method === methodId)?.chunk_mae_rad;
}

function activeEventAt(dataset, frame) {
  return activeWristAnalysis()?.event_timeline.find(
    (event) => event.sequence === dataset.sequence && frame >= event.start_frame && frame <= event.end_frame,
  );
}

function renderCameraRoles(dataset, frame) {
  const event = activeEventAt(dataset, frame);
  const leftActive = event?.active_camera === "cam_l";
  const rightActive = event?.active_camera === "cam_r";
  $("cam-l-header").classList.toggle("active-wrist", leftActive);
  $("cam-l-header").classList.toggle("control-wrist", !leftActive);
  $("cam-r-header").classList.toggle("active-wrist", rightActive);
  $("cam-r-header").classList.toggle("control-wrist", !rightActive);
  $("cam-l-role").textContent = leftActive ? "执行腕 · D405" : "非执行腕对照 · D405";
  $("cam-r-role").textContent = rightActive ? "执行腕 · D405" : "非执行腕对照 · D405";
  const activeLabel = leftActive ? "左腕执行" : rightActive ? "右腕执行" : "抓取区间外";
  $("viewer-eyebrow").textContent = `30 FPS · 头部 + 执行腕为主要结果 · ${activeLabel}`;
}

function renderRoiMethod(prefix, methodId, dataset) {
  const method = methodById(methodId);
  const report = roiMethod(methodId);
  const sequence = roiSequence(methodId, dataset);
  if (!report || !sequence) return;
  const metrics = report.event_balanced;
  const delta = report.delta_vs_raw_event_balanced.roi_attention_lift;
  const ci = report.paired_delta_vs_raw_event_ci95.roi_attention_lift;
  $(`roi-${prefix}-name`).textContent = method.label;
  $(`roi-${prefix}-sequence-lift`).textContent = sequence.roi_attention_lift.toFixed(3);
  $(`roi-${prefix}-lift`).textContent = metrics.roi_attention_lift.toFixed(3);
  $(`roi-${prefix}-global-mass`).textContent = `${(metrics.roi_global_attention_mass * 100).toFixed(2)}%`;
  $(`roi-${prefix}-delta`).textContent = signedMetric(delta, 3);
  $(`roi-${prefix}-density`).textContent = metrics.roi_background_density_ratio.toFixed(3);
  $(`roi-${prefix}-hit`).textContent = `${(metrics.top_token_in_roi * 100).toFixed(1)}%`;
  $(`roi-${prefix}-ci`).textContent = `相对原始深度配对 95% CI：${signedMetric(ci[0], 3)} 至 ${signedMetric(ci[1], 3)}`;
}

function significanceAgainstRaw(report) {
  if (report.method === "raw_aligned") return "原始对齐深度是配对比较基线。";
  const [low, high] = report.paired_delta_vs_raw_event_ci95.roi_attention_lift;
  if (low > 0) return `${methodById(report.method).label} 的执行腕 ROI lift 相对原始深度为稳定正向。`;
  if (high < 0) return `${methodById(report.method).label} 的执行腕 ROI lift 相对原始深度为稳定负向。`;
  return `${methodById(report.method).label} 相对原始深度的配对区间跨 0，不能认定稳定提升。`;
}

function renderRoiRanking(selection) {
  const rows = activeWristAnalysis().ranking_by_active_wrist_roi_attention_lift
    .map((methodId) => roiMethod(methodId))
    .filter(Boolean);
  const scaleMax = Math.max(1.5, ...rows.map((row) => row.event_balanced.roi_attention_lift));
  $("roi-ranking").innerHTML = rows.map((row) => {
    const activeClass = row.method === selection.a ? "active-a" : row.method === selection.b ? "active-b" : "";
    const lift = row.event_balanced.roi_attention_lift;
    return `<li class="${activeClass}"><span class="roi-ranking-label">${methodById(row.method).label}</span><span class="roi-ranking-track"><span class="roi-ranking-bar" style="width:${Math.min(100, lift / scaleMax * 100).toFixed(1)}%"></span></span><span class="roi-ranking-value">${lift.toFixed(3)}</span></li>`;
  }).join("");
}

function renderRoiAnalysis(dataset, selection) {
  if (!state.roiMetrics) return;
  const reportA = roiMethod(selection.a);
  const reportB = roiMethod(selection.b);
  if (!reportA || !reportB) return;
  renderRoiMethod("a", selection.a, dataset);
  renderRoiMethod("b", selection.b, dataset);
  renderRoiRanking(selection);

  $("roi-event-count").textContent = activeWristAnalysis().events.toLocaleString();
  $("roi-view-count").textContent = activeWristAnalysis().wrist_view_records_per_method.toLocaleString();
  const liftA = reportA.event_balanced.roi_attention_lift;
  const liftB = reportB.event_balanced.roi_attention_lift;
  const globalMassA = reportA.event_balanced.roi_global_attention_mass;
  const globalMassB = reportB.event_balanced.roi_global_attention_mass;
  const maeA = actionMae(selection.a);
  const maeB = actionMae(selection.b);
  const liftWinner = liftA >= liftB ? methodById(selection.a).label : methodById(selection.b).label;
  const maeWinner = maeA <= maeB ? methodById(selection.a).label : methodById(selection.b).label;
  $("roi-comparison-title").textContent = `${dataset.label} · A/B 综合分析`;
  $("roi-comparison-text").textContent = `${liftWinner} 的执行腕 ROI lift 更高（A ${liftA.toFixed(3)} / B ${liftB.toFixed(3)}）；未重新归一化的全局 ROI mass 为 A ${(globalMassA * 100).toFixed(2)}% / B ${(globalMassB * 100).toFixed(2)}%。${maeWinner} 的留出集动作 MAE 更低（A ${maeA.toFixed(4)} / B ${maeB.toFixed(4)}）。当前数据集执行腕 ROI lift 为 A ${roiSequence(selection.a, dataset).roi_attention_lift.toFixed(3)}、B ${roiSequence(selection.b, dataset).roi_attention_lift.toFixed(3)}。`;
  $("roi-significance").textContent = `${significanceAgainstRaw(reportA)} ${significanceAgainstRaw(reportB)} 全局 ROI mass 以三相机全部注意力为分母，未对执行腕重新归一化；非执行腕仅作显示对照。注意力集中不能单独证明任务成功。`;
}

function renderMetrics(prefix, methodId, dataset) {
  const method = methodById(methodId);
  const metrics = dataset.metrics[methodId];
  $(`metric-${prefix}-name`).textContent = method.label;
  $(`metric-${prefix}-dataset`).textContent = metricText(metrics.chunk_mae_rad);
  $(`metric-${prefix}-global`).textContent = metricText(method.globalMaeRad);
  $(`metric-${prefix}-entropy`).textContent = metricText(metrics.depth_attention_entropy);
}

function renderLabels() {
  const dataset = currentDataset();
  const selection = currentSelection();
  $("dataset-select").value = dataset.id;
  $("method-a-select").value = selection.a;
  $("method-b-select").value = selection.b;
  $("dataset-title").textContent = dataset.label;
  $("method-a-label").textContent = methodById(selection.a).label;
  $("method-b-label").textContent = methodById(selection.b).label;
  if (state.view === "attention") {
    renderMetrics("a", selection.a, dataset);
    renderMetrics("b", selection.b, dataset);
    renderRoiAnalysis(dataset, selection);
  }
  updateReadouts();
}

function renderViewChrome() {
  const attention = state.view === "attention";
  document.body.dataset.view = state.view;
  document.title = attention ? "RGB-D Depth Lab | 完整 ACT 热力图" : "RGB-D Depth Lab | 完整深度视频";
  $("brand-subtitle").textContent = attention ? "完整视频 · 无 Prompt ACT 热力图对比" : "完整视频 · 三相机深度处理对比";
  $("stamp-label").textContent = attention ? "评测模型" : "显示范围";
  $("stamp-value").textContent = attention ? "Depth-only ACT · ckpt 009000" : DEPTH_CAPTION;
  $("method-a-caption").textContent = attention ? ATTENTION_CAPTION : DEPTH_CAPTION;
  $("method-b-caption").textContent = attention ? ATTENTION_CAPTION : DEPTH_CAPTION;
  $("depth-summary").classList.toggle("hidden", attention);
  $("metrics-band").classList.toggle("hidden", !attention);
  $("roi-analysis").classList.toggle("hidden", !attention);
  $("legend-ramp").className = `legend-ramp ${attention ? "attention-ramp" : "depth-ramp"}`;
  $("legend-low").textContent = attention ? "低" : "0.2 m";
  $("legend-high").textContent = attention ? "高" : "4.0 m";
  $("method-note-text").textContent = attention
    ? "完整热力图视频仅表示同一 ACT 模型对不同深度输入的视觉关注分布；任务结论需结合动作误差与负对照。"
    : "深度视频使用统一量程伪彩显示，可连续观察空洞、边缘、噪声和时序稳定性。";
  document.querySelectorAll(".view-switcher button").forEach((button) => {
    const active = button.dataset.view === state.view;
    button.classList.toggle("active", active);
    button.setAttribute("aria-current", active ? "page" : "false");
  });
  lucide.createIcons({ attrs: { "stroke-width": 1.8 } });
}

async function setView(view) {
  if (!['depth', 'attention'].includes(view) || view === state.view) return;
  state.view = view;
  window.history.replaceState({}, "", `?view=${view}`);
  refreshMethodControls();
  renderViewChrome();
  renderLabels();
  await loadVideos(true);
}

function setupControls() {
  for (const dataset of state.catalog.datasets) {
    const option = document.createElement("option");
    option.value = dataset.id;
    option.textContent = dataset.label;
    $("dataset-select").append(option);
  }
  refreshMethodControls();
  document.querySelectorAll(".view-switcher button").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
  $("dataset-select").addEventListener("change", async (event) => {
    state.datasetIndex = state.catalog.datasets.findIndex((dataset) => dataset.id === event.target.value);
    renderLabels();
    await loadVideos(false);
  });
  $("method-a-select").addEventListener("change", async (event) => {
    currentSelection().a = event.target.value;
    renderLabels();
    await loadVideos(true);
  });
  $("method-b-select").addEventListener("change", async (event) => {
    currentSelection().b = event.target.value;
    renderLabels();
    await loadVideos(true);
  });
  $("swap-methods").addEventListener("click", async () => {
    const selection = currentSelection();
    [selection.a, selection.b] = [selection.b, selection.a];
    refreshMethodControls();
    renderLabels();
    await loadVideos(true);
  });
  $("play-pause").addEventListener("click", togglePlayback);
  $("previous-frame").addEventListener("click", () => seekTo($("rgb-video").currentTime - 1 / state.catalog.fps));
  $("next-frame").addEventListener("click", () => seekTo($("rgb-video").currentTime + 1 / state.catalog.fps));
  $("timeline").addEventListener("input", (event) => seekTo(Number(event.target.value)));
  $("speed-select").addEventListener("change", (event) => {
    state.playbackRate = Number(event.target.value);
    videos().forEach((video) => { video.playbackRate = state.playbackRate; });
  });
  $("rgb-video").addEventListener("ended", pauseVideos);
  window.addEventListener("keydown", (event) => {
    if (["SELECT", "INPUT"].includes(document.activeElement?.tagName)) return;
    if (event.key === "ArrowLeft") seekTo($("rgb-video").currentTime - 1 / state.catalog.fps);
    if (event.key === "ArrowRight") seekTo($("rgb-video").currentTime + 1 / state.catalog.fps);
    if (event.key === " ") { event.preventDefault(); togglePlayback(); }
  });
}

async function initialize() {
  try {
    if (window.RGBD_CATALOG) {
      state.catalog = window.RGBD_CATALOG;
    } else {
      const response = await fetch("data/catalog.json", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      state.catalog = await response.json();
    }
    state.roiMetrics = window.ACT_ROI_METRICS || null;
    if (!state.roiMetrics) throw new Error("最新 ROI 注意力指标未加载");
    state.depthQualityMetrics = window.DEPTH_QUALITY_METRICS || null;
    const videoCount = state.catalog.datasets.length * (1 + 7 + 9);
    $("media-count").textContent = `${videoCount} 个视频`;
    renderProjectMetrics();
    setupControls();
    renderViewChrome();
    renderLabels();
    renderPlayIcon();
    await loadVideos(false);
  } catch (error) {
    setLoading(true, `页面数据加载失败：${error.message}`);
  }
}

initialize();
