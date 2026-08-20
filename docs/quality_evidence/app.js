const report = window.DEPTH_QUALITY_EVIDENCE;

if (!report) {
  document.body.innerHTML = "<p style='padding:24px'>评估数据未加载。</p>";
  throw new Error("DEPTH_QUALITY_EVIDENCE is unavailable");
}

const $ = (id) => document.getElementById(id);
const methodMap = new Map(report.methods.map((method) => [method.id, method]));
const state = { tab: "quality", a: "lingbot_v05", b: "raw_aligned" };

const tabs = {
  quality: {
    kicker: "证据一",
    title: "无模型的深度质量证据",
    description: "不调用训练模型，联合观察深度完整率、平坦区局部残差、孤立尖峰、RGB/深度边缘对齐和光流补偿时序残差。平滑度必须与边缘保持和原始测量改动一起解释。",
    primary: "rgb_depth_edge_f1",
    primaryLabel: "RGB/深度边缘对齐 F1 ↑",
    source(method) {
      return {
        ...report.no_model_quality.spatial_by_method[method],
        ...report.no_model_quality.temporal_by_method[method],
      };
    },
    metrics: [
      ["valid_fraction", "有效深度覆盖率", "percent", true],
      ["flat_region_roughness_m", "平坦区局部残差", "mm", false],
      ["isolated_spike_rate", "孤立尖峰比例", "percent", false],
      ["rgb_depth_edge_f1", "RGB/深度边缘 F1", "decimal", true],
      ["flow_compensated_median_residual_m", "光流补偿时序残差", "mm", false],
      ["sensor_preservation_median_ae_m", "原始有效测量改动", "mm", false, "仅表示改动幅度"],
    ],
    note: "边缘 F1 越高越好；较低粗糙度可能来自真实去噪，也可能来自过度平滑。",
  },
  recovery: {
    kicker: "证据二",
    title: "自然遮挡与空洞恢复测试",
    description: "只在当前原始深度为空、前后原始帧经 RGB 光流对齐且深度相差不超过 6 cm 的位置建立时序伪真值。覆盖率表示能否填上，5/10 cm 准确率与 MAE 表示填得是否可信。",
    primary: "within_5cm_fraction",
    primaryLabel: "恢复像素 5 cm 内准确率 ↑",
    source(method) { return report.occlusion_recovery.by_method[method]; },
    metrics: [
      ["recovery_coverage", "可验证空洞恢复覆盖率", "percent", true],
      ["within_5cm_fraction", "5 cm 内准确率", "percent", true],
      ["within_10cm_fraction", "10 cm 内准确率", "percent", true],
      ["mae_m", "恢复 MAE", "mm", false],
      ["rmse_m", "恢复 RMSE", "mm", false],
      ["evaluable_hole_pixels", "可验证空洞像素", "integer", true, "各方法使用同一集合"],
    ],
    note: "覆盖率和误差必须成对阅读；只恢复少量容易像素的方法可能获得较低 MAE。",
  },
  geometry: {
    kicker: "证据三",
    title: "三视角几何一致性",
    description: "在同步头部、左腕和右腕深度中分别拟合主要平面，以三视角平均平面 RMSE 衡量表面几何质量，以视角间 RMSE 标准差衡量一致性。严格跨相机重投影因缺少公共机器人坐标系外参而不报告。",
    primary: "mean_plane_rmse_m",
    primaryLabel: "三视角平均平面 RMSE ↓",
    source(method) { return report.multiview_geometry.planarity_by_method[method]; },
    metrics: [
      ["mean_plane_rmse_m", "平均平面 RMSE", "mm", false],
      ["three_view_plane_rmse_std_m", "三视角 RMSE 离散度", "mm", false],
      ["mean_plane_inlier_fraction", "主要平面内点率", "percent", true],
      ["synchronized_frames", "同步评估时刻", "integer", true],
    ],
    note: "该指标比较旋转不变的平面残差；不等同于已标定相机间点云重投影误差。",
  },
  roi: {
    kicker: "证据四",
    title: "任务目标 ROI 几何指标",
    description: "在最终人工批准的 19 次抓取、头部与执行腕 SAM2 mask 内按事件等权统计。重点检查目标区域覆盖、原始空洞填充、表面粗糙度、边界完整度和对原始有效测量的改动。",
    primary: "roi_valid_fraction",
    primaryLabel: "任务 ROI 有效深度覆盖率 ↑",
    source(method) { return report.task_roi_geometry.by_method[method]; },
    metrics: [
      ["roi_valid_fraction", "ROI 有效深度覆盖率", "percent", true],
      ["roi_raw_hole_fill_fraction", "ROI 原始空洞填充率", "percent", true],
      ["roi_surface_roughness_m", "ROI 表面局部残差", "mm", false],
      ["roi_boundary_valid_fraction", "ROI 边界完整率", "percent", true],
      ["roi_boundary_contrast_m", "目标/背景边界深度差", "mm", null, "非单调指标"],
      ["roi_sensor_preservation_median_ae_m", "ROI 原始有效测量改动", "mm", false, "仅表示改动幅度"],
    ],
    note: "ROI 覆盖越高并不自动等于几何正确，需同时检查恢复误差、边界与原始测量保持。",
  },
};

function valueOf(method, key) {
  const value = tabs[state.tab].source(method)?.[key];
  return Number.isFinite(Number(value)) ? Number(value) : null;
}

function formatValue(value, format) {
  if (value === null) return "--";
  if (format === "percent") return `${(value * 100).toFixed(1)}%`;
  if (format === "mm") return `${(value * 1000).toFixed(value < 0.001 ? 2 : 1)} mm`;
  if (format === "integer") return Math.round(value).toLocaleString();
  return value.toFixed(3);
}

function options() {
  for (const select of [$("method-a"), $("method-b")]) {
    select.replaceChildren();
    for (const method of report.methods) {
      const option = document.createElement("option");
      option.value = method.id;
      option.textContent = method.label;
      select.append(option);
    }
  }
  $("method-a").value = state.a;
  $("method-b").value = state.b;
}

function renderScope() {
  const scope = report.sampling;
  $("scope-sequences").textContent = scope.sequences;
  $("scope-cameras").textContent = scope.camera_views;
  $("scope-spatial").textContent = scope.spatial_frames;
  $("scope-temporal").textContent = scope.temporal_triplets;
  $("scope-roi").textContent = scope.roi_frames;
  $("scope-elapsed").textContent = `${report.elapsed_seconds.toFixed(1)} s`;
}

function renderMethod(prefix, methodId) {
  const tab = tabs[state.tab];
  $(`method-${prefix}-name`).textContent = methodMap.get(methodId).label;
  $(`method-${prefix}-metrics`).innerHTML = tab.metrics.map(([key, label, format, , footnote]) => {
    const value = valueOf(methodId, key);
    return `<div class="metric"><small>${label}</small><strong>${formatValue(value, format)}</strong><em>${footnote || ""}</em></div>`;
  }).join("");
}

function orderedRows() {
  const tab = tabs[state.tab];
  const metric = tab.metrics.find(([key]) => key === tab.primary);
  const higher = metric[3];
  return report.methods
    .map((method) => ({ method, value: valueOf(method.id, tab.primary) }))
    .sort((left, right) => {
      if (left.value === null) return 1;
      if (right.value === null) return -1;
      return higher ? right.value - left.value : left.value - right.value;
    });
}

function renderPrimary() {
  const tab = tabs[state.tab];
  const rows = orderedRows();
  const winner = rows.find((row) => row.value !== null);
  $("primary-label").textContent = tab.primaryLabel;
  $("primary-winner").textContent = winner?.method.label || "证据不足";
  const format = tab.metrics.find(([key]) => key === tab.primary)[2];
  $("primary-value").textContent = formatValue(winner?.value ?? null, format);
}

function renderInterpretation() {
  const tab = tabs[state.tab];
  const primaryMetric = tab.metrics.find(([key]) => key === tab.primary);
  const a = valueOf(state.a, tab.primary);
  const b = valueOf(state.b, tab.primary);
  const labelA = methodMap.get(state.a).label;
  const labelB = methodMap.get(state.b).label;
  let comparison = "当前证据不足以比较所选方法。";
  if (a !== null && b !== null) {
    const aBetter = primaryMetric[3] ? a > b : a < b;
    const equal = Math.abs(a - b) < 1e-12;
    comparison = equal
      ? `${labelA} 与 ${labelB} 在主指标上相同。`
      : `${aBetter ? labelA : labelB} 在“${primaryMetric[1]}”上更优（A ${formatValue(a, primaryMetric[2])} / B ${formatValue(b, primaryMetric[2])}）。`;
  }
  $("interpretation-title").textContent = `${tab.title} · A/B 解释`;
  $("interpretation-text").textContent = `${comparison} ${tab.note}`;
}

function renderRanking() {
  const tab = tabs[state.tab];
  const rows = orderedRows();
  const tableMetrics = tab.metrics.slice(0, state.tab === "quality" || state.tab === "roi" ? 5 : 4);
  $("ranking-title").textContent = `${tab.primaryLabel} 排名`;
  $("ranking-note").textContent = tab.note;
  $("ranking-head").innerHTML = `<tr><th>排名</th><th>方法</th><th class="bar-cell">主指标</th>${tableMetrics.map(([, label]) => `<th>${label}</th>`).join("")}</tr>`;
  const finite = rows.map((row) => row.value).filter((value) => value !== null);
  const minimum = Math.min(...finite);
  const maximum = Math.max(...finite);
  const higher = tab.metrics.find(([key]) => key === tab.primary)[3];
  $("ranking-body").innerHTML = rows.map((row, index) => {
    const activeClass = row.method.id === state.a ? "active-a" : row.method.id === state.b ? "active-b" : "";
    let normalized = 0;
    if (row.value !== null && maximum > minimum) normalized = higher ? (row.value - minimum) / (maximum - minimum) : (maximum - row.value) / (maximum - minimum);
    return `<tr class="${activeClass}"><td class="rank-number">${index + 1}</td><td class="method-cell">${row.method.label}</td><td class="bar-cell"><span class="bar-track"><span class="bar-fill" style="width:${(18 + normalized * 82).toFixed(1)}%"></span></span></td>${tableMetrics.map(([key, , format]) => `<td>${formatValue(valueOf(row.method.id, key), format)}</td>`).join("")}</tr>`;
  }).join("");
}

function renderRepresentatives() {
  const select = $("representative-select");
  select.replaceChildren();
  report.representatives.forEach((item, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = item.sequence;
    select.append(option);
  });
  renderRepresentative(0);
}

function renderRepresentative(index) {
  const item = report.representatives[index];
  if (!item) return;
  $("representative-image").src = item.image;
  $("representative-caption").textContent = `${item.event_id} · ${item.camera} · 帧 ${item.frame_index}`;
}

function render() {
  const tab = tabs[state.tab];
  document.querySelectorAll(".evidence-tabs button").forEach((button) => button.classList.toggle("active", button.dataset.tab === state.tab));
  $("evidence-kicker").textContent = tab.kicker;
  $("evidence-title").textContent = tab.title;
  $("evidence-description").textContent = tab.description;
  renderMethod("a", state.a);
  renderMethod("b", state.b);
  renderPrimary();
  renderInterpretation();
  renderRanking();
}

function initialize() {
  const limitations = [
    "原始传感器深度和时序一致结果只是参考，不是激光扫描仪真值。",
    "较低粗糙度可能表示去噪，也可能表示过度平滑，必须与边缘对齐和原始测量保持共同解释。",
    "自然空洞恢复只在时序可见且前后双向一致的像素上评估。",
    "三视角平面性是旋转不变代理指标；由于缺少公共机器人坐标系外参，不能计算严格标定重投影误差。",
    "四类证据均未使用注意力图或训练后的策略模型输出。",
  ];
  options();
  renderScope();
  renderRepresentatives();
  $("limitations").innerHTML = limitations.map((item) => `<li>${item}</li>`).join("");
  $("geometry-warning").querySelector("span").textContent = "rosbag 只提供每个相机内部 TF，没有连接头部、左腕和右腕的公共机器人坐标系外参。因此此处报告旋转不变的三视角平面残差；RGB 特征刚体拟合因重叠不足未形成有效样本，不报告伪精确的重投影误差。";
  document.querySelectorAll(".evidence-tabs button").forEach((button) => button.addEventListener("click", () => { state.tab = button.dataset.tab; render(); }));
  $("method-a").addEventListener("change", (event) => { state.a = event.target.value; render(); });
  $("method-b").addEventListener("change", (event) => { state.b = event.target.value; render(); });
  $("swap-methods").addEventListener("click", () => { [state.a, state.b] = [state.b, state.a]; $("method-a").value = state.a; $("method-b").value = state.b; render(); });
  $("representative-select").addEventListener("change", (event) => renderRepresentative(Number(event.target.value)));
  render();
  lucide.createIcons({ attrs: { "stroke-width": 1.8 } });
}

initialize();
