const report = window.RGBD_AUGMENTATION_REVIEW;

if (!report) {
  document.body.innerHTML = "<p style='padding:24px'>增强审计数据未加载。</p>";
  throw new Error("RGBD_AUGMENTATION_REVIEW is unavailable");
}

const $ = (id) => document.getElementById(id);
const state = { dataset: 0, transform: "random_mask", lightboxIndex: 0, lightboxItems: [] };
const transformMap = new Map(report.transforms.map((item) => [item.id, item]));
const descriptions = {
  notransform: "Identity 作为 3.0 权重的稳定基线，用于保留未增强样本。",
  brightness: "对每个相机的 RGB 采样 0.5–1.5 亮度因子；物理深度数值保持不变。",
  contrast: "对每个相机的 RGB 采样 0.5–1.5 对比度因子；不改变像素坐标或 metric depth。",
  saturation: "对 RGB 颜色饱和度采样 0.5–1.5；深度没有颜色语义，因此不做同名运算。",
  hue: "RGB 色相在 ±0.05 色轮范围内变化；深度保持原始物理量纲。",
  sharpness: "对 RGB 采样 0.5–1.5 锐度因子，用于模拟成像清晰度变化。",
  random_mask: "在每个相机中遮挡 10%×10% 区域。版本一只遮 RGB，版本二把相同区域同步写为无效深度。",
  random_border_cutout: "随机从一个边缘裁除 15% 画面。版本二对 RGB 与深度复用相同边缘掩码。",
  gaussian_noise: "向归一化 RGB 加入标准差 0.05 的高斯噪声；不向 metric depth 注入无物理依据的同尺度噪声。",
  gamma_correction: "对 RGB 采样 0.5–2.0 Gamma；深度数值保持线性米制尺度。",
};
const cameraLabels = { cam_h: "头部", cam_l: "左腕", cam_r: "右腕" };

function percent(value) {
  return value === null || value === undefined ? "不适用" : `${(value * 100).toFixed(1)}%`;
}

function initializeControls() {
  report.datasets.forEach((dataset, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${dataset.label} · ${dataset.sequence}`;
    $("dataset-select").append(option);
  });
  report.transforms.forEach((transform) => {
    const option = document.createElement("option");
    option.value = transform.id;
    option.textContent = `${transform.label} · ${(transform.probability * 100).toFixed(2)}%`;
    $("transform-select").append(option);
  });
  $("transform-select").value = state.transform;
}

function renderScope() {
  const scope = report.sampling;
  $("scope-datasets").textContent = scope.datasets;
  $("scope-cameras").textContent = scope.camera_views;
  $("scope-transforms").textContent = scope.transforms;
  $("scope-samples").textContent = scope.review_samples;
  $("scope-max").textContent = scope.max_num_transforms;
  $("scope-seed").textContent = report.seed;
}

function formatParams(audit) {
  const entries = Object.entries(audit.params);
  if (!entries.length) return "Identity · 无参数";
  return entries.map(([key, value]) => {
    if (typeof value === "number" && !Number.isInteger(value)) return `${key}=${value.toFixed(4)}`;
    return `${key}=${value}`;
  }).join(" · ");
}

function renderParameters(sample) {
  $("parameter-grid").innerHTML = sample.audits.map((audit) => `
    <article>
      <strong>${cameraLabels[audit.camera]}</strong>
      <code>${formatParams(audit)}</code>
    </article>
  `).join("");
}

function renderPolicy() {
  $("policy-body").innerHTML = report.transforms.map((transform) => `
    <tr data-transform="${transform.id}" class="${transform.id === state.transform ? "active" : ""}">
      <td>${transform.label}</td>
      <td>${transform.type}</td>
      <td>${transform.weight.toFixed(1)}</td>
      <td>${(transform.probability * 100).toFixed(2)}%</td>
      <td>${transform.rgb_only_depth_rule}</td>
      <td>${transform.paired_depth_rule}</td>
    </tr>
  `).join("");
  document.querySelectorAll("#policy-body tr").forEach((row) => row.addEventListener("click", () => {
    state.transform = row.dataset.transform;
    $("transform-select").value = state.transform;
    render();
  }));
}

function setImage(id, path) {
  $(id).src = path;
}

function buildLightboxItems() {
  state.lightboxItems = [...document.querySelectorAll(".image-button img")].map((image) => ({
    src: image.src,
    caption: image.closest("figure").querySelector("figcaption").textContent,
  }));
}

function render() {
  const dataset = report.datasets[state.dataset];
  const transform = transformMap.get(state.transform);
  const sample = dataset.samples[state.transform];
  $("transform-family").textContent = transform.family === "spatial_occlusion" ? "空间遮挡" : transform.family === "identity" ? "基线" : "RGB 光度增强";
  $("transform-title").textContent = transform.label;
  $("transform-description").textContent = descriptions[transform.id];
  $("transform-probability").textContent = `${(transform.probability * 100).toFixed(2)}%`;
  $("transform-weight").textContent = `权重 ${transform.weight.toFixed(1)} / 总权重 12.0`;
  $("metric-rgb").textContent = percent(sample.rgb_changed_fraction);
  $("metric-v1-depth").textContent = percent(sample.rgb_only_depth_changed_fraction);
  $("metric-v2-depth").textContent = percent(sample.paired_depth_changed_fraction);
  $("metric-iou").textContent = sample.paired_mask_iou === null ? "不适用" : sample.paired_mask_iou.toFixed(3);
  $("frame-caption").textContent = `${dataset.sequence} · 帧 ${dataset.frame_index}`;
  setImage("original-rgb", dataset.original_rgb);
  setImage("original-depth", dataset.original_depth);
  setImage("v1-rgb", sample.augmented_rgb);
  setImage("v1-depth", dataset.original_depth);
  setImage("v2-rgb", sample.augmented_rgb);
  setImage("v2-depth", sample.paired_depth);

  const verdict = $("verdict");
  verdict.classList.toggle("warning", sample.rgb_only_cross_modal_mismatch);
  verdict.querySelector("span").textContent = sample.rgb_only_cross_modal_mismatch
    ? `版本一产生 ${percent(sample.intended_spatial_mask_fraction)} 的 RGB/深度可见性不一致；版本二以 IoU ${sample.paired_mask_iou.toFixed(3)} 同步空间掩码。`
    : "当前策略只改变 RGB 成像属性，两个版本都保留 metric depth；这不是缺少增强，而是在保持物理尺度。";
  $("v2-depth-caption").textContent = transform.family === "spatial_occlusion" ? "同步写入无效值 0 的深度" : "保持物理尺度的深度";
  renderParameters(sample);
  renderPolicy();
  buildLightboxItems();
}

function openLightbox(index) {
  state.lightboxIndex = index;
  const item = state.lightboxItems[index];
  $("lightbox-image").src = item.src;
  $("lightbox-caption").textContent = item.caption;
  $("lightbox").showModal();
}

function stepLightbox(delta) {
  state.lightboxIndex = (state.lightboxIndex + delta + state.lightboxItems.length) % state.lightboxItems.length;
  const item = state.lightboxItems[state.lightboxIndex];
  $("lightbox-image").src = item.src;
  $("lightbox-caption").textContent = item.caption;
}

function initializeEvents() {
  $("dataset-select").addEventListener("change", (event) => { state.dataset = Number(event.target.value); render(); });
  $("transform-select").addEventListener("change", (event) => { state.transform = event.target.value; render(); });
  document.querySelectorAll(".image-button").forEach((button, index) => button.addEventListener("click", () => openLightbox(index)));
  $("lightbox-close").addEventListener("click", () => $("lightbox").close());
  $("lightbox-prev").addEventListener("click", () => stepLightbox(-1));
  $("lightbox-next").addEventListener("click", () => stepLightbox(1));
  $("lightbox").addEventListener("click", (event) => { if (event.target === $("lightbox")) $("lightbox").close(); });
  window.addEventListener("keydown", (event) => {
    if (!$("lightbox").open) return;
    if (event.key === "ArrowLeft") stepLightbox(-1);
    if (event.key === "ArrowRight") stepLightbox(1);
  });
}

initializeControls();
renderScope();
render();
initializeEvents();
lucide.createIcons({ attrs: { "stroke-width": 1.8 } });
