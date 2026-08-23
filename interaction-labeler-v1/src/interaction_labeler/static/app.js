const state = {
  session: null,
  eventId: null,
  frame: 0,
  stage: "contact_frame",
  drag: null,
};

const stages = [
  ["start_frame", "开始"],
  ["contact_frame", "接触"],
  ["release_frame", "释放"],
  ["end_frame", "结束"],
];

const roleLabels = {
  head: "头部视角",
  left_wrist: "左腕视角",
  right_wrist: "右腕视角",
  other: "对照视角",
};

const $ = selector => document.querySelector(selector);
const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
}[char]));

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => node.classList.remove("show"), 2200);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function currentEvent() {
  return state.session.events.find(event => event.event_id === state.eventId);
}

function viewKey(eventId, camera) {
  return `${eventId}::${camera}`;
}

function manualAnnotation(eventId, camera, frame) {
  const items = state.session.annotations?.items || [];
  return items.find(item => item.event_id === eventId && item.camera === camera && Number(item.frame_index) === Number(frame));
}

function viewHasManual(event, view) {
  return (state.session.annotations?.items || []).some(item => item.event_id === event.event_id && item.camera === view.camera);
}

function eventNeedsReview(event) {
  return event.views.some(view => state.session.ranked?.[viewKey(event.event_id, view.camera)]?.needs_vlm)
    || Boolean(state.session.v2_evidence?.[event.event_id]?.needs_review)
    || (state.session.v2_review_queue || []).some(row => row.event_id === event.event_id)
    || state.session.vlm_resolutions?.[event.event_id]?.status === "pending_human_review";
}

function renderEventList() {
  const query = $("#search").value.trim().toLowerCase();
  const filter = $("#filter").value;
  const events = state.session.events.filter(event => {
    const haystack = `${event.event_id} ${event.instruction}`.toLowerCase();
    if (query && !haystack.includes(query)) return false;
    if (filter === "review" && !eventNeedsReview(event)) return false;
    if (filter === "manual" && !event.views.some(view => viewHasManual(event, view))) return false;
    return true;
  });
  $("#events").innerHTML = events.map(event => {
    const manualCount = event.views.filter(view => viewHasManual(event, view)).length;
    const warning = eventNeedsReview(event) ? '<span class="flag">需要风险复核</span>' : "";
    return `<button class="event-row ${event.event_id === state.eventId ? "active" : ""}" data-event="${esc(event.event_id)}">
      <strong>${esc(event.sequence)} · ${event.side === "left" ? "左手" : "右手"}</strong>
      <span>${esc(event.event_id)}</span>
      <span>${manualCount} 个人工视角 ${warning}</span>
    </button>`;
  }).join("") || '<div class="empty">没有匹配的动作</div>';
  document.querySelectorAll(".event-row").forEach(button => {
    button.onclick = () => selectEvent(button.dataset.event);
  });
}

function selectEvent(eventId) {
  state.eventId = eventId;
  state.stage = "contact_frame";
  const event = currentEvent();
  const primary = event.views[0];
  state.frame = primary?.contact_frame || 0;
  render();
}

function renderStageControl() {
  $("#stage-control").innerHTML = stages.map(([key, label]) =>
    `<button data-stage="${key}" class="${state.stage === key ? "active" : ""}">${label}</button>`
  ).join("");
  document.querySelectorAll("[data-stage]").forEach(button => {
    button.onclick = () => {
      state.stage = button.dataset.stage;
      const event = currentEvent();
      state.frame = event.views[0]?.[state.stage] || 0;
      renderMain();
    };
  });
}

function frameForView(view) {
  const primary = currentEvent().views[0];
  if (!primary || primary.frame_count <= 1) return Math.min(view.frame_count - 1, state.frame);
  const ratio = state.frame / (primary.frame_count - 1);
  return Math.round(ratio * (view.frame_count - 1));
}

function frameUrl(event, view, frame) {
  const query = new URLSearchParams({ event_id: event.event_id, camera: view.camera, frame: String(frame) });
  return `/api/frame?${query}`;
}

function boxesFor(event, view, frame) {
  const boxes = [];
  const ranked = state.session.ranked?.[viewKey(event.event_id, view.camera)];
  if (Number(frame) === Number(view.contact_frame) && ranked) {
    ranked.candidates.forEach(candidate => boxes.push({
      box: candidate.box_xyxy,
      label: `#${candidate.candidate_id} ${candidate.label}`,
      kind: Number(candidate.candidate_id) === Number(ranked.selected_candidate_id) ? "selected" : "candidate",
    }));
  }
  const track = state.session.tracks?.[viewKey(event.event_id, view.camera)]?.find(row => Number(row.frame_index) === Number(frame));
  if (track?.bbox_xyxy) boxes.push({ box: track.bbox_xyxy, label: "track", kind: track.needs_review ? "warning" : "selected" });
  const manual = manualAnnotation(event.event_id, view.camera, frame);
  if (manual) boxes.push({ box: manual.box_xyxy, label: "人工覆盖", kind: "manual" });
  return boxes;
}

function drawCanvas(canvas, image, boxes, draft = null) {
  const rect = image.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const context = canvas.getContext("2d");
  context.scale(dpr, dpr);
  context.clearRect(0, 0, rect.width, rect.height);
  if (!image.naturalWidth || !image.naturalHeight) return;
  const scale = Math.min(rect.width / image.naturalWidth, rect.height / image.naturalHeight);
  const shownWidth = image.naturalWidth * scale;
  const shownHeight = image.naturalHeight * scale;
  const offsetX = (rect.width - shownWidth) / 2;
  const offsetY = (rect.height - shownHeight) / 2;
  const colors = { selected: "#ffd400", candidate: "#48b9d2", manual: "#e95454", warning: "#ff8c32" };
  [...boxes, ...(draft ? [{ box: draft, label: "新人工框", kind: "manual" }] : [])].forEach(item => {
    const [x1, y1, x2, y2] = item.box;
    const x = offsetX + x1 * scale;
    const y = offsetY + y1 * scale;
    const width = (x2 - x1) * scale;
    const height = (y2 - y1) * scale;
    context.strokeStyle = colors[item.kind] || colors.candidate;
    context.lineWidth = item.kind === "selected" || item.kind === "manual" ? 3 : 2;
    context.strokeRect(x, y, width, height);
    context.font = "12px Microsoft YaHei, sans-serif";
    const labelWidth = context.measureText(item.label).width + 10;
    context.fillStyle = colors[item.kind] || colors.candidate;
    context.fillRect(x, Math.max(0, y - 20), labelWidth, 20);
    context.fillStyle = "#101719";
    context.fillText(item.label, x + 5, Math.max(13, y - 6));
  });
  canvas.dataset.scale = String(scale);
  canvas.dataset.offsetX = String(offsetX);
  canvas.dataset.offsetY = String(offsetY);
}

function sourcePoint(event, canvas) {
  const rect = canvas.getBoundingClientRect();
  const scale = Number(canvas.dataset.scale || 1);
  const offsetX = Number(canvas.dataset.offsetX || 0);
  const offsetY = Number(canvas.dataset.offsetY || 0);
  return [Math.max(0, (event.clientX - rect.left - offsetX) / scale), Math.max(0, (event.clientY - rect.top - offsetY) / scale)];
}

function bindCanvas(canvas, image, event, view, frame) {
  const redraw = draft => drawCanvas(canvas, image, boxesFor(event, view, frame), draft);
  image.onload = () => redraw();
  if (image.complete) requestAnimationFrame(() => redraw());
  canvas.onpointerdown = pointer => {
    canvas.setPointerCapture(pointer.pointerId);
    state.drag = { start: sourcePoint(pointer, canvas), current: sourcePoint(pointer, canvas), event, view, frame };
  };
  canvas.onpointermove = pointer => {
    if (!state.drag || state.drag.view.camera !== view.camera) return;
    state.drag.current = sourcePoint(pointer, canvas);
    const [x1, y1] = state.drag.start;
    const [x2, y2] = state.drag.current;
    redraw([Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2)]);
  };
  canvas.onpointerup = async pointer => {
    if (!state.drag || state.drag.view.camera !== view.camera) return;
    state.drag.current = sourcePoint(pointer, canvas);
    const [x1, y1] = state.drag.start;
    const [x2, y2] = state.drag.current;
    const box = [Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2)];
    state.drag = null;
    if (box[2] - box[0] < 4 || box[3] - box[1] < 4) return redraw();
    try {
      await api("/api/annotations", { method: "POST", body: JSON.stringify({
        event_id: event.event_id,
        camera: view.camera,
        frame_index: frame,
        box_xyxy: box,
        label: state.session.task.target.names[0] || "target",
        stage: state.session.phase,
      }) });
      await refresh(false);
      toast("人工框已保存");
    } catch (error) { toast(error.message); }
  };
}

function renderViews() {
  const event = currentEvent();
  $("#views").innerHTML = event.views.map((view, index) => {
    const frame = frameForView(view);
    const ranked = state.session.ranked?.[viewKey(event.event_id, view.camera)];
    const footer = ranked ? `${ranked.candidates.length} 个候选 · margin ${Number(ranked.selection_margin || 0).toFixed(3)}` : "等待自动检测";
    return `<article class="view-panel" data-camera="${esc(view.camera)}">
      <div class="view-head"><strong>${esc(roleLabels[view.role] || view.role)} · ${esc(view.camera)}</strong><span>F${frame}</span></div>
      <div class="image-stage"><img src="${frameUrl(event, view, frame)}" alt="${esc(view.camera)} frame ${frame}"><canvas></canvas></div>
      <div class="view-footer"><span>${esc(footer)}</span><span>${manualAnnotation(event.event_id, view.camera, frame) ? "人工覆盖" : ""}</span></div>
    </article>`;
  }).join("");
  document.querySelectorAll(".view-panel").forEach(panel => {
    const view = event.views.find(item => item.camera === panel.dataset.camera);
    const frame = frameForView(view);
    bindCanvas(panel.querySelector("canvas"), panel.querySelector("img"), event, view, frame);
  });
}

function renderCandidates() {
  const event = currentEvent();
  const rows = [];
  event.views.forEach(view => {
    const ranked = state.session.ranked?.[viewKey(event.event_id, view.camera)];
    (ranked?.candidates || []).forEach(candidate => rows.push({ view, ranked, candidate }));
    const manual = manualAnnotation(event.event_id, view.camera, frameForView(view));
    if (manual) rows.push({ view, manual });
  });
  $("#candidates").innerHTML = rows.map(row => {
    if (row.manual) return `<div class="candidate-row manual"><strong>人工</strong><span>${esc(row.view.camera)} · ${esc(row.manual.label)}</span><span>1.000</span><span>当前帧</span><span>${row.manual.box_xyxy.map(x => Math.round(x)).join(", ")}</span><span></span></div>`;
    const selected = Number(row.candidate.candidate_id) === Number(row.ranked.selected_candidate_id);
    return `<div class="candidate-row ${selected ? "selected" : ""}"><strong>#${row.candidate.candidate_id}</strong><span>${esc(row.view.camera)} · ${esc(row.candidate.label)}</span><span>${Number(row.candidate.interaction_score || row.candidate.score || 0).toFixed(3)}</span><span>${esc(row.candidate.role || "target")}</span><span>${row.candidate.box_xyxy.map(x => Math.round(x)).join(", ")}</span><button class="choose-candidate" data-camera="${esc(row.view.camera)}" data-candidate="${row.candidate.candidate_id}">选为目标</button></div>`;
  }).join("") || '<div class="empty">尚无候选；可直接在任一画面拖动标注目标框</div>';
  document.querySelectorAll(".choose-candidate").forEach(button => {
    button.onclick = async () => {
      const row = rows.find(item => !item.manual
        && item.view.camera === button.dataset.camera
        && Number(item.candidate.candidate_id) === Number(button.dataset.candidate));
      if (!row) return;
      try {
        await api("/api/annotations", { method: "POST", body: JSON.stringify({
          event_id: event.event_id,
          camera: row.view.camera,
          frame_index: row.view.contact_frame,
          box_xyxy: row.candidate.box_xyxy,
          label: row.candidate.label,
          note: "Candidate selected with one-click correction",
          stage: state.session.phase,
        }) });
        await refresh(false);
        toast("目标候选已设为人工覆盖");
      } catch (error) { toast(error.message); }
    };
  });
  const count = rows.length;
  const evidence = state.session.v2_evidence?.[event.event_id];
  const review = (state.session.v2_review_queue || []).find(row => row.event_id === event.event_id);
  const vlm = state.session.vlm_resolutions?.[event.event_id];
  const advice = $("#review-advice");
  if (vlm) {
    const choice = vlm.choice || {};
    const target = choice.camera && choice.candidate_id != null
      ? `${choice.camera} / #${choice.candidate_id}`
      : "未给出可靠候选";
    advice.textContent = `VLM 建议：${target} · ${choice.reason_zh || choice.reason || vlm.raw_response || "无说明"}`;
    advice.hidden = false;
  } else {
    advice.hidden = true;
    advice.textContent = "";
  }
  $("#selection-summary").textContent = evidence
    ? `${count} 条记录 · V2 实例分 ${Number(evidence.score || 0).toFixed(3)} · margin ${Number(evidence.margin || 0).toFixed(3)}${review ? ` · ${review.flags.join(" / ")}` : ""}`
    : `${count} 条标注记录`;
}

function lines(value) {
  return (value || []).join("\n");
}

function readLines(selector) {
  return $(selector).value.split(/\r?\n|,/).map(value => value.trim()).filter(Boolean);
}

function openTaskEditor() {
  const task = state.session.task || {};
  const target = task.target || {};
  $("#task-instruction").value = task.instruction || "";
  $("#target-names").value = lines(target.names);
  $("#target-attributes").value = lines(target.attributes);
  $("#source-region").value = target.source_region || "";
  $("#destination-region").value = target.destination_region || "";
  $("#negative-descriptions").value = lines(target.negative_descriptions);
  $("#exemplar-images").value = lines(target.exemplar_images);
  $("#detector-prompt").value = lines(task.detector_prompt);
  $("#task-dialog").showModal();
}

function renderMain() {
  const event = currentEvent();
  if (!event) return;
  $("#event-title").textContent = `${event.sequence} · ${event.side === "left" ? "左手" : "右手"}`;
  $("#instruction").textContent = event.instruction;
  renderStageControl();
  const primary = event.views[0];
  $("#frame-slider").max = Math.max(0, primary.frame_count - 1);
  $("#frame-slider").value = state.frame;
  $("#frame-readout").textContent = `${state.frame} / ${Math.max(0, primary.frame_count - 1)}`;
  renderViews();
  renderCandidates();
}

function renderStatus() {
  const status = state.session.pipeline_status || { stage: "idle", state: "idle" };
  const labels = { idle: "空闲", running: "处理中", completed: "已完成", failed: "处理失败" };
  const node = $("#pipeline-status");
  node.textContent = `${labels[status.state] || status.state} · ${status.stage}`;
  node.className = `status ${status.state}`;
  $("#run-pipeline").disabled = status.state === "running";
}

function render() {
  const session = state.session;
  $("#dataset-meta").textContent = `${session.input_format} · ${session.events.length} 次动作 · ${session.input_path}`;
  if (!state.eventId && session.events.length) {
    state.eventId = session.events[0].event_id;
    state.frame = session.events[0].views[0]?.contact_frame || 0;
  }
  renderEventList();
  renderStatus();
  renderMain();
}

async function refresh(keepFrame = true) {
  const oldFrame = state.frame;
  state.session = await api("/api/session");
  if (!state.session.events.some(event => event.event_id === state.eventId)) {
    state.eventId = state.session.events[0]?.event_id || null;
    state.frame = state.session.events[0]?.views[0]?.contact_frame || 0;
  } else if (keepFrame) {
    state.frame = oldFrame;
  }
  render();
}

$("#search").oninput = renderEventList;
$("#filter").onchange = renderEventList;
$("#refresh").onclick = () => refresh().catch(error => toast(error.message));
$("#edit-task").onclick = openTaskEditor;
$("#close-task").onclick = () => $("#task-dialog").close();
$("#cancel-task").onclick = () => $("#task-dialog").close();
$("#task-form").onsubmit = async event => {
  event.preventDefault();
  try {
    await api("/api/task", { method: "POST", body: JSON.stringify({
      instruction: $("#task-instruction").value.trim(),
      target: {
        names: readLines("#target-names"),
        attributes: readLines("#target-attributes"),
        source_region: $("#source-region").value.trim(),
        destination_region: $("#destination-region").value.trim(),
        negative_descriptions: readLines("#negative-descriptions"),
        exemplar_images: readLines("#exemplar-images"),
      },
      detector_prompt: readLines("#detector-prompt"),
    }) });
    $("#task-dialog").close();
    await refresh(false);
    toast("目标配置已保存");
  } catch (error) { toast(error.message); }
};
$("#frame-slider").oninput = event => { state.frame = Number(event.target.value); state.stage = "custom"; renderMain(); };
$("#previous-frame").onclick = () => { state.frame = Math.max(0, state.frame - 1); state.stage = "custom"; renderMain(); };
$("#next-frame").onclick = () => { const max = Number($("#frame-slider").max); state.frame = Math.min(max, state.frame + 1); state.stage = "custom"; renderMain(); };
$("#set-contact").onclick = async () => {
  const event = currentEvent();
  try {
    await Promise.all(event.views.map(view => api("/api/event-timing", { method: "POST", body: JSON.stringify({ event_id: event.event_id, camera: view.camera, contact_frame: frameForView(view) }) })));
    state.stage = "contact_frame";
    await refresh();
    toast("接触帧已更新");
  } catch (error) { toast(error.message); }
};
$("#delete-box").onclick = async () => {
  const event = currentEvent();
  const items = event.views.map(view => ({ event_id: event.event_id, camera: view.camera, frame_index: frameForView(view) })).filter(item => manualAnnotation(item.event_id, item.camera, item.frame_index));
  if (!items.length) return toast("当前帧没有人工框");
  try {
    await Promise.all(items.map(item => api("/api/annotations/delete", { method: "POST", body: JSON.stringify(item) })));
    await refresh();
    toast("人工框已删除");
  } catch (error) { toast(error.message); }
};
$("#run-pipeline").onclick = async () => {
  try { await api("/api/run", { method: "POST", body: "{}" }); toast("自动处理已启动"); await refresh(); }
  catch (error) { toast(error.message); }
};
$("#export-json").onclick = () => {
  const blob = new Blob([JSON.stringify(state.session, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = state.session.pipeline_version === "v2"
    ? "robot_interaction_annotations_v2.json"
    : "robot_interaction_annotations_v1.json";
  link.click();
  URL.revokeObjectURL(link.href);
};
setInterval(async () => {
  if (state.session?.pipeline_status?.state === "running") await refresh();
}, 2500);
window.addEventListener("resize", () => renderViews());
refresh(false).catch(error => toast(error.message));
