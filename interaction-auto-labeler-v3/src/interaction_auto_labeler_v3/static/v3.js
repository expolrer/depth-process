(() => {
  const byId = id => document.getElementById(id);
  const escape = value => String(value ?? "").replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[char]));

  function render(payload) {
    const summary = payload.summary || {};
    const quality = summary.quality || {};
    const corrections = payload.corrections || {};
    byId("v3-version").textContent = summary.dataset_id
      ? `${summary.dataset_id} · ${summary.dataset_version || "未版本化"}`
      : "事件图、可靠度、GVL、异常与几何审计";
    byId("v3-state").textContent = summary.state === "completed" ? "已完成" : "等待处理";
    const metrics = [
      [summary.event_graphs || 0, "事件图"],
      [quality.accept || 0, "自动接收"],
      [quality.review || 0, "等待复核"],
      [quality.reject || 0, "自动拒收"],
      [corrections.correction_count || 0, "人工纠错"],
    ];
    byId("v3-metrics").innerHTML = metrics.map(([value, label]) =>
      `<div class="v3-metric"><strong>${escape(value)}</strong><span>${escape(label)}</span></div>`
    ).join("");
    const pending = (payload.quality || []).filter(row => row.quality_gate?.decision !== "accept");
    byId("v3-quality").innerHTML = pending.length
      ? `<span class="warning">${pending.length} 个 episode 仍需人工复核；展开左侧动作逐帧修正，纠错会写入 V3 蒸馏日志。</span>`
      : "当前没有待复核的 V3 质量项。";
  }

  async function refresh() {
    try {
      const response = await fetch("/api/v3/overview", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      render(await response.json());
    } catch (error) {
      byId("v3-quality").textContent = `V3 状态读取失败：${error.message}`;
    }
  }

  refresh();
  window.setInterval(refresh, 3000);
})();
