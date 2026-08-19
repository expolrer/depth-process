#!/usr/bin/env python3
"""Build a static human-review workbench for interaction target proposals."""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def output_url(value: str | None) -> str | None:
    if not value:
        return None
    value = value.replace("\\", "/")
    marker = "/outputs/"
    if marker in value:
        value = value.split(marker, 1)[1]
    elif value.startswith("outputs/"):
        value = value[len("outputs/") :]
    return "../" + value.lstrip("/")


def choose_frames(rows: list[dict], event: dict) -> list[dict]:
    rows = [row for row in rows if row.get("overlay_path")]
    if not rows:
        return []
    by_frame = {int(row["frame_index"]): row for row in rows}
    start = int(event["start_frame"])
    anchor = int(event["anchor_frame"])
    release = int(event["release_frame"])
    end = int(event["end_frame"])
    targets = [
        start,
        anchor,
        round((anchor + release) / 2),
        release,
        end,
    ]
    anomalous = [int(row["frame_index"]) for row in rows if row.get("needs_review")]
    if anomalous:
        targets.append(anomalous[len(anomalous) // 2])
    picked: list[dict] = []
    seen: set[int] = set()
    for target in targets:
        frame = min(by_frame, key=lambda value: abs(value - target))
        if frame not in seen:
            picked.append(by_frame[frame])
            seen.add(frame)
    return picked


def short_sequence(value: str) -> str:
    aliases = {
        "chengzhong_xianxia_main1": "称重",
        "dajian_xianxia_main1": "大件",
        "zhoumian_xianxia_main1": "桌面",
    }
    if value in aliases:
        return aliases[value]
    if "leju_claw" in value:
        return "玩具堆 / 夹爪"
    if "dex_hand" in value:
        return "玩具堆 / 灵巧手"
    return value


def vlm_reason_zh(vlm_row: dict | None, side: str) -> str | None:
    if not vlm_row:
        return None
    choice = vlm_row.get("choice") or {}
    camera_names = {"cam_h": "头部相机", "cam_l": "左腕相机", "cam_r": "右腕相机"}
    hand = "左手" if side == "left" else "右手"
    camera = camera_names.get(str(choice.get("camera")), str(choice.get("camera", "未知视角")))
    candidate_id = choice.get("candidate_id", "未知")
    if vlm_row.get("valid_choice"):
        return f"本地 Qwen3.5-2B 根据三视角候选拼图判断：{camera}中的候选 #{candidate_id} 最可能是与{hand}夹爪接触或被其包围的实际交互物体。"
    return f"本地 Qwen3.5-2B 建议选择{camera}中的候选 #{candidate_id}，但该编号不在有效候选表中，因此建议已被系统拒绝并保留人工复核。"


def build(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    outputs = root / "outputs"
    summary = json.loads((outputs / "target_tracks" / "summary.json").read_text(encoding="utf-8"))
    tracks = read_jsonl(outputs / "target_tracks" / "track_index.jsonl")
    ranked = read_jsonl(outputs / "interaction_candidates" / "ranked_index.jsonl")
    queue = read_jsonl(outputs / "interaction_candidates" / "ambiguity_queue.jsonl")
    vlm = read_jsonl(outputs / "interaction_candidates" / "vlm_resolutions.jsonl")
    overrides = read_jsonl(outputs / "interaction_candidates" / "human_overrides.jsonl")

    tracks_by_event: dict[str, list[dict]] = defaultdict(list)
    ranked_by_event: dict[str, list[dict]] = defaultdict(list)
    for row in tracks:
        tracks_by_event[row["event_id"]].append(row)
    for row in ranked:
        ranked_by_event[row["event_id"]].append(row)
    queue_by_event = {row["event_id"]: row for row in queue}
    vlm_by_event = {row["event_id"]: row for row in vlm}
    override_by_event = {row["event_id"]: row for row in overrides}

    events = []
    for event in summary["events_detail"]:
        event_id = event["event_id"]
        sequence = event_id.split(":grasp_", 1)[0]
        queue_row = queue_by_event.get(event_id, {})
        vlm_row = vlm_by_event.get(event_id)
        camera_rows = ranked_by_event.get(event_id, [])
        selected_row = next(
            (
                row
                for row in camera_rows
                if row.get("camera") == event["camera"]
                and int(row.get("selected_candidate_id", -1)) == int(event["candidate_id"])
            ),
            None,
        )
        candidate = None
        if selected_row:
            candidate = next(
                (
                    item
                    for item in selected_row.get("ranked_candidates", [])
                    if int(item.get("candidate_id", -1)) == int(event["candidate_id"])
                ),
                None,
            )
        collage = queue_row.get("collage_path")
        if not collage and selected_row:
            collage = selected_row.get("ranked_visualization_path") or selected_row.get("visualization_path")

        sample_rows = choose_frames(tracks_by_event[event_id], event)
        anomaly_ratio = event["needs_review_frames"] / max(1, event["tracked_frames"])
        priority = "critical" if anomaly_ratio >= 0.2 else "warning" if event["needs_review_frames"] else "normal"
        choice = (vlm_row or {}).get("choice", {})
        events.append(
            {
                **event,
                "sequence": sequence,
                "sequence_label": short_sequence(sequence),
                "priority": priority,
                "anomaly_ratio": anomaly_ratio,
                "collage_url": output_url(collage),
                "candidate_label": (candidate or {}).get("label", "目标候选"),
                "candidate_score": (candidate or {}).get("interaction_score"),
                "vlm": {
                    "used": vlm_row is not None,
                    "valid": (vlm_row or {}).get("valid_choice"),
                    "confidence": choice.get("confidence"),
                    "reason": vlm_reason_zh(vlm_row, event_id.rsplit("_", 1)[-1]),
                },
                "override": override_by_event.get(event_id),
                "samples": [
                    {
                        "frame": row["frame_index"],
                        "overlay_url": output_url(row.get("overlay_path")),
                        "area_ratio": row.get("area_ratio"),
                        "needs_review": bool(row.get("needs_review")),
                    }
                    for row in sample_rows
                ],
            }
        )

    priority_order = {"critical": 0, "warning": 1, "normal": 2}
    events.sort(key=lambda row: (priority_order[row["priority"]], row["sequence"], row["anchor_frame"]))
    payload = json.dumps(events, ensure_ascii=False).replace("</", "<\\/")
    tracked_frames = int(summary["tracked_frames"])
    out_dir = outputs / "interaction_review"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "review_manifest.json").write_text(
        json.dumps({"schema": "interaction_review_v1", "events": events}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>交互目标人工抽检</title>
<style>
:root{{--ink:#172026;--muted:#65727a;--line:#d9dfe2;--paper:#f4f6f6;--panel:#fff;--accent:#126b62;--warn:#a45d12;--bad:#a12d34;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"Microsoft YaHei",sans-serif;letter-spacing:0}}
header{{position:sticky;top:0;z-index:4;background:#fff;border-bottom:1px solid var(--line);padding:14px 24px}}
.head{{max-width:1520px;margin:auto;display:flex;gap:18px;align-items:center;justify-content:space-between;flex-wrap:wrap}}
h1{{font-size:20px;margin:0}} .summary{{color:var(--muted);font-size:13px}} .toolbar{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
button{{border:1px solid #bac4c7;background:#fff;color:var(--ink);height:34px;padding:0 12px;border-radius:6px;cursor:pointer;font-weight:600}}
button:hover{{border-color:var(--accent)}} button.active{{background:var(--accent);color:#fff;border-color:var(--accent)}}
main{{max-width:1520px;margin:18px auto;padding:0 20px 56px;display:grid;gap:16px}}
.card{{background:var(--panel);border:1px solid var(--line);border-left:5px solid #87969a;border-radius:7px;overflow:hidden}}
.card.warning{{border-left-color:var(--warn)}} .card.critical{{border-left-color:var(--bad)}}
.card-head{{display:flex;gap:14px;justify-content:space-between;align-items:flex-start;padding:14px 16px;border-bottom:1px solid var(--line)}}
.title{{font-size:17px;font-weight:750}} .meta{{font-size:12px;color:var(--muted);margin-top:5px;line-height:1.6}} .tag{{display:inline-block;border:1px solid var(--line);padding:2px 6px;border-radius:4px;margin-right:5px;background:#f8f9f9}}
.body{{display:grid;grid-template-columns:minmax(320px,1fr) minmax(500px,1.8fr);gap:14px;padding:14px 16px}}
.candidate{{width:100%;max-height:430px;object-fit:contain;background:#111;border-radius:4px}} .samples{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}}
.sample{{margin:0;position:relative;background:#111;border-radius:4px;overflow:hidden;min-height:120px}} .sample img{{width:100%;height:100%;object-fit:contain;display:block}}
.sample figcaption{{position:absolute;left:5px;bottom:5px;background:rgba(0,0,0,.72);color:#fff;padding:3px 6px;border-radius:3px;font-size:11px}}
.sample.bad{{outline:3px solid var(--bad)}} .explain{{font-size:13px;line-height:1.55;margin-top:10px;color:#3c474d}}
.review{{display:grid;grid-template-columns:auto auto auto 1fr;gap:8px;padding:12px 16px;border-top:1px solid var(--line);align-items:center}}
.review button.selected.approve{{background:var(--accent);color:#fff}} .review button.selected.reject{{background:var(--bad);color:#fff}} .review button.selected.correct{{background:var(--warn);color:#fff}}
input{{height:34px;border:1px solid #bac4c7;border-radius:5px;padding:0 10px;min-width:220px}} .hidden{{display:none}}
@media(max-width:900px){{.body{{grid-template-columns:1fr}}.samples{{grid-template-columns:repeat(2,minmax(0,1fr))}}.review{{grid-template-columns:1fr 1fr}}input{{grid-column:1/-1;width:100%}}}}
</style></head>
<body><header><div class="head"><div><h1>交互目标人工抽检</h1><div class="summary" id="summary"></div></div><div class="toolbar">
<button data-filter="all" class="active">全部</button><button data-filter="pending">待审核</button><button data-filter="problem">异常优先</button><button data-filter="approved">已批准</button><button id="export">导出 JSON</button>
</div></div></header><main id="events"></main>
<script>
const EVENTS={payload}; const TRACKED_FRAMES={tracked_frames}; const KEY='interaction-review-v1'; let state=JSON.parse(localStorage.getItem(KEY)||'{{}}'); let filter='all';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function save(){{localStorage.setItem(KEY,JSON.stringify(state));render()}}
function decision(id,value){{state[id]=state[id]||{{}};state[id].decision=value;state[id].updated_at=new Date().toISOString();save()}}
function note(id,value){{state[id]=state[id]||{{}};state[id].note=value;state[id].updated_at=new Date().toISOString();localStorage.setItem(KEY,JSON.stringify(state));updateSummary()}}
function updateSummary(){{let a=0,r=0,c=0;Object.values(state).forEach(x=>{{a+=x.decision==='approved';r+=x.decision==='rejected';c+=x.decision==='correction'}});document.getElementById('summary').textContent=`${{EVENTS.length}} 个抓取事件 · ${{TRACKED_FRAMES}} 帧跟踪 · 已批准 ${{a}} · 拒绝 ${{r}} · 待修正 ${{c}}`;}}
function visible(e){{const d=(state[e.event_id]||{{}}).decision||'pending';if(filter==='all')return true;if(filter==='problem')return e.priority!=='normal';if(filter==='approved')return d==='approved';return d==='pending';}}
function render(){{const host=document.getElementById('events');host.innerHTML=EVENTS.filter(visible).map(e=>{{const s=state[e.event_id]||{{}};const pct=(100*e.anomaly_ratio).toFixed(1);const source=e.selection_source==='vlm'?'Qwen3.5-2B 建议':e.selection_source==='review_override'?'时序视觉复核覆盖':'规则自动选择';const v=e.vlm.used?`<div class="explain"><b>Qwen3.5-2B 建议${{e.override?'（已被时序复核覆盖）':''}}：</b>置信度 ${{esc(e.vlm.confidence)}}；${{esc(e.vlm.reason)}}</div>`:'';const o=e.override?`<div class="explain"><b>${{esc(e.override.reviewer||'Codex 辅助时序视觉复核')}}：</b>${{esc(e.override.reason)}}</div>`:'';const images=e.samples.map(x=>`<figure class="sample ${{x.needs_review?'bad':''}}"><img loading="lazy" src="${{esc(x.overlay_url)}}"><figcaption>帧 ${{x.frame}} · mask ${{(100*(x.area_ratio||0)).toFixed(1)}}%${{x.needs_review?' · 异常':''}}</figcaption></figure>`).join('');return `<section class="card ${{e.priority}}" data-id="${{esc(e.event_id)}}"><div class="card-head"><div><div class="title">${{esc(e.sequence_label)}} · 第 ${{e.event_id.match(/grasp_([0-9]+)/)?.[1]||'?'}} 次抓取 · ${{e.side==='left'?'左手':'右手'}}</div><div class="meta"><span class="tag">${{esc(e.camera)}}</span><span class="tag">${{source}}</span><span class="tag">候选 #${{e.candidate_id}}</span><span class="tag">${{e.tracked_frames}} 帧</span>区间 ${{e.start_frame}} → ${{e.end_frame}}，初始化 ${{e.anchor_frame}}，释放 ${{e.release_frame}}</div></div><div><b>${{e.needs_review_frames}}</b> 异常帧 / ${{pct}}%</div></div><div class="body"><div>${{e.collage_url?`<img class="candidate" src="${{esc(e.collage_url)}}">`:''}}<div class="explain"><b>候选：</b>${{esc(e.candidate_label)}}${{e.candidate_score!=null?'，融合分 '+e.candidate_score.toFixed(3):''}}</div>${{o}}${{v}}</div><div class="samples">${{images}}</div></div><div class="review"><button class="approve ${{s.decision==='approved'?'selected':''}}" onclick="decision('${{esc(e.event_id)}}','approved')">批准</button><button class="reject ${{s.decision==='rejected'?'selected':''}}" onclick="decision('${{esc(e.event_id)}}','rejected')">拒绝</button><button class="correct ${{s.decision==='correction'?'selected':''}}" onclick="decision('${{esc(e.event_id)}}','correction')">需要修正</button><input placeholder="审核备注" value="${{esc(s.note||'')}}" oninput="note('${{esc(e.event_id)}}',this.value)"></div></section>`}}).join('');updateSummary()}}
document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{{filter=b.dataset.filter;document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x===b));render()}});
document.getElementById('export').onclick=()=>{{const blob=new Blob([JSON.stringify({{schema:'interaction_review_decisions_v1',exported_at:new Date().toISOString(),decisions:state}},null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='interaction_review_decisions.json';a.click();URL.revokeObjectURL(a.href)}};render();
</script></body></html>"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"events": len(events), "output": str(out_dir / "index.html")}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
