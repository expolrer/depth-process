#!/usr/bin/env python3
"""Build the detailed Qwen/Codex multiview grasp review page."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def output_url(value: str | None) -> str | None:
    if not value:
        return None
    value = value.replace("\\", "/")
    if "/outputs/" in value:
        value = value.split("/outputs/", 1)[1]
    elif value.startswith("outputs/"):
        value = value[len("outputs/") :]
    return "../" + value.lstrip("/")


def sequence_label(value: str) -> str:
    if value == "chengzhong_xianxia_main1":
        return "称重"
    if value == "dajian_xianxia_main1":
        return "大件"
    if value == "zhoumian_xianxia_main1":
        return "桌面"
    if "leju_claw" in value:
        return "玩具堆 / 夹爪"
    if "dex_hand" in value:
        return "玩具堆 / 灵巧手"
    return value


def choose_overlays(rows: list[dict[str, Any]], event: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in rows if row.get("overlay_path")]
    if not rows:
        return []
    targets = [
        int(event["start_frame"]),
        int(event["anchor_frame"]),
        round((int(event["anchor_frame"]) + int(event["release_frame"])) / 2),
        int(event["release_frame"]),
        int(event["end_frame"]),
    ]
    picked, seen = [], set()
    for target in targets:
        row = min(rows, key=lambda item: abs(int(item["frame_index"]) - target))
        frame = int(row["frame_index"])
        if frame not in seen:
            picked.append({
                "frame": frame,
                "url": output_url(row["overlay_path"]),
                "area_ratio": row.get("area_ratio", 0),
                "needs_review": bool(row.get("needs_review")),
            })
            seen.add(frame)
    return picked


def build(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    outputs = root / "outputs"
    summary = json.loads((outputs / "target_tracks" / "summary.json").read_text(encoding="utf-8"))
    tracks = read_jsonl(outputs / "target_tracks" / "track_index.jsonl")
    ranked = read_jsonl(outputs / "interaction_candidates" / "ranked_index.jsonl")
    queue = read_jsonl(outputs / "interaction_candidates" / "ambiguity_queue.jsonl")
    temporal = read_jsonl(outputs / "interaction_review" / "temporal_evidence.jsonl")
    qwen = read_jsonl(outputs / "interaction_review" / "qwen_temporal_annotations.jsonl")
    codex = read_jsonl(outputs / "interaction_review" / "codex_temporal_audits.jsonl")

    tracks_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tracks:
        tracks_by_event[row["event_id"]].append(row)
    ranked_by_key = {(row["event_id"], row["camera"]): row for row in ranked}
    queue_by_event = {row["event_id"]: row for row in queue}
    temporal_by_event = {row["event_id"]: row for row in temporal}
    qwen_by_event = {row["event_id"]: row for row in qwen}
    codex_by_event = {row["event_id"]: row for row in codex}

    events = []
    for event in summary["events_detail"]:
        event_id = event["event_id"]
        sequence = event_id.split(":grasp_", 1)[0]
        selected_row = ranked_by_key[(event_id, event["camera"])]
        candidate = next(
            (item for item in selected_row.get("ranked_candidates", []) if int(item["candidate_id"]) == int(event["candidate_id"])),
            {},
        )
        timeline = temporal_by_event[event_id]
        timeline = {
            **timeline,
            "temporal_collage_url": output_url(timeline["temporal_collage_path"]),
            "candidate_collage_url": output_url(timeline["candidate_collage_path"]),
        }
        for camera in timeline["cameras"]:
            for stage in camera["stages"]:
                stage["rgb_url"] = output_url(stage["rgb_path"])
        anomaly_ratio = int(event["needs_review_frames"]) / max(1, int(event["tracked_frames"]))
        priority = "warning" if event["needs_review_frames"] else "normal"
        if event["selection_source"] == "review_override":
            priority = "warning"
        events.append({
            **event,
            "sequence": sequence,
            "sequence_label": sequence_label(sequence),
            "side_label": "左手" if event_id.endswith("_left") else "右手",
            "priority": priority,
            "anomaly_ratio": anomaly_ratio,
            "candidate_label": candidate.get("label", "目标候选"),
            "candidate_score": candidate.get("interaction_score"),
            "candidate_depth": candidate.get("depth"),
            "candidate_collage_url": output_url(queue_by_event[event_id]["collage_path"]),
            "overlays": choose_overlays(tracks_by_event[event_id], event),
            "temporal": timeline,
            "qwen": qwen_by_event[event_id],
            "codex": codex_by_event[event_id],
        })

    events.sort(key=lambda row: (0 if row["priority"] == "warning" else 1, row["sequence"], row["anchor_frame"]))
    payload = json.dumps(events, ensure_ascii=False).replace("</", "<\\/")
    out_dir = outputs / "interaction_review"
    (out_dir / "review_manifest.json").write_text(
        json.dumps({"schema": "detailed_interaction_review_v2", "events": events}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>三视角抓取详细标注</title>
<style>
:root{{--ink:#172126;--muted:#66747b;--line:#d6dde0;--paper:#f3f5f5;--panel:#fff;--qwen:#155e75;--codex:#31633a;--warn:#a45d12;--bad:#a12d34;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"Microsoft YaHei",sans-serif;letter-spacing:0}}
header{{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid var(--line);padding:13px 22px}}
.head{{max-width:1800px;margin:auto;display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}}
h1{{font-size:20px;margin:0}} .summary{{font-size:13px;color:var(--muted);margin-top:4px}} .toolbar{{display:flex;gap:7px;flex-wrap:wrap}}
button{{height:34px;border:1px solid #b9c3c7;background:#fff;border-radius:6px;padding:0 12px;cursor:pointer;font-weight:650;color:var(--ink)}}
button.active{{background:#1d6c64;color:#fff;border-color:#1d6c64}} main{{max-width:1800px;margin:16px auto;padding:0 18px 60px;display:grid;gap:16px}}
.event{{background:var(--panel);border:1px solid var(--line);border-left:5px solid #829094;border-radius:7px;overflow:hidden}} .event.warning{{border-left-color:var(--warn)}}
.event-head{{display:flex;justify-content:space-between;gap:18px;padding:14px 16px;border-bottom:1px solid var(--line)}} .event-title{{font-size:17px;font-weight:750}}
.meta{{font-size:12px;color:var(--muted);margin-top:5px;line-height:1.7}} .tag{{display:inline-block;padding:2px 6px;border:1px solid var(--line);border-radius:4px;margin-right:5px;background:#f8f9f9}}
.models{{display:grid;grid-template-columns:1fr 1fr;border-bottom:1px solid var(--line)}} .model{{padding:15px 16px}} .model+ .model{{border-left:1px solid var(--line)}}
.model h3{{font-size:15px;margin:0 0 9px}} .model.qwen h3{{color:var(--qwen)}} .model.codex h3{{color:var(--codex)}}
.model dl{{display:grid;grid-template-columns:105px 1fr;gap:6px 10px;margin:0;font-size:13px;line-height:1.55}} .model dt{{font-weight:700;color:#46545a}} .model dd{{margin:0}}
.recommend{{margin-top:10px;padding-top:9px;border-top:1px solid var(--line);font-size:13px;font-weight:700}}
.evidence{{display:grid;grid-template-columns:minmax(340px,.9fr) minmax(620px,1.6fr);gap:14px;padding:14px 16px;border-bottom:1px solid var(--line)}}
.candidate{{width:100%;max-height:420px;object-fit:contain;background:#101415;border-radius:4px;cursor:zoom-in}} .mask-strip{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}}
.figure{{margin:0;position:relative;background:#101415;border-radius:4px;overflow:hidden;min-height:120px}} .figure img{{width:100%;height:100%;object-fit:contain;display:block;cursor:zoom-in}}
.figure figcaption{{position:absolute;left:5px;bottom:5px;background:rgba(0,0,0,.74);color:#fff;padding:3px 6px;border-radius:3px;font-size:11px}} .figure.bad{{outline:3px solid var(--bad)}}
details{{border-bottom:1px solid var(--line)}} summary{{cursor:pointer;padding:13px 16px;font-weight:750;background:#fafbfb}} details[open] summary{{border-bottom:1px solid var(--line)}}
.timeline{{padding:14px 16px;display:grid;gap:18px}} .camera-row h4{{margin:0 0 7px;font-size:14px}} .strip-wrap{{overflow-x:auto;padding-bottom:5px}}
.timeline-strip{{display:grid;grid-template-columns:repeat(6,minmax(220px,1fr));gap:7px;min-width:1320px}} .timeline-strip .figure{{aspect-ratio:848/480;min-height:0}}
.review{{display:grid;grid-template-columns:auto auto auto 1fr;gap:8px;padding:12px 16px;align-items:center}} .review button.selected.approve{{background:#1d6c64;color:#fff}} .review button.selected.reject{{background:var(--bad);color:#fff}} .review button.selected.correct{{background:var(--warn);color:#fff}}
input{{height:34px;border:1px solid #b9c3c7;border-radius:5px;padding:0 10px;min-width:250px}}
dialog{{border:0;border-radius:6px;padding:0;background:#0c1011;max-width:96vw;max-height:96vh;box-shadow:0 20px 80px rgba(0,0,0,.45)}} dialog::backdrop{{background:rgba(0,0,0,.78)}} dialog img{{display:block;max-width:94vw;max-height:88vh;object-fit:contain}} .dialog-head{{display:flex;justify-content:space-between;align-items:center;color:#fff;padding:8px 12px;font-size:13px}} .dialog-head button{{background:#20282b;color:#fff;border-color:#465156}}
@media(max-width:980px){{.models,.evidence{{grid-template-columns:1fr}}.model+.model{{border-left:0;border-top:1px solid var(--line)}}.mask-strip{{grid-template-columns:repeat(2,minmax(0,1fr))}}.review{{grid-template-columns:1fr 1fr}}input{{grid-column:1/-1;width:100%}}}}
</style></head><body>
<header><div class="head"><div><h1>三视角抓取详细标注</h1><div class="summary" id="summary"></div></div><div class="toolbar"><button data-filter="all" class="active">全部</button><button data-filter="pending">待审核</button><button data-filter="problem">异常优先</button><button data-filter="approved">已批准</button><button id="export">导出完整 JSON</button></div></div></header>
<main id="events"></main><dialog id="viewer"><div class="dialog-head"><span id="viewer-title"></span><button onclick="viewer.close()">关闭</button></div><img id="viewer-image"></dialog>
<script>
const EVENTS={payload}; const KEY='interaction-review-v2'; let state=JSON.parse(localStorage.getItem(KEY)||'{{}}'); let filter='all';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const sourceName=s=>s==='vlm'?'Qwen/VLM':s==='review_override'?'Codex 时序复核覆盖':'规则自动选择';
function showImage(src,title){{document.getElementById('viewer-image').src=src;document.getElementById('viewer-title').textContent=title;document.getElementById('viewer').showModal()}}
function save(){{localStorage.setItem(KEY,JSON.stringify(state));render()}} function decision(id,value){{state[id]=state[id]||{{}};state[id].decision=value;state[id].updated_at=new Date().toISOString();save()}}
function note(id,value){{state[id]=state[id]||{{}};state[id].note=value;state[id].updated_at=new Date().toISOString();localStorage.setItem(KEY,JSON.stringify(state));updateSummary()}}
function updateSummary(){{let a=0,r=0,c=0;Object.values(state).forEach(x=>{{a+=x.decision==='approved';r+=x.decision==='rejected';c+=x.decision==='correction'}});document.getElementById('summary').textContent=`${{EVENTS.length}} 次抓取 · 57 个视角 · 342 张阶段图 · 已批准 ${{a}} · 拒绝 ${{r}} · 待修正 ${{c}}`;}}
function visible(e){{const d=(state[e.event_id]||{{}}).decision||'pending';if(filter==='all')return true;if(filter==='problem')return e.priority==='warning';if(filter==='approved')return d==='approved';return d==='pending'}}
function modelPanels(e){{const q=e.qwen.annotation||{{}};const c=e.codex;const qchoice=e.qwen.valid_choice?`${{q.camera}} / 候选 #${{q.candidate_id}} / 置信度 ${{q.confidence}}`:'候选字段未通过校验';return `<div class="models"><section class="model qwen"><h3>Qwen3.5-2B 多视角时序标注</h3><dl><dt>选择</dt><dd>${{esc(qchoice)}}</dd><dt>目标描述</dt><dd>${{esc(q.target_description)}}</dd><dt>时序证据</dt><dd>${{esc(q.temporal_evidence)}}</dd><dt>跨视角证据</dt><dd>${{esc(q.cross_view_evidence)}}</dd><dt>歧义与风险</dt><dd>${{esc(q.ambiguity)}}</dd></dl><div class="recommend">${{esc(q.recommendation)}}</div></section><section class="model codex"><h3>Codex 辅助时序视觉复核</h3><dl><dt>当前实例</dt><dd>${{esc(c.selected_target)}}</dd><dt>机器人信号</dt><dd>${{esc(c.robot_evidence)}}</dd><dt>RGB-D 证据</dt><dd>${{esc(c.rgbd_evidence)}}</dd><dt>SAM2 连贯性</dt><dd>${{esc(c.tracking_evidence)}}</dd><dt>模型对照</dt><dd>${{esc(c.qwen_comparison)}}</dd></dl><div class="recommend">${{esc(c.recommendation)}}</div></section></div>`}}
function evidence(e){{const masks=e.overlays.map(x=>`<figure class="figure ${{x.needs_review?'bad':''}}"><img loading="lazy" src="${{esc(x.url)}}" onclick="showImage(this.src,'SAM2 帧 ${{x.frame}}')"><figcaption>帧 ${{x.frame}} · mask ${{(100*x.area_ratio).toFixed(2)}}%${{x.needs_review?' · 异常':''}}</figcaption></figure>`).join('');return `<div class="evidence"><div><img class="candidate" loading="lazy" src="${{esc(e.candidate_collage_url)}}" onclick="showImage(this.src,'三视角候选框')"><div class="meta">当前候选：${{esc(e.candidate_label)}} · 融合分 ${{e.candidate_score==null?'不可用':e.candidate_score.toFixed(3)}}</div></div><div class="mask-strip">${{masks}}</div></div>`}}
function timeline(e){{const rows=e.temporal.cameras.map(c=>`<section class="camera-row"><h4>${{esc(c.label)}} · 初始化帧 ${{c.anchor}} · 释放帧 ${{c.release}}</h4><div class="strip-wrap"><div class="timeline-strip">${{c.stages.map(s=>`<figure class="figure"><img loading="lazy" src="${{esc(s.rgb_url)}}" onclick="showImage(this.src,'${{esc(c.label)}} · ${{esc(s.label)}} · 帧 ${{s.frame}}')"><figcaption>${{esc(s.label)}} · 帧 ${{s.frame}}</figcaption></figure>`).join('')}}</div></div></section>`).join('');return `<details><summary>三视角六阶段大图（18 张）</summary><div class="timeline">${{rows}}</div></details>`}}
function render(){{document.getElementById('events').innerHTML=EVENTS.filter(visible).map(e=>{{const s=state[e.event_id]||{{}};return `<article class="event ${{e.priority}}"><div class="event-head"><div><div class="event-title">${{esc(e.sequence_label)}} · 第 ${{e.event_id.match(/grasp_([0-9]+)/)?.[1]||'?'}} 次抓取 · ${{e.side_label}}</div><div class="meta"><span class="tag">${{e.camera}}</span><span class="tag">${{sourceName(e.selection_source)}}</span><span class="tag">候选 #${{e.candidate_id}}</span><span class="tag">${{e.tracked_frames}} 帧</span>初始化 ${{e.anchor_frame}}，释放 ${{e.release_frame}}，结束 ${{e.end_frame}}</div></div><div><b>${{e.needs_review_frames}}</b> 异常帧 / ${{(100*e.anomaly_ratio).toFixed(1)}}%</div></div>${{modelPanels(e)}}${{evidence(e)}}${{timeline(e)}}<div class="review"><button class="approve ${{s.decision==='approved'?'selected':''}}" onclick="decision('${{esc(e.event_id)}}','approved')">批准</button><button class="reject ${{s.decision==='rejected'?'selected':''}}" onclick="decision('${{esc(e.event_id)}}','rejected')">拒绝</button><button class="correct ${{s.decision==='correction'?'selected':''}}" onclick="decision('${{esc(e.event_id)}}','correction')">需要修正</button><input placeholder="审核备注" value="${{esc(s.note||'')}}" oninput="note('${{esc(e.event_id)}}',this.value)"></div></article>`}}).join('');updateSummary()}}
document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{{filter=b.dataset.filter;document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x===b));render()}});
document.getElementById('export').onclick=()=>{{const blob=new Blob([JSON.stringify({{schema:'detailed_interaction_review_export_v2',exported_at:new Date().toISOString(),decisions:state,events:EVENTS}},null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='detailed_interaction_review.json';a.click();URL.revokeObjectURL(a.href)}};render();
</script></body></html>"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"events": len(events), "qwen_annotations": len(qwen), "codex_audits": len(codex), "output": str(out_dir / "index.html")}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
