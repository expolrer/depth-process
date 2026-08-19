#!/usr/bin/env python3
"""Build the head plus active-wrist six-stage annotation review page."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


STAGES = (
    ("动作开始", "start"),
    ("接触前", "pre_contact"),
    ("接触/初始化", "contact"),
    ("搬运中", "transport"),
    ("释放", "release"),
    ("动作结束", "end"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def output_url(value: str) -> str:
    value = value.replace("\\", "/")
    if "/outputs/" in value:
        value = value.split("/outputs/", 1)[1]
    elif value.startswith("outputs/"):
        value = value[len("outputs/") :]
    return "../" + value.lstrip("/")


def sequence_label(value: str) -> str:
    labels = {
        "chengzhong_xianxia_main1": "称重",
        "dajian_xianxia_main1": "大件",
        "zhoumian_xianxia_main1": "桌面",
    }
    if value in labels:
        return labels[value]
    if "leju_claw" in value:
        return "玩具堆 / 夹爪"
    if "dex_hand" in value:
        return "玩具堆 / 灵巧手"
    return value


def stage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: int(row["frame_index"]))
    first = rows[0]
    anchor = int(first["anchor_frame_index"])
    release = int(first["release_frame_index"])
    targets = (
        int(first["start_frame_index"]),
        max(int(first["start_frame_index"]), anchor - 10),
        anchor,
        round((anchor + release) / 2),
        release,
        int(first["end_frame_index"]),
    )
    picked = []
    for (label, key), target in zip(STAGES, targets):
        row = min(rows, key=lambda item: abs(int(item["frame_index"]) - target))
        status = row.get("visibility_status")
        if float(row.get("area_ratio", 0)) > 0:
            status_label = "黄色框：目标可见"
        elif status == "unobservable":
            status_label = "无框：目标不可辨认"
        elif status == "post_release_out_of_view":
            status_label = "无框：释放后离开视野"
        else:
            status_label = "无框：需要修正"
        picked.append({
            "label": label,
            "key": key,
            "frame": int(row["frame_index"]),
            "url": output_url(row["overlay_path"]),
            "area_ratio": float(row.get("area_ratio", 0)),
            "bbox_xyxy": row.get("bbox_xyxy"),
            "status": status,
            "status_label": status_label,
            "visibility_reason": row.get("visibility_reason"),
            "needs_review": bool(row.get("needs_review")),
        })
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    outputs = args.root.resolve() / "outputs"
    selections = read_jsonl(outputs / "interaction_review" / "required_view_selections.jsonl")
    tracks = read_jsonl(outputs / "target_tracks_required" / "track_index.jsonl")
    qwen_rows = read_jsonl(outputs / "interaction_review" / "qwen_required_views.jsonl")
    audit_rows = read_jsonl(outputs / "interaction_review" / "codex_required_view_audits.jsonl")
    summary = json.loads((outputs / "target_tracks_required" / "summary.json").read_text(encoding="utf-8"))

    tracks_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in tracks:
        tracks_by_key[(row["event_id"], row["role"])].append(row)
    qwen_by_event = {row["event_id"]: row for row in qwen_rows}
    audit_by_event = {row["event_id"]: row for row in audit_rows}
    selections_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selections:
        selections_by_event[row["event_id"]].append(row)

    events = []
    for event_id, event_selections in selections_by_event.items():
        sequence = event_id.split(":grasp_", 1)[0]
        views = []
        for selection in sorted(event_selections, key=lambda row: row["role"] != "head"):
            rows = tracks_by_key[(event_id, selection["role"])]
            views.append({
                **selection,
                "role_label": "头部相机" if selection["role"] == "head" else "执行手腕相机",
                "stages": stage_rows(rows),
                "tracked_frames": len(rows),
                "flagged_frames": sum(bool(row.get("needs_review")) for row in rows),
                "unobservable_frames": sum(row.get("visibility_status") == "unobservable" for row in rows),
                "tracking_failure_frames": sum(row.get("visibility_status") == "tracking_failure" for row in rows),
            })
        audit = audit_by_event[event_id]
        events.append({
            "event_id": event_id,
            "sequence": sequence,
            "sequence_label": sequence_label(sequence),
            "event_index": int(event_id.split(":grasp_", 1)[1].split("_", 1)[0]),
            "side": "left" if event_id.endswith("_left") else "right",
            "side_label": "左手" if event_id.endswith("_left") else "右手",
            "views": views,
            "qwen": qwen_by_event.get(event_id, {}),
            "codex": audit,
            "priority": "problem" if audit["status"] == "needs_correction" else ("review" if audit["status"] == "manual_review" else "normal"),
        })
    events.sort(key=lambda row: ({"problem": 0, "review": 1, "normal": 2}[row["priority"]], row["sequence"], row["event_index"]))

    out_dir = outputs / "interaction_review"
    manifest = {
        "schema": "dual_required_view_review_v5",
        "summary": {key: value for key, value in summary.items() if key != "views_detail"},
        "events": events,
    }
    (out_dir / "review_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = json.dumps(events, ensure_ascii=False).replace("</", "<\\/")
    page = PAGE.replace("__EVENTS__", payload)
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"events": len(events), "views": sum(len(row["views"]) for row in events), "stage_images": sum(len(view["stages"]) for row in events for view in row["views"]), "output": str(out_dir / "index.html")}, ensure_ascii=False))


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>双视角抓取目标标注复核</title>
<style>
:root{--ink:#172126;--muted:#66747b;--line:#d5dcdf;--paper:#f2f4f4;--panel:#fff;--green:#1d665d;--blue:#155e75;--yellow:#ffd400;--warn:#a35d12;--bad:#a12d34}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"Microsoft YaHei",sans-serif;letter-spacing:0;overflow-x:hidden}header{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid var(--line);padding:13px 20px}.head{max-width:1900px;margin:auto;display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap;min-width:0}h1{font-size:20px;margin:0}.summary{font-size:13px;color:var(--muted);margin-top:4px}.toolbar{display:flex;gap:7px;flex-wrap:wrap}button{height:34px;border:1px solid #b7c1c5;background:#fff;border-radius:6px;padding:0 12px;cursor:pointer;font-weight:650;color:var(--ink)}button.active{background:var(--green);border-color:var(--green);color:#fff}main{max-width:1900px;margin:16px auto;padding:0 16px 60px;display:grid;gap:15px;min-width:0}.event{background:var(--panel);border:1px solid var(--line);border-left:5px solid #7d8b90;border-radius:7px;overflow:hidden;min-width:0}.event.review{border-left-color:var(--warn)}.event.problem{border-left-color:var(--bad)}.event-head{display:flex;justify-content:space-between;gap:16px;padding:13px 16px;border-bottom:1px solid var(--line);min-width:0}.event-head>div{min-width:0}.event-title{font-size:17px;font-weight:760}.meta{font-size:12px;color:var(--muted);margin-top:5px;line-height:1.6;overflow-wrap:anywhere}.tag{display:inline-block;padding:2px 6px;border:1px solid var(--line);border-radius:4px;background:#f8f9f9;margin-right:5px}.legend{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--muted);min-width:0}.swatch{width:19px;height:14px;border:3px solid var(--yellow);background:rgba(255,212,0,.28);flex:0 0 auto}.models{display:grid;grid-template-columns:1fr 1fr;border-bottom:1px solid var(--line);min-width:0}.model{padding:14px 16px;min-width:0}.model+.model{border-left:1px solid var(--line)}.model h3{font-size:15px;margin:0 0 9px}.qwen h3{color:var(--blue)}.codex h3{color:#31633a}.model-grid{display:grid;grid-template-columns:110px minmax(0,1fr);gap:6px 10px;font-size:13px;line-height:1.55}.model-grid>div{min-width:0;overflow-wrap:anywhere}.term{font-weight:700;color:#45535a}.recommend{margin-top:9px;padding-top:8px;border-top:1px solid var(--line);font-size:13px;font-weight:700;overflow-wrap:anywhere}.view{padding:14px 16px;border-bottom:1px solid var(--line);min-width:0}.view-head{display:flex;justify-content:space-between;gap:14px;align-items:flex-start;margin-bottom:9px;min-width:0}.view-head>div{min-width:0}.view-title{font-size:15px;font-weight:760}.view-note{font-size:12px;color:var(--muted);margin-top:4px;line-height:1.55;overflow-wrap:anywhere}.strip-wrap{overflow-x:auto;padding-bottom:5px;max-width:100%}.strip{display:grid;grid-template-columns:repeat(6,minmax(245px,1fr));gap:8px;min-width:1500px}.figure{margin:0;background:#111719;border-radius:4px;overflow:hidden;position:relative;aspect-ratio:848/540}.figure img{width:100%;height:100%;display:block;object-fit:contain;cursor:zoom-in}.figure figcaption{position:absolute;left:0;right:0;bottom:0;background:rgba(5,8,9,.82);color:#fff;padding:7px 8px;font-size:11px;line-height:1.35}.figure.unobservable figcaption{border-top:3px solid var(--warn)}.figure.failure figcaption{border-top:3px solid var(--bad)}.review-controls{display:grid;grid-template-columns:auto auto auto 1fr;gap:8px;padding:12px 16px;align-items:center}.review-controls button.selected.approve{background:var(--green);color:#fff}.review-controls button.selected.reject{background:var(--bad);color:#fff}.review-controls button.selected.correct{background:var(--warn);color:#fff}input{height:34px;border:1px solid #b7c1c5;border-radius:5px;padding:0 10px;min-width:260px}dialog{border:0;border-radius:6px;padding:0;background:#0c1011;max-width:98vw;max-height:98vh;box-shadow:0 20px 80px rgba(0,0,0,.45)}dialog::backdrop{background:rgba(0,0,0,.8)}dialog img{display:block;max-width:96vw;max-height:91vh;object-fit:contain}.dialog-head{display:flex;justify-content:space-between;align-items:center;color:#fff;padding:7px 11px;font-size:13px}.dialog-head button{background:#20282b;color:#fff;border-color:#465156}@media(max-width:980px){.models{grid-template-columns:1fr}.model+.model{border-left:0;border-top:1px solid var(--line)}.view-head{flex-direction:column}.event-head{flex-direction:column}.review-controls{grid-template-columns:1fr 1fr}.review-controls input{grid-column:1/-1;width:100%}}
dialog .dialog-head{display:grid;grid-template-columns:38px minmax(0,1fr) auto 38px 38px;gap:7px;align-items:center;padding:7px 9px}.dialog-title{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.dialog-counter{color:#b9c4c8;font-variant-numeric:tabular-nums}.dialog-head button{width:38px;padding:0;font-size:20px;line-height:1}.dialog-head button:disabled{opacity:.32;cursor:default}@media(max-width:980px){dialog .dialog-head{grid-template-columns:36px minmax(0,1fr) auto 36px 36px}.dialog-head button{width:36px}}
</style></head><body>
<header><div class="head"><div><h1>双视角抓取目标标注复核</h1><div class="summary" id="summary"></div></div><div class="toolbar"><button data-filter="all" class="active">全部</button><button data-filter="pending">待审核</button><button data-filter="problem">异常优先</button><button data-filter="approved">已批准</button><button id="export">导出完整 JSON</button></div></div></header>
<main id="events"></main><dialog id="viewer"><div class="dialog-head"><button id="viewer-prev" title="上一张" aria-label="上一张" onclick="stepImage(-1)">←</button><span class="dialog-title" id="viewer-title"></span><span class="dialog-counter" id="viewer-counter"></span><button id="viewer-next" title="下一张" aria-label="下一张" onclick="stepImage(1)">→</button><button title="关闭" aria-label="关闭" onclick="document.getElementById('viewer').close()">×</button></div><img id="viewer-image"></dialog>
<script>
const EVENTS=__EVENTS__;const KEY='dual-view-interaction-review-v5';let state=JSON.parse(localStorage.getItem(KEY)||'{}');let filter='all';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sourceName=s=>({codex_dual_view_override:'Codex 双视角复核覆盖',codex_temporal_override:'Codex 时序复核覆盖',existing_verified_track:'沿用已验证轨迹',qwen_dual_view:'Qwen 双视角建议',qwen_single_view:'Qwen 单视角建议',automatic_rank_fallback:'自动候选排序'}[s]||s);
let viewerItems=[];let viewerIndex=-1;
function syncViewer(){const item=viewerItems[viewerIndex];if(!item)return;document.getElementById('viewer-image').src=item.src;document.getElementById('viewer-title').textContent=item.dataset.title||item.alt||'';document.getElementById('viewer-counter').textContent=`${viewerIndex+1} / ${viewerItems.length}`;document.getElementById('viewer-prev').disabled=viewerIndex<=0;document.getElementById('viewer-next').disabled=viewerIndex>=viewerItems.length-1}
function showImage(item){viewerItems=[...document.querySelectorAll('.figure img')];viewerIndex=viewerItems.indexOf(item);syncViewer();const viewer=document.getElementById('viewer');if(!viewer.open)viewer.showModal()}
function stepImage(offset){const next=viewerIndex+offset;if(next<0||next>=viewerItems.length)return;viewerIndex=next;syncViewer()}
function save(){localStorage.setItem(KEY,JSON.stringify(state));render()}function decision(id,value){state[id]=state[id]||{};state[id].decision=value;state[id].updated_at=new Date().toISOString();save()}function note(id,value){state[id]=state[id]||{};state[id].note=value;state[id].updated_at=new Date().toISOString();localStorage.setItem(KEY,JSON.stringify(state));updateSummary()}
function visible(e){const d=(state[e.event_id]||{}).decision||'pending';if(filter==='all')return true;if(filter==='problem')return e.priority!=='normal';if(filter==='approved')return d==='approved';return d==='pending'}
function updateSummary(){let a=0,r=0,c=0;Object.values(state).forEach(x=>{a+=x.decision==='approved';r+=x.decision==='rejected';c+=x.decision==='correction'});document.getElementById('summary').textContent=`${EVENTS.length} 次抓取 · 38 个必要视角 · 228 张六阶段标注图 · 已批准 ${a} · 拒绝 ${r} · 待修正 ${c}`}
function qwenPanel(e){const q=e.qwen.annotation||{};const role=(name,label,valid)=>{const x=q[name]||{};return `<div class="term">${label}</div><div>${valid?`${esc(x.camera)} / 候选 #${esc(x.candidate_id)} / 置信度 ${esc(x.confidence)}`:'候选编号未通过校验'}<br>${esc(x.evidence)}</div>`};return `<section class="model qwen"><h3>Qwen3.5-2B 双视角时序建议</h3><div class="model-grid">${role('head','头部视角',e.qwen.head_valid)}${role('wrist','执行腕视角',e.qwen.wrist_valid)}<div class="term">同一实例</div><div>${esc(q.same_instance_evidence)}</div><div class="term">歧义</div><div>${esc(q.ambiguity)}</div></div><div class="recommend">${esc(q.recommendation)}</div></section>`}
function codexPanel(e){const lines=e.codex.views.map(v=>`<div class="term">${v.role==='head'?'头部复核':'执行腕复核'}</div><div>${esc(v.camera)} / 最终候选 #${esc(v.candidate_id)} / ${sourceName(v.selection_source)}<br>Qwen 与最终选择${v.qwen_matches_final?'一致':'不一致'}；可见框 ${v.visible_box_frames}/${v.tracked_frames} 帧；不可辨认 ${v.unobservable_frames} 帧；漏跟踪 ${v.tracking_failure_frames} 帧。<br>${esc(v.selection_evidence)}</div>`).join('');return `<section class="model codex"><h3>Codex 辅助时序视觉复核</h3><div class="model-grid">${lines}<div class="term">视角规则</div><div>${esc(e.codex.same_instance_check)}</div></div><div class="recommend">${esc(e.codex.recommendation)}</div></section>`}
function viewSection(e,v){const figures=v.stages.map(s=>{const cls=s.status==='unobservable'||s.status==='post_release_out_of_view'?'unobservable':(s.status==='tracking_failure'?'failure':'');const detail=s.visibility_reason?` · ${esc(s.visibility_reason)}`:'';const title=`${v.role_label} · ${s.label} · 帧 ${s.frame}`;return `<figure class="figure ${cls}"><img loading="lazy" src="${esc(s.url)}" data-title="${esc(title)}" alt="${esc(title)}" onclick="showImage(this)"><figcaption><b>${esc(s.label)}</b> · 帧 ${s.frame}<br>${esc(s.status_label)} · mask ${(100*s.area_ratio).toFixed(2)}%${detail}</figcaption></figure>`}).join('');return `<section class="view"><div class="view-head"><div><div class="view-title">${esc(v.role_label)} · ${esc(v.camera)} · 候选 #${v.candidate_id}</div><div class="view-note">${sourceName(v.selection_source)}：${esc(v.evidence)}<br>共 ${v.tracked_frames} 帧；不可辨认 ${v.unobservable_frames} 帧；漏跟踪 ${v.tracking_failure_frames} 帧。</div></div><div class="legend"><span class="swatch"></span>黄色半透明区域为 SAM2 mask，黄色外框为 mask 外接框</div></div><div class="strip-wrap"><div class="strip">${figures}</div></div></section>`}
function render(){document.getElementById('events').innerHTML=EVENTS.filter(visible).map(e=>{const s=state[e.event_id]||{};return `<article class="event ${e.priority}"><div class="event-head"><div><div class="event-title">${esc(e.sequence_label)} · 第 ${e.event_index+1} 次抓取 · ${e.side_label}</div><div class="meta"><span class="tag">头部 + 执行腕</span><span class="tag">六阶段</span><span class="tag">${e.codex.status==='approved'?'自动检查通过':e.codex.status==='manual_review'?'建议人工抽检':'需要修正'}</span>${esc(e.event_id)}</div></div></div><div class="models">${qwenPanel(e)}${codexPanel(e)}</div>${e.views.map(v=>viewSection(e,v)).join('')}<div class="review-controls"><button class="approve ${s.decision==='approved'?'selected':''}" onclick="decision('${esc(e.event_id)}','approved')">批准</button><button class="reject ${s.decision==='rejected'?'selected':''}" onclick="decision('${esc(e.event_id)}','rejected')">拒绝</button><button class="correct ${s.decision==='correction'?'selected':''}" onclick="decision('${esc(e.event_id)}','correction')">需要修正</button><input placeholder="审核备注" value="${esc(s.note||'')}" oninput="note('${esc(e.event_id)}',this.value)"></div></article>`}).join('');updateSummary()}
document.addEventListener('keydown',event=>{const viewer=document.getElementById('viewer');if(!viewer.open)return;if(event.key==='ArrowLeft'){event.preventDefault();stepImage(-1)}else if(event.key==='ArrowRight'){event.preventDefault();stepImage(1)}});
document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{filter=b.dataset.filter;document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x===b));render()});document.getElementById('export').onclick=()=>{const blob=new Blob([JSON.stringify({schema:'dual_required_view_review_export_v5',exported_at:new Date().toISOString(),decisions:state,events:EVENTS},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='dual_view_interaction_review_v5.json';a.click();URL.revokeObjectURL(a.href)};render();
</script></body></html>'''


if __name__ == "__main__":
    main()
