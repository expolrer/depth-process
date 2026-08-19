#!/usr/bin/env python3
"""Build 19 dual-view annotated grasp videos and a final local review page."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SEQUENCE_INFO = (
    ("leju_claw", "leju_claw", "玩具堆 / 夹爪", 0),
    ("dex_hand", "dex_hand", "灵巧手物体搬运", 1),
    ("chengzhong", "chengzhong", "称重", 2),
    ("dajian", "dajian", "大件搬运", 3),
    ("zhoumian", "zhoumian", "桌面整理", 4),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sequence_info(sequence: str) -> tuple[str, str, int]:
    lowered = sequence.lower()
    for needle, slug, label, order in SEQUENCE_INFO:
        if needle in lowered:
            return slug, label, order
    slug = "".join(char if char.isalnum() else "-" for char in lowered).strip("-")
    return slug or "sequence", sequence, len(SEQUENCE_INFO)


def probe(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=width,height,nb_read_frames,duration",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        return None
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream.get("nb_read_frames") or 0),
        "duration": float(stream.get("duration") or 0),
    }


def render_panel(root: Path, row: dict[str, Any], label: str) -> np.ndarray:
    frame_index = int(row["frame_index"])
    rgb_path = (
        root / "outputs" / "extracted" / row["sequence"] / row["camera"]
        / "rgb_raw" / f"{frame_index:06d}.jpg"
    )
    mask_path = root / row["mask_path"]
    frame = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise FileNotFoundError(rgb_path)
    if mask is None:
        raise FileNotFoundError(mask_path)
    if frame.shape[:2] != (480, 848):
        frame = cv2.resize(frame, (848, 480), interpolation=cv2.INTER_AREA)
    if mask.shape != (480, 848):
        mask = cv2.resize(mask, (848, 480), interpolation=cv2.INTER_NEAREST)
    selected = mask > 0
    if np.any(selected):
        tinted = frame.copy()
        tinted[selected] = (0, 212, 255)
        frame = cv2.addWeighted(frame, 0.68, tinted, 0.32, 0)
    bbox = row.get("bbox_xyxy")
    if bbox:
        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 212, 255), 3, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (848, 38), (12, 16, 17), -1)
    title = f"{label} | {row['camera']} | F{frame_index:06d}"
    cv2.putText(
        frame, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.64,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    return frame


def encode_event(
    root: Path,
    output_dir: Path,
    event: dict[str, Any],
    fps: float,
    crf: int,
) -> dict[str, Any]:
    frame_count = min(len(event["head_rows"]), len(event["wrist_rows"]))
    head_start = int(event["head_rows"][0]["frame_index"])
    wrist_start = int(event["wrist_rows"][0]["frame_index"])

    video_path = output_dir / "media" / f"{event['slug']}.mp4"
    poster_path = output_dir / "posters" / f"{event['slug']}.jpg"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    poster_path.parent.mkdir(parents=True, exist_ok=True)
    existing = probe(video_path)
    status = "reused"
    if not existing or existing["frames"] != frame_count or existing["width"] != 1696:
        status = "encoded"
        temporary = video_path.with_suffix(".part.mp4")
        temporary.unlink(missing_ok=True)
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s:v", "1696x480",
            "-r", str(fps), "-i", "-", "-frames:v", str(frame_count),
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "2",
            "-y", str(temporary),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        try:
            for head_row, wrist_row in zip(event["head_rows"], event["wrist_rows"]):
                head = render_panel(root, head_row, "HEAD")
                wrist = render_panel(root, wrist_row, "ACTIVE WRIST")
                process.stdin.write(np.concatenate((head, wrist), axis=1).tobytes())
            process.stdin.close()
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            returncode = process.wait()
        except Exception:
            process.kill()
            process.wait()
            temporary.unlink(missing_ok=True)
            raise
        if returncode:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(stderr.strip() or f"ffmpeg exited {returncode}")
        temporary.replace(video_path)
        existing = probe(video_path)
        if not existing or existing["frames"] != frame_count:
            raise RuntimeError(f"Invalid encoded video: {video_path}: {existing}")

    if not poster_path.is_file() or poster_path.stat().st_mtime < video_path.stat().st_mtime:
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
                "-frames:v", "1", "-q:v", "2", "-y", str(poster_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f"poster ffmpeg exited {result.returncode}")

    return {
        "eventId": event["event_id"],
        "slug": event["slug"],
        "sequence": event["sequence"],
        "sequenceSlug": event["sequence_slug"],
        "sequenceLabel": event["sequence_label"],
        "sequenceOrder": event["sequence_order"],
        "eventIndex": event["event_index"],
        "side": event["side"],
        "sideLabel": "左手" if event["side"] == "left" else "右手",
        "headCamera": event["head_camera"],
        "wristCamera": event["wrist_camera"],
        "headStartFrame": head_start,
        "wristStartFrame": wrist_start,
        "frameCount": frame_count,
        "durationSeconds": frame_count / fps,
        "video": video_path.relative_to(output_dir).as_posix(),
        "poster": poster_path.relative_to(output_dir).as_posix(),
        "bytes": video_path.stat().st_size,
        "status": status,
    }


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def build_page(catalog: dict[str, Any]) -> str:
    payload = json.dumps(catalog["events"], ensure_ascii=False).replace("</", "<\\/")
    return PAGE.replace("__EVENTS__", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or root / "outputs" / "final_interaction_video_review").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(root / "outputs" / "target_tracks_required" / "track_index.jsonl")
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(row["event_id"], {}).setdefault(row["role"], []).append(row)

    jobs = []
    for event_id, roles in grouped.items():
        if set(roles) != {"head", "wrist"}:
            raise RuntimeError(f"Missing required role for {event_id}: {sorted(roles)}")
        head_rows = sorted(roles["head"], key=lambda row: int(row["frame_index"]))
        wrist_rows = sorted(roles["wrist"], key=lambda row: int(row["frame_index"]))
        first = head_rows[0]
        sequence = first["sequence"]
        sequence_slug, sequence_label, sequence_order = sequence_info(sequence)
        event_index = int(first["event_index"])
        side = first["side"]
        jobs.append({
            "event_id": event_id,
            "slug": f"{sequence_slug}_grasp_{event_index:03d}_{side}",
            "sequence": sequence,
            "sequence_slug": sequence_slug,
            "sequence_label": sequence_label,
            "sequence_order": sequence_order,
            "event_index": event_index,
            "side": side,
            "head_camera": first["camera"],
            "wrist_camera": wrist_rows[0]["camera"],
            "head_rows": head_rows,
            "wrist_rows": wrist_rows,
        })
    jobs.sort(key=lambda row: (row["sequence_order"], row["event_index"]))

    built = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(encode_event, root, output_dir, job, args.fps, args.crf): job
            for job in jobs
        }
        for index, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                result = future.result()
                built.append(result)
                print(f"[{index}/{len(jobs)}] {result['status']} {result['slug']}", flush=True)
            except Exception as error:
                failures.append(f"{job['event_id']}: {error}")
                print(f"[{index}/{len(jobs)}] FAILED {job['event_id']}: {error}", flush=True)
    if failures:
        raise RuntimeError("\n".join(failures))
    built.sort(key=lambda row: (row["sequenceOrder"], row["eventIndex"]))

    generated_at = datetime.now(timezone.utc).isoformat()
    catalog = {
        "schema": "final_dual_view_interaction_video_review_v1",
        "generatedAt": generated_at,
        "fps": args.fps,
        "events": built,
    }
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = data_dir / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    approval = {
        "schema": "approved_dual_view_image_review_v1",
        "confirmedAt": generated_at,
        "decision": "approved",
        "eventCount": len(built),
        "eventIds": [event["eventId"] for event in built],
        "source": "用户确认 interaction_review v5 的 19 次抓取全部批准",
    }
    approval_path = data_dir / "approved_image_review.json"
    approval_path.write_text(json.dumps(approval, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "index.html").write_text(build_page(catalog), encoding="utf-8")

    manifest_paths = [output_dir / "index.html", catalog_path, approval_path]
    manifest_paths.extend(output_dir / event["video"] for event in built)
    manifest_paths.extend(output_dir / event["poster"] for event in built)
    manifest = output_dir / "manifest.sha256"
    manifest.write_text(
        "".join(
            f"{digest(path)}  {path.relative_to(output_dir).as_posix()}\n"
            for path in sorted(manifest_paths)
        ),
        encoding="ascii",
    )
    print(json.dumps({
        "events": len(built),
        "frames": sum(event["frameCount"] for event in built),
        "videoBytes": sum(event["bytes"] for event in built),
        "output": str(output_dir),
    }, ensure_ascii=False))


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>19 次抓取双视角标注视频终审</title>
<style>
:root{--ink:#172126;--muted:#65747b;--line:#d4dcdf;--paper:#eef1f1;--panel:#fff;--green:#17675c;--red:#a12d34;--yellow:#f5c400}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"Microsoft YaHei",sans-serif;letter-spacing:0}header{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid var(--line);padding:12px 18px}.head{max-width:1800px;margin:auto;display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap}.title h1{font-size:20px;margin:0}.summary{font-size:13px;color:var(--muted);margin-top:4px}.toolbar{display:flex;gap:7px;flex-wrap:wrap}button{height:34px;border:1px solid #b7c1c5;background:#fff;border-radius:6px;padding:0 12px;cursor:pointer;font-weight:700;color:var(--ink)}button.active{background:var(--green);border-color:var(--green);color:#fff}main{max-width:1800px;margin:16px auto 60px;padding:0 16px;display:grid;gap:14px}.event{background:var(--panel);border:1px solid var(--line);border-left:5px solid #829095;border-radius:7px;overflow:hidden}.event.approved{border-left-color:var(--green)}.event.correction{border-left-color:var(--red)}.event-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;padding:12px 14px;border-bottom:1px solid var(--line)}.event-title{font-size:16px;font-weight:760}.meta{font-size:12px;color:var(--muted);margin-top:4px;overflow-wrap:anywhere}.status{font-size:12px;font-weight:750;padding:3px 7px;border:1px solid var(--line);border-radius:4px;white-space:nowrap}.video-wrap{background:#090d0e}.video-wrap video{display:block;width:100%;aspect-ratio:1696/480;background:#090d0e}.controls{display:grid;grid-template-columns:auto auto minmax(260px,1fr);gap:8px;padding:11px 14px}.controls .approve.selected{background:var(--green);border-color:var(--green);color:#fff}.controls .correct.selected{background:var(--red);border-color:var(--red);color:#fff}input{height:34px;border:1px solid #b7c1c5;border-radius:5px;padding:0 10px;width:100%;min-width:0}@media(max-width:760px){.event-head{flex-direction:column}.controls{grid-template-columns:1fr 1fr}.controls input{grid-column:1/-1}}
</style></head><body>
<header><div class="head"><div class="title"><h1>19 次抓取双视角标注视频终审</h1><div class="summary" id="summary"></div></div><div class="toolbar"><button data-filter="all" class="active">全部</button><button data-filter="pending">待审核</button><button data-filter="approved">已通过</button><button data-filter="correction">需修正</button><button id="export">导出终审 JSON</button></div></div></header>
<main id="events"></main>
<script>
const EVENTS=__EVENTS__;const KEY='final-interaction-video-review-v1';let state=JSON.parse(localStorage.getItem(KEY)||'{}');let filter='all';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function decision(id,value){state[id]=state[id]||{};state[id].decision=value;state[id].updatedAt=new Date().toISOString();save()}
function note(id,value){state[id]=state[id]||{};state[id].note=value;state[id].updatedAt=new Date().toISOString();localStorage.setItem(KEY,JSON.stringify(state));updateSummary()}
function save(){localStorage.setItem(KEY,JSON.stringify(state));render()}
function visible(event){const value=(state[event.eventId]||{}).decision||'pending';return filter==='all'||value===filter}
function updateSummary(){let approved=0,correction=0;Object.values(state).forEach(row=>{approved+=row.decision==='approved';correction+=row.decision==='correction'});document.getElementById('summary').textContent=`${EVENTS.length} 次抓取 · 双视角逐帧 SAM2 标注 · 已通过 ${approved} · 需修正 ${correction} · 待审核 ${EVENTS.length-approved-correction}`}
function render(){document.getElementById('events').innerHTML=EVENTS.filter(visible).map(event=>{const review=state[event.eventId]||{};const value=review.decision||'pending';return `<article class="event ${value}"><div class="event-head"><div><div class="event-title">${esc(event.sequenceLabel)} · 第 ${event.eventIndex+1} 次抓取 · ${esc(event.sideLabel)}</div><div class="meta">头部 ${esc(event.headCamera)} + 执行腕 ${esc(event.wristCamera)} · ${event.frameCount} 帧 · ${event.durationSeconds.toFixed(1)} 秒 · ${esc(event.eventId)}</div></div><span class="status">${value==='approved'?'已通过':value==='correction'?'需修正':'待审核'}</span></div><div class="video-wrap"><video controls preload="metadata" poster="${esc(event.poster)}"><source src="${esc(event.video)}" type="video/mp4"></video></div><div class="controls"><button class="approve ${value==='approved'?'selected':''}" onclick="decision('${esc(event.eventId)}','approved')">通过</button><button class="correct ${value==='correction'?'selected':''}" onclick="decision('${esc(event.eventId)}','correction')">需修正</button><input placeholder="终审备注" value="${esc(review.note||'')}" oninput="note('${esc(event.eventId)}',this.value)"></div></article>`}).join('');updateSummary()}
document.querySelectorAll('[data-filter]').forEach(button=>button.onclick=()=>{filter=button.dataset.filter;document.querySelectorAll('[data-filter]').forEach(item=>item.classList.toggle('active',item===button));render()});
document.getElementById('export').onclick=()=>{const blob=new Blob([JSON.stringify({schema:'final_dual_view_interaction_video_review_export_v1',exportedAt:new Date().toISOString(),decisions:state,events:EVENTS},null,2)],{type:'application/json'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='final_interaction_video_review.json';link.click();URL.revokeObjectURL(link.href)};render();
</script></body></html>'''


if __name__ == "__main__":
    main()
