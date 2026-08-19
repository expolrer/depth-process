#!/usr/bin/env python3
"""Export approved dual-view ACT attention keyframes from full-sequence videos."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CAMERAS = ("cam_h", "cam_l", "cam_r")
METHODS = (
    "raw_aligned",
    "rgb_guided",
    "temporal_rgb_guided",
    "lingbot_v05",
    "depth_anything_v2_fused",
    "lingbot_v05_sensor_fused",
    "ai_consensus_fused",
)
METHOD_ZH = {
    "raw_aligned": "原始对齐深度",
    "rgb_guided": "RGB 引导",
    "temporal_rgb_guided": "时序 RGB 引导",
    "lingbot_v05": "LingBot-Depth v0.5",
    "depth_anything_v2_fused": "Depth Anything V2 融合",
    "lingbot_v05_sensor_fused": "LingBot + 传感器融合",
    "ai_consensus_fused": "AI 共识融合",
}
SEQUENCE_SLUGS = {
    "A10-A15-G-S-01-TQ_03_01-4_304-leju_claw-20260512172205-200049-8c435b-v003": "leju_claw",
    "A10-A15-G-S-01-TQ_09_01-P4_297-dex_hand-20260629103123-49-9a5c2d-v003": "dex_hand",
    "chengzhong_xianxia_main1": "chengzhong",
    "dajian_xianxia_main1": "dajian",
    "zhoumian_xianxia_main1": "zhoumian",
}
SEQUENCE_ZH = {
    "leju_claw": "玩具堆 / 夹爪",
    "dex_hand": "灵巧手物体搬运",
    "chengzhong": "称重",
    "dajian": "大件搬运",
    "zhoumian": "桌面整理",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def frame_number(path: str) -> int:
    return int(Path(path).stem)


def resolve(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def nearest_track(
    rows: list[dict[str, Any]], frame: int, root: Path
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: abs(int(row["frame_index"]) - frame))
    for row in ordered:
        mask = cv2.imread(str(resolve(root, row["mask_path"])), cv2.IMREAD_GRAYSCALE)
        if mask is not None and np.any(mask):
            return row
    raise RuntimeError(f"No visible approved mask near frame {frame}")


def read_video_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index} from {path}")
    return frame


def draw_roi(image: np.ndarray, mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or not np.any(mask):
        return image
    resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours((resized > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()
    cv2.drawContours(output, contours, -1, (0, 220, 255), 3, cv2.LINE_AA)
    return output


def label_panel(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    bar = np.full((52, image.shape[1], 3), (246, 247, 249), dtype=np.uint8)
    cv2.putText(bar, title, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (24, 31, 38), 1, cv2.LINE_AA)
    cv2.putText(bar, subtitle, (12, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (80, 93, 106), 1, cv2.LINE_AA)
    return np.vstack([bar, image])


def metric_text(metric: dict[str, Any] | None) -> str:
    if not metric:
        return "ROI metrics unavailable"
    return (
        f"ROI lift {metric['roi_attention_lift']:.3f} | "
        f"ROI mass {metric['roi_within_camera_attention_mass'] * 100:.2f}% | "
        f"centroid {metric['centroid_distance_normalized']:.3f}"
    )


def build_image(
    root: Path,
    act_row: dict[str, Any],
    video_frame: np.ndarray,
    head_track: dict[str, Any],
    wrist_track: dict[str, Any],
    method: str,
    metrics: dict[tuple[str, int, str, str], dict[str, Any]],
) -> np.ndarray:
    panel_width = video_frame.shape[1] // 3
    panel_height = video_frame.shape[0]
    columns = []
    for role, track in (("head", head_track), ("wrist", wrist_track)):
        camera = track["camera"]
        camera_index = CAMERAS.index(camera)
        attention = video_frame[:, camera_index * panel_width : (camera_index + 1) * panel_width]
        attention = cv2.resize(attention, (640, 480), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.imread(act_row["rgb_paths"][camera], cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"Failed to read {act_row['rgb_paths'][camera]}")
        rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)
        mask_path = resolve(root, track["mask_path"])
        rgb = draw_roi(rgb, mask_path)
        attention = draw_roi(attention, mask_path)
        metric = metrics.get((track["event_id"], int(act_row["frame_position"]), role, method))
        raw = label_panel(rgb, f"{role.upper()} {camera} | RGB + approved ROI", f"frame {track['frame_index']}")
        heat = label_panel(attention, f"{role.upper()} {camera} | ACT depth attention", metric_text(metric))
        columns.append(np.hstack([raw, heat]))
    canvas = np.vstack(columns)
    divider_x = canvas.shape[1] // 2
    cv2.line(canvas, (divider_x, 0), (divider_x, canvas.shape[0]), (205, 211, 218), 2)
    cv2.line(canvas, (0, canvas.shape[0] // 2), (canvas.shape[1], canvas.shape[0] // 2), (205, 211, 218), 2)
    return canvas


def render_html(records: list[dict[str, Any]], output: Path) -> None:
    data = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    method_options = "".join(
        f'<option value="{method}">{html.escape(METHOD_ZH[method])}</option>' for method in METHODS
    )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ACT 双视角注意力热图</title>
<style>
:root{{--ink:#182129;--muted:#66727b;--line:#d6dde1;--paper:#f4f6f7;--green:#17675c}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:Arial,"Microsoft YaHei",sans-serif;letter-spacing:0}}
header{{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid var(--line);padding:14px 18px}}
.head{{max-width:1680px;margin:auto;display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap}}h1{{font-size:22px;margin:0 0 4px}}p{{margin:0;color:var(--muted);font-size:13px}}
.tools{{display:flex;gap:8px;flex-wrap:wrap}}select{{height:36px;border:1px solid #aeb9bf;border-radius:5px;background:#fff;padding:0 32px 0 10px;font-weight:700}}
main{{max-width:1680px;margin:18px auto 60px;padding:0 16px;display:grid;gap:14px}}.event{{background:#fff;border:1px solid var(--line);border-radius:7px;overflow:hidden}}.event h2{{font-size:16px;margin:0;padding:12px 14px;border-bottom:1px solid var(--line)}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1px;background:var(--line)}}figure{{margin:0;background:#fff;padding:10px}}figure img{{display:block;width:100%;aspect-ratio:1280/1064;object-fit:contain;background:#101416;cursor:zoom-in}}figcaption{{font-size:13px;font-weight:700;padding-top:8px}}
.modal{{position:fixed;inset:0;z-index:20;background:rgba(8,12,14,.94);display:none;align-items:center;justify-content:center}}.modal.open{{display:flex}}.modal img{{max-width:calc(100vw - 120px);max-height:calc(100vh - 80px)}}button{{position:absolute;width:44px;height:44px;border:1px solid #87939a;background:#fff;border-radius:50%;font-size:26px;cursor:pointer}}#prev{{left:18px}}#next{{right:18px}}#close{{right:18px;top:14px;font-size:20px}}
@media(max-width:820px){{.grid{{grid-template-columns:1fr}}.modal img{{max-width:100vw;max-height:calc(100vh - 110px)}}#prev,#next{{bottom:16px;top:auto}}}}
</style></head><body><header><div class="head"><div><h1>ACT 双视角注意力热图</h1><p>接触关键帧；左列 RGB，右列注意力；黄色轮廓为最终批准目标 ROI。</p></div><div class="tools"><select id="sequence"><option value="all">全部数据集</option></select><select id="method">{method_options}</select></div></div></header>
<main id="content"></main><div class="modal" id="modal"><button id="close" title="关闭">×</button><button id="prev" title="上一张">‹</button><img id="large" alt="注意力热图"><button id="next" title="下一张">›</button></div>
<script>const DATA={data};const METHOD_ZH={json.dumps(METHOD_ZH, ensure_ascii=False)};const SEQ_ZH={json.dumps(SEQUENCE_ZH, ensure_ascii=False)};const seq=document.getElementById('sequence'),method=document.getElementById('method'),content=document.getElementById('content');[...new Set(DATA.map(x=>x.sequenceSlug))].forEach(x=>seq.insertAdjacentHTML('beforeend',`<option value="${{x}}">${{SEQ_ZH[x]}}</option>`));let visible=[],active=0;
function render(){{const rows=DATA.filter(x=>(seq.value==='all'||x.sequenceSlug===seq.value)&&x.method===method.value);visible=rows;const groups=Object.groupBy?Object.groupBy(rows,x=>x.eventId):rows.reduce((a,x)=>((a[x.eventId]??=[]).push(x),a),{{}});content.innerHTML=Object.values(groups).map(items=>`<section class="event"><h2>${{SEQ_ZH[items[0].sequenceSlug]}} · 第 ${{items[0].eventIndex+1}} 次抓取 · ${{items[0].side==='left'?'左手':'右手'}}</h2><div class="grid">${{items.map((x,i)=>`<figure><img loading="lazy" src="${{x.image}}" data-path="${{x.image}}"><figcaption>${{METHOD_ZH[x.method]}} · Head F${{x.headFrame}} · Wrist F${{x.wristFrame}}</figcaption></figure>`).join('')}}</div></section>`).join('');document.querySelectorAll('figure img').forEach((img,i)=>img.onclick=()=>openAt(i))}}
function openAt(i){{active=(i+visible.length)%visible.length;large.src=visible[active].image;modal.classList.add('open')}}const modal=document.getElementById('modal'),large=document.getElementById('large');document.getElementById('close').onclick=()=>modal.classList.remove('open');document.getElementById('prev').onclick=()=>openAt(active-1);document.getElementById('next').onclick=()=>openAt(active+1);modal.onclick=e=>{{if(e.target===modal)modal.classList.remove('open')}};document.onkeydown=e=>{{if(!modal.classList.contains('open'))return;if(e.key==='ArrowLeft')openAt(active-1);if(e.key==='ArrowRight')openAt(active+1);if(e.key==='Escape')modal.classList.remove('open')}};seq.onchange=render;method.onchange=render;method.value='lingbot_v05';render();</script></body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir.resolve()
    (output / "images").mkdir(parents=True, exist_ok=True)

    act_rows = read_jsonl(args.index)
    act_by_head = {
        (row["sequence"], frame_number(row["rgb_paths"]["cam_h"])): row for row in act_rows
    }
    tracks = read_jsonl(args.tracks)
    event_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tracks:
        event_rows[row["event_id"]].append(row)

    metrics: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for method in METHODS:
        for row in read_jsonl(args.metrics_dir / "frames" / f"{method}.jsonl"):
            metrics[(row["event_id"], int(row["head_frame"]), row["role"], method)] = row

    records = []
    ordered_events = sorted(
        event_rows.items(), key=lambda item: (SEQUENCE_SLUGS[item[1][0]["sequence"]], int(item[1][0]["event_index"]))
    )
    for event_id, rows in ordered_events:
        head_rows = [row for row in rows if row["role"] == "head"]
        wrist_rows = [row for row in rows if row["role"] == "wrist"]
        head_anchor = int(head_rows[0]["anchor_frame_index"])
        act_row = act_by_head[(head_rows[0]["sequence"], head_anchor)]
        head_track = nearest_track(head_rows, head_anchor, root)
        wrist_camera = wrist_rows[0]["camera"]
        wrist_frame = frame_number(act_row["rgb_paths"][wrist_camera])
        wrist_track = nearest_track(wrist_rows, wrist_frame, root)
        sequence_slug = SEQUENCE_SLUGS[head_rows[0]["sequence"]]
        event_slug = safe_slug(event_id.split(":")[-1])

        for method in METHODS:
            video_path = args.video_dir / sequence_slug / "attention" / f"{method}.mp4"
            video_frame = read_video_frame(video_path, int(act_row["frame_position"]))
            image = build_image(root, act_row, video_frame, head_track, wrist_track, method, metrics)
            relative = Path("images") / sequence_slug / f"{event_slug}__{method}.jpg"
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f"Failed to write {destination}")
            records.append(
                {
                    "eventId": event_id,
                    "eventIndex": int(head_rows[0]["event_index"]),
                    "sequenceSlug": sequence_slug,
                    "side": head_rows[0]["side"],
                    "method": method,
                    "headFrame": int(head_track["frame_index"]),
                    "wristFrame": int(wrist_track["frame_index"]),
                    "image": relative.as_posix(),
                }
            )
        print(f"{event_id}: {len(METHODS)} methods", flush=True)

    (output / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_html(records, output)
    checksums = []
    for path in sorted(candidate for candidate in output.rglob("*") if candidate.is_file() and candidate.name != "SHA256SUMS.txt"):
        checksums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output).as_posix()}")
    (output / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="ascii")
    print(json.dumps({"events": len(ordered_events), "images": len(records), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
