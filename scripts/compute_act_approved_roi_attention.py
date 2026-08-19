#!/usr/bin/env python3
"""Measure prompt-free ACT depth attention against approved dual-view target masks."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from act_no_prompt_benchmark import (
    CAMERAS,
    CONTROLS,
    DEPTH_KEYS,
    NORMAL_METHODS,
    active_image_keys,
    closest_grid,
    load_depth,
    make_config,
)
from lerobot.policies.act.modeling_act import ACTPolicy


METRICS = (
    "camera_attention_mass",
    "roi_global_attention_mass",
    "roi_within_camera_attention_mass",
    "roi_area_ratio",
    "roi_attention_lift",
    "roi_background_density_ratio",
    "top_token_in_roi",
    "within_camera_attention_entropy",
    "centroid_distance_normalized",
)

METHOD_ZH = {
    "raw_aligned": "原始对齐深度",
    "rgb_guided": "RGB 引导",
    "temporal_rgb_guided": "时序 RGB 引导",
    "lingbot_v05": "LingBot-Depth v0.5",
    "depth_anything_v2_fused": "Depth Anything V2 融合",
    "lingbot_v05_sensor_fused": "LingBot + 传感器融合",
    "ai_consensus_fused": "AI 共识融合",
    "zero_depth": "零深度负对照",
    "spatially_shuffled_raw": "空间打乱负对照",
}

METRIC_ZH = {
    "camera_attention_mass": "视角注意力占比",
    "roi_global_attention_mass": "目标 ROI / 全部深度注意力",
    "roi_within_camera_attention_mass": "目标 ROI / 当前视角注意力",
    "roi_area_ratio": "目标 ROI 面积占比",
    "roi_attention_lift": "ROI 注意力提升倍数",
    "roi_background_density_ratio": "目标/背景注意力密度比",
    "top_token_in_roi": "注意力峰值命中率",
    "within_camera_attention_entropy": "视角内注意力熵",
    "centroid_distance_normalized": "注意力质心到目标距离",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def frame_from_path(path: str) -> int:
    return int(Path(path).stem)


def resolved(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def build_samples(
    act_rows: list[dict[str, Any]], tracks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    act_by_head_frame: dict[tuple[str, int], int] = {}
    for index, row in enumerate(act_rows):
        act_by_head_frame[(row["sequence"], frame_from_path(row["rgb_paths"]["cam_h"]))] = index

    by_event_camera_frame: dict[tuple[str, str, int], dict[str, Any]] = {}
    head_rows: list[dict[str, Any]] = []
    wrist_camera_by_event: dict[str, str] = {}
    for row in tracks:
        key = (row["event_id"], row["camera"], int(row["frame_index"]))
        by_event_camera_frame[key] = row
        if row.get("role") == "head":
            head_rows.append(row)
        elif row.get("role") == "wrist":
            wrist_camera_by_event[row["event_id"]] = row["camera"]

    samples: list[dict[str, Any]] = []
    missing_act = 0
    missing_wrist = 0
    for head in sorted(head_rows, key=lambda row: (row["event_id"], int(row["frame_index"]))):
        act_index = act_by_head_frame.get((head["sequence"], int(head["frame_index"])))
        if act_index is None:
            missing_act += 1
            continue
        act_row = act_rows[act_index]
        wrist_camera = wrist_camera_by_event.get(head["event_id"])
        views = [head]
        if wrist_camera:
            wrist_frame = frame_from_path(act_row["rgb_paths"][wrist_camera])
            wrist = by_event_camera_frame.get((head["event_id"], wrist_camera, wrist_frame))
            if wrist is None:
                missing_wrist += 1
            else:
                views.append(wrist)
        samples.append(
            {
                "act_index": act_index,
                "event_id": head["event_id"],
                "event_index": int(head["event_index"]),
                "sequence": head["sequence"],
                "head_frame": int(head["frame_index"]),
                "side": head["side"],
                "views": views,
            }
        )
    diagnostics = {
        "act_rows": len(act_rows),
        "track_rows": len(tracks),
        "head_track_rows": len(head_rows),
        "samples": len(samples),
        "events": len({sample["event_id"] for sample in samples}),
        "missing_act_rows": missing_act,
        "missing_wrist_rows": missing_wrist,
    }
    return samples, diagnostics


class ApprovedROIDataset(Dataset):
    def __init__(
        self,
        act_rows: list[dict[str, Any]],
        samples: list[dict[str, Any]],
        stats: dict[str, list[float]],
        method: str,
        width: int,
        height: int,
        min_depth_m: float,
        max_depth_m: float,
    ) -> None:
        self.act_rows = act_rows
        self.samples = samples
        self.method = method
        self.width = width
        self.height = height
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.state_mean = torch.tensor(stats["state_mean"], dtype=torch.float32)
        self.state_std = torch.tensor(stats["state_std"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.act_rows[self.samples[index]["act_index"]]
        sample: dict[str, Any] = {
            "observation.state": (
                torch.tensor(row["state"], dtype=torch.float32) - self.state_mean
            ) / self.state_std,
            "sample_index": index,
        }
        for key, camera in zip(DEPTH_KEYS, CAMERAS, strict=True):
            sample[key] = load_depth(
                row,
                camera,
                self.method,
                self.width,
                self.height,
                self.min_depth_m,
                self.max_depth_m,
            )
        return sample


def load_mask_fraction(root: Path, track: dict[str, Any], grid_w: int, grid_h: int) -> np.ndarray | None:
    mask = cv2.imread(str(resolved(root, track["mask_path"])), cv2.IMREAD_GRAYSCALE)
    if mask is None or not np.any(mask):
        return None
    return cv2.resize(mask.astype(np.float32) / 255.0, (grid_w, grid_h), interpolation=cv2.INTER_AREA)


def attention_metrics(values: np.ndarray, all_depth_mass: float, mask: np.ndarray) -> dict[str, float]:
    eps = 1e-12
    values = np.maximum(values.astype(np.float64), 0.0)
    camera_mass = float(values.sum())
    mask = np.clip(mask.astype(np.float64), 0.0, 1.0)
    roi_area = float(mask.mean())
    roi_mass = float((values * mask).sum())
    within = roi_mass / max(camera_mass, eps)
    lift = within / max(roi_area, eps)
    background_area = max(1.0 - roi_area, eps)
    background_mass = max(camera_mass - roi_mass, 0.0)
    density_ratio = (within / max(roi_area, eps)) / (
        (background_mass / max(camera_mass, eps)) / background_area + eps
    )
    probabilities = values.reshape(-1) / max(camera_mass, eps)
    entropy = float(
        -(probabilities * np.log(np.maximum(probabilities, eps))).sum()
        / math.log(max(probabilities.size, 2))
    )
    peak_y, peak_x = np.unravel_index(int(np.argmax(values)), values.shape)

    yy, xx = np.mgrid[0 : values.shape[0], 0 : values.shape[1]]
    target_total = float(mask.sum())
    target_x = float((xx * mask).sum() / max(target_total, eps))
    target_y = float((yy * mask).sum() / max(target_total, eps))
    attention_x = float((xx * values).sum() / max(camera_mass, eps))
    attention_y = float((yy * values).sum() / max(camera_mass, eps))
    diagonal = math.hypot(max(values.shape[1] - 1, 1), max(values.shape[0] - 1, 1))

    return {
        "camera_attention_mass": camera_mass / max(all_depth_mass, eps),
        "roi_global_attention_mass": roi_mass / max(all_depth_mass, eps),
        "roi_within_camera_attention_mass": within,
        "roi_area_ratio": roi_area,
        "roi_attention_lift": lift,
        "roi_background_density_ratio": density_ratio,
        "top_token_in_roi": float(mask[peak_y, peak_x] >= 0.5),
        "within_camera_attention_entropy": entropy,
        "centroid_distance_normalized": math.hypot(attention_x - target_x, attention_y - target_y)
        / diagonal,
    }


@torch.no_grad()
def evaluate_method(
    policy: ACTPolicy,
    act_rows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    stats: dict[str, list[float]],
    method: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    output_path = args.output_dir / "frames" / f"{method}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        print(f"[{method}] already exists: {output_path}", flush=True)
        return {"method": method, "skipped": True}

    dataset = ApprovedROIDataset(
        act_rows,
        samples,
        stats,
        method,
        args.width,
        args.height,
        args.min_depth_m,
        args.max_depth_m,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: tuple[torch.Tensor, torch.Tensor]) -> None:
        captured["weights"] = output[1].detach()

    handle = policy.model.decoder.layers[-1].multihead_attn.register_forward_hook(hook)
    policy.eval()
    written = 0
    skipped_masks = 0
    try:
        with output_path.open("w", encoding="utf-8") as output:
            for batch_index, batch in enumerate(loader):
                tensor_batch = {
                    key: batch[key].to(device, non_blocking=True)
                    for key in ("observation.state", *DEPTH_KEYS)
                }
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                    policy.predict_action_chunk(tensor_batch)
                weights = captured["weights"].float()
                if weights.ndim == 4:
                    weights = weights.mean(dim=1)
                query_mean = weights.mean(dim=1)
                token_count = (query_mean.shape[-1] - 2) // len(active_image_keys("depth_only"))
                grid_h, grid_w = closest_grid(token_count, args.width / args.height)
                maps = []
                for camera_index in range(3):
                    start = 2 + camera_index * token_count
                    maps.append(
                        query_mean[:, start : start + token_count]
                        .reshape(-1, grid_h, grid_w)
                        .cpu()
                        .numpy()
                    )

                for local_index, sample_index_tensor in enumerate(batch["sample_index"]):
                    sample_index = int(sample_index_tensor)
                    sample = samples[sample_index]
                    all_depth_mass = sum(float(camera_map[local_index].sum()) for camera_map in maps)
                    for track in sample["views"]:
                        mask = load_mask_fraction(args.root, track, grid_w, grid_h)
                        if mask is None:
                            skipped_masks += 1
                            continue
                        camera_index = CAMERAS.index(track["camera"])
                        record = {
                            "method": method,
                            "event_id": sample["event_id"],
                            "event_index": sample["event_index"],
                            "sequence": sample["sequence"],
                            "side": sample["side"],
                            "role": track["role"],
                            "camera": track["camera"],
                            "head_frame": sample["head_frame"],
                            "camera_frame": int(track["frame_index"]),
                            "attention_grid": [grid_h, grid_w],
                            **attention_metrics(maps[camera_index][local_index], all_depth_mass, mask),
                        }
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        written += 1
                if (batch_index + 1) % args.log_every == 0:
                    print(f"[{method}] batches={batch_index + 1}, views={written}", flush=True)
    finally:
        handle.remove()
    print(f"[{method}] complete: views={written}, empty_masks={skipped_masks}", flush=True)
    return {"method": method, "views": written, "empty_masks": skipped_masks}


def group_means(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for group_key, members in sorted(groups.items(), key=lambda item: item[0]):
        record = {key: value for key, value in zip(keys, group_key, strict=True)}
        record["views"] = len(members)
        for metric in METRICS:
            record[metric] = float(np.mean([member[metric] for member in members]))
        output.append(record)
    return output


def bootstrap_event_ci(values: list[float], seed: int = 20260814) -> tuple[float, float]:
    if len(values) < 2:
        return (values[0], values[0]) if values else (float("nan"), float("nan"))
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(5000, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_html(summary: dict[str, Any], output_path: Path) -> None:
    normal = [row for row in summary["by_method"] if row["method"] in NORMAL_METHODS]
    controls = [row for row in summary["by_method"] if row["method"] in CONTROLS]
    normal.sort(key=lambda row: row["event_balanced"]["roi_attention_lift"], reverse=True)
    rows_html = []
    for rank, row in enumerate(normal + controls, 1):
        event = row["event_balanced"]
        delta = row["delta_vs_raw_event_balanced"]["roi_attention_lift"]
        delta_ci = row["paired_delta_vs_raw_event_ci95"]["roi_attention_lift"]
        mae = row.get("chunk_mae_rad")
        cls = "control" if row["method"] in CONTROLS else ""
        rows_html.append(
            f"<tr class='{cls}'><td>{rank if not cls else '对照'}</td>"
            f"<td><b>{html.escape(METHOD_ZH[row['method']])}</b><small>{row['method']}</small></td>"
            f"<td>{event['roi_attention_lift']:.3f}<small>较原始 {delta:+.3f}；配对 95% CI {delta_ci[0]:+.3f} 至 {delta_ci[1]:+.3f}</small></td>"
            f"<td>{event['roi_background_density_ratio']:.3f}</td>"
            f"<td>{event['top_token_in_roi'] * 100:.1f}%</td>"
            f"<td>{event['roi_within_camera_attention_mass'] * 100:.2f}%</td>"
            f"<td>{event['centroid_distance_normalized']:.3f}</td>"
            f"<td>{mae:.4f}</td></tr>" if mae is not None else ""
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>不同深度处理方法的 ACT 目标注意力指标</title>
<style>
:root{{--ink:#17202a;--muted:#667085;--line:#d9dee7;--paper:#f7f8fa;--accent:#0f766e;--warn:#9a3412}}
*{{box-sizing:border-box}} body{{margin:0;font-family:Arial,"Microsoft YaHei",sans-serif;color:var(--ink);background:var(--paper);letter-spacing:0;overflow-x:hidden}}
header{{background:#fff;border-bottom:1px solid var(--line);padding:24px max(24px,calc((100% - 1440px)/2))}}
h1{{font-size:26px;margin:0 0 8px;overflow-wrap:anywhere}} header p,.note{{color:var(--muted);margin:0;line-height:1.65;overflow-wrap:anywhere}}
main{{max-width:1440px;margin:auto;padding:24px}} .summary{{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:1px;background:var(--line);border:1px solid var(--line);margin-bottom:20px}}
.summary div{{background:#fff;padding:16px;min-width:0;overflow-wrap:anywhere}} .summary b{{display:block;font-size:24px;color:var(--accent)}}
.table-wrap{{overflow:auto;background:#fff;border:1px solid var(--line)}} table{{width:100%;border-collapse:collapse;min-width:1080px}}
th,td{{padding:12px 14px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} th{{background:#eef1f5;font-size:13px;position:sticky;top:0}} th:nth-child(2),td:nth-child(2){{text-align:left}}
small{{display:block;color:var(--muted);font-weight:normal;margin-top:4px}} tr.control{{background:#fff7ed}} .note{{background:#fff;border-left:4px solid var(--warn);padding:16px;margin-top:20px}}
@media(max-width:760px){{.summary{{grid-template-columns:1fr 1fr}} main{{padding:12px}} h1{{font-size:22px}}}}
</style></head><body>
<header><h1>不同深度处理方法的 ACT 目标注意力指标</h1><p>无 Prompt、Depth-only ACT checkpoint_009000；ROI 来自最终批准的头部与执行腕部 SAM2 目标轨迹。</p></header>
<main><section class="summary"><div><b>{summary['diagnostics']['events']}</b>次批准抓取</div><div><b>{summary['diagnostics']['samples']}</b>个对齐时刻</div><div><b>{summary['frame_records_per_method']}</b>个有效双视角 ROI</div><div><b>{len(NORMAL_METHODS)}</b>种处理方法 + 2 个负对照</div></section>
<div class="table-wrap"><table><thead><tr><th>排序</th><th>深度方法</th><th>ROI 提升倍数 ↑</th><th>目标/背景密度比 ↑</th><th>峰值命中率 ↑</th><th>视角内 ROI 占比 ↑</th><th>质心距离 ↓</th><th>动作 MAE ↓</th></tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>
<p class="note"><b>解释边界：</b>排名主指标按 19 次抓取等权，避免长抓取主导结果。注意力更集中只能说明同一个 ACT 对该深度输入分配了更多目标区域注意力，不能单独证明任务成功；动作 MAE 是独立的留出集离线指标。零深度和空间打乱用于检查指标是否具备基本辨别力。</p>
</main></body></html>"""
    output_path.write_text(document, encoding="utf-8")


def summarize(args: argparse.Namespace, diagnostics: dict[str, int]) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    for method in NORMAL_METHODS + CONTROLS:
        path = args.output_dir / "frames" / f"{method}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing method output: {path}")
        all_rows.extend(read_jsonl(path))

    by_event = group_means(all_rows, ("method", "event_id", "sequence"))
    by_sequence = group_means(all_rows, ("method", "sequence"))
    by_view = group_means(all_rows, ("method", "role", "camera"))
    frame_weighted = {row["method"]: row for row in group_means(all_rows, ("method",))}
    events_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in by_event:
        events_by_method[row["method"]].append(row)

    benchmark = json.loads(args.benchmark_report.read_text(encoding="utf-8"))
    benchmark_by_method = {row["method"]: row["metrics"] for row in benchmark["results"]}
    raw_event = group_means(events_by_method["raw_aligned"], tuple())[0]
    raw_by_event = {row["event_id"]: row for row in events_by_method["raw_aligned"]}
    by_method = []
    for method in NORMAL_METHODS + CONTROLS:
        event_rows = events_by_method[method]
        event_balanced = {metric: float(np.mean([row[metric] for row in event_rows])) for metric in METRICS}
        ci = {metric: list(bootstrap_event_ci([row[metric] for row in event_rows])) for metric in METRICS}
        paired_delta_ci = {
            metric: list(
                bootstrap_event_ci(
                    [row[metric] - raw_by_event[row["event_id"]][metric] for row in event_rows],
                    seed=20260815,
                )
            )
            for metric in METRICS
        }
        record: dict[str, Any] = {
            "method": method,
            "method_zh": METHOD_ZH[method],
            "events": len(event_rows),
            "views": frame_weighted[method]["views"],
            "event_balanced": event_balanced,
            "frame_weighted": {metric: frame_weighted[method][metric] for metric in METRICS},
            "event_ci95": ci,
            "paired_delta_vs_raw_event_ci95": paired_delta_ci,
            "delta_vs_raw_event_balanced": {
                metric: event_balanced[metric] - raw_event[metric] for metric in METRICS
            },
        }
        record.update(benchmark_by_method.get(method, {}))
        by_method.append(record)

    primary_ranking = sorted(
        NORMAL_METHODS,
        key=lambda method: next(
            row["event_balanced"]["roi_attention_lift"] for row in by_method if row["method"] == method
        ),
        reverse=True,
    )
    summary = {
        "schema": "act_approved_dual_view_roi_attention_v1",
        "model": "prompt-free depth-only ACT",
        "checkpoint": str(args.checkpoint),
        "attention_source": "last decoder action-query cross-attention, averaged over heads and action queries",
        "roi_source": "human-approved dual-view SAM2 target tracks",
        "aggregation": "primary ranking uses equal weight for each of 19 grasp events",
        "diagnostics": diagnostics,
        "frame_records_per_method": int(min(row["views"] for row in by_method)),
        "primary_metric": "event-balanced roi_attention_lift",
        "ranking_by_roi_attention_lift": primary_ranking,
        "by_method": by_method,
        "by_event": by_event,
        "by_sequence": by_sequence,
        "by_view": by_view,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "attention_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    flat_methods = []
    for row in by_method:
        flat = {
            "method": row["method"],
            "method_zh": row["method_zh"],
            "events": row["events"],
            "views": row["views"],
            "chunk_mae_rad": row.get("chunk_mae_rad"),
        }
        for metric in METRICS:
            flat[f"event_balanced_{metric}"] = row["event_balanced"][metric]
            flat[f"frame_weighted_{metric}"] = row["frame_weighted"][metric]
            flat[f"delta_vs_raw_{metric}"] = row["delta_vs_raw_event_balanced"][metric]
            flat[f"paired_delta_ci95_low_{metric}"] = row["paired_delta_vs_raw_event_ci95"][metric][0]
            flat[f"paired_delta_ci95_high_{metric}"] = row["paired_delta_vs_raw_event_ci95"][metric][1]
        flat_methods.append(flat)
    write_csv(args.output_dir / "attention_metrics_by_method.csv", flat_methods)
    write_csv(args.output_dir / "attention_metrics_by_event.csv", by_event)
    write_csv(args.output_dir / "attention_metrics_by_sequence.csv", by_sequence)
    write_csv(args.output_dir / "attention_metrics_by_view.csv", by_view)
    render_html(summary, args.output_dir / "index.html")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--min-depth-m", type=float, default=0.2)
    parser.add_argument("--max-depth-m", type=float, default=4.0)
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    args.root = args.root.resolve()
    os.chdir(args.root)
    act_rows = read_jsonl(args.index)
    tracks = read_jsonl(args.tracks)
    samples, diagnostics = build_samples(act_rows, tracks)
    print(json.dumps({"alignment": diagnostics}, ensure_ascii=False), flush=True)
    if diagnostics["events"] != 19 or diagnostics["missing_act_rows"]:
        raise RuntimeError(f"Unexpected alignment coverage: {diagnostics}")

    if not args.summarize_only:
        methods = args.methods or list(NORMAL_METHODS + CONTROLS)
        invalid = [method for method in methods if method not in NORMAL_METHODS + CONTROLS]
        if invalid:
            raise ValueError(f"Unknown methods: {invalid}")
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        stats = payload.get("stats")
        if stats is None:
            stats = json.loads((args.checkpoint.parent / "stats.json").read_text())
        device = torch.device(args.device)
        policy = ACTPolicy(make_config(args.chunk_size, args.width, args.height, "depth_only"))
        policy.load_state_dict(payload["model"])
        policy.to(device)
        for method in methods:
            evaluate_method(policy, act_rows, samples, stats, method, args, device)

    if args.summarize_only or set(args.methods or ()) == set(NORMAL_METHODS + CONTROLS):
        summary = summarize(args, diagnostics)
        print(json.dumps({"ranking": summary["ranking_by_roi_attention_lift"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
