#!/usr/bin/env python3
"""Select the same grasp target in the head and active-wrist camera views."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from resolve_ambiguity_with_vlm import event_hint, task_instruction


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def available(row: dict[str, Any], camera: str) -> str:
    view = next(item for item in row["views"] if item["camera"] == camera)
    return ", ".join(
        f"#{item['candidate_id']} {item['label']}（交互分 {item['interaction_score']:.3f}）"
        for item in view["ranked_candidates"]
    )


def valid_id(row: dict[str, Any], camera: str, candidate_id: Any) -> bool:
    try:
        candidate_id = int(candidate_id)
    except (TypeError, ValueError):
        return False
    view = next((item for item in row["views"] if item["camera"] == camera), None)
    return bool(view and any(int(item["candidate_id"]) == candidate_id for item in view["ranked_candidates"]))


def prompt(row: dict[str, Any]) -> str:
    wrist = "cam_l" if row["side"] == "left" else "cam_r"
    hand = "左手" if row["side"] == "left" else "右手"
    return f"""你正在为机器人抓取视频建立双视角目标跟踪标注。
任务：{task_instruction(row['sequence'])}
事件：{event_hint(row)}
执行手：{hand}；执行腕相机：{wrist}。

第一张图是三相机六阶段时序图；第二张图是初始化帧候选框。
请在头部相机 cam_h 和执行腕相机 {wrist} 中，分别选择表示同一个实际抓取目标的候选框。
必须利用目标随执行手移动并在释放位置出现的时序证据。不要选择夹爪、机器人手、桌面、箱体、平台或同类静止干扰物。

cam_h 有效候选：{available(row, 'cam_h')}
{wrist} 有效候选：{available(row, wrist)}

camera 字段已经固定，不得修改。只输出中文 JSON，不要 Markdown：
{{"head":{{"camera":"cam_h","candidate_id":0,"confidence":0.0,"evidence":"头部视角证据"}},"wrist":{{"camera":"{wrist}","candidate_id":0,"confidence":0.0,"evidence":"执行腕视角证据"}},"same_instance_evidence":"两视角为何是同一实例","ambiguity":"遮挡、同类目标或大框风险；没有则写无","recommendation":"建议批准或建议人工复核或建议修正"}}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal", type=Path, default=Path("outputs/interaction_review/temporal_evidence.jsonl"))
    parser.add_argument("--queue", type=Path, default=Path("outputs/interaction_candidates/ambiguity_queue.jsonl"))
    parser.add_argument("--model", type=Path, default=Path("/ssd/hhw/models/internvla_a1_5/Qwen3.5-2B"))
    parser.add_argument("--output", type=Path, default=Path("outputs/interaction_review/qwen_required_views.jsonl"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    temporal = {row["event_id"]: row for row in read_jsonl(args.temporal)}
    rows = read_jsonl(args.queue)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(args.device)
    model.eval()

    results = []
    for index, row in enumerate(rows):
        evidence = temporal[row["event_id"]]
        messages = [{"role": "user", "content": [
            {"type": "image", "image": str(Path(evidence["temporal_collage_path"]).resolve())},
            {"type": "image", "image": str(Path(evidence["candidate_collage_path"]).resolve())},
            {"type": "text", "text": prompt(row)},
        ]}]
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[chat], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=420, do_sample=False)
        trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
        response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        annotation = parse_json(response)
        wrist = "cam_l" if row["side"] == "left" else "cam_r"
        head_valid = bool(annotation and annotation.get("head", {}).get("camera") == "cam_h" and valid_id(row, "cam_h", annotation.get("head", {}).get("candidate_id")))
        wrist_valid = bool(annotation and annotation.get("wrist", {}).get("camera") == wrist and valid_id(row, wrist, annotation.get("wrist", {}).get("candidate_id")))
        results.append({
            "event_id": row["event_id"],
            "sequence": row["sequence"],
            "event_index": row["event_index"],
            "side": row["side"],
            "model": "Qwen3.5-2B",
            "annotation": annotation,
            "head_valid": head_valid,
            "wrist_valid": wrist_valid,
            "valid_both": head_valid and wrist_valid,
            "raw_response": response,
            "status": "pending_human_review",
        })
        print(f"[qwen-required] {index + 1}/{len(rows)} {row['event_id']} head={head_valid} wrist={wrist_valid}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    print(json.dumps({
        "events": len(results),
        "head_valid": sum(row["head_valid"] for row in results),
        "wrist_valid": sum(row["wrist_valid"] for row in results),
        "valid_both": sum(row["valid_both"] for row in results),
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
