#!/usr/bin/env python3
"""Create detailed Chinese annotations for all grasps with local Qwen3.5-2B."""

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


def candidate_table(row: dict[str, Any]) -> str:
    lines = []
    for view in row["views"]:
        items = ", ".join(
            f"#{item['candidate_id']} {item['label']}（交互分 {item['interaction_score']:.3f}）"
            for item in view["ranked_candidates"]
        )
        lines.append(f"{view['camera']}: {items or '无候选'}")
    return "\n".join(lines)


def valid_choice(annotation: dict[str, Any] | None, row: dict[str, Any]) -> bool:
    if not annotation:
        return False
    camera = str(annotation.get("camera", ""))
    try:
        candidate_id = int(annotation.get("candidate_id"))
    except (TypeError, ValueError):
        return False
    return any(
        view["camera"] == camera
        and any(int(item["candidate_id"]) == candidate_id for item in view["ranked_candidates"])
        for view in row["views"]
    )


def prompt(row: dict[str, Any]) -> str:
    hand = "左手" if row["side"] == "left" else "右手"
    return f"""你正在复核离线机器人抓取演示。
任务：{task_instruction(row['sequence'])}
事件提示：{event_hint(row)}
当前由{hand}执行抓取。

第一张图是三视角六阶段时序图，每行依次为头部、左腕、右腕相机，每列依次为接近、接触前、初始化/接触、搬运中段、释放、释放后。
第二张图是初始化帧的三视角候选框，框内标有 RANK/ID；不同相机的候选 ID 独立。
请根据夹爪接触、目标随手运动、释放位置及跨视角一致性，识别实际交互实例。不要选择机器人手、夹爪、桌面、箱体、平台或同类干扰物。

有效候选：
{candidate_table(row)}

camera 字段必须只填写一个实际选中的相机名称，禁止照抄选项列表。
只输出一个中文 JSON 对象，不要 Markdown。格式示例：
{{"camera":"cam_l","candidate_id":0,"confidence":0.0,"target_description":"目标外观、类别和初始位置","temporal_evidence":"从接近到释放的时序证据","cross_view_evidence":"三视角支持或冲突","ambiguity":"遮挡、同类物体或候选框风险；没有则写无","recommendation":"建议批准或建议人工复核或建议修正"}}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal", type=Path, default=Path("outputs/interaction_review/temporal_evidence.jsonl"))
    parser.add_argument("--queue", type=Path, default=Path("outputs/interaction_candidates/ambiguity_queue.jsonl"))
    parser.add_argument("--model", type=Path, default=Path("/ssd/hhw/models/internvla_a1_5/Qwen3.5-2B"))
    parser.add_argument("--output", type=Path, default=Path("outputs/interaction_review/qwen_temporal_annotations.jsonl"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--event-id")
    parser.add_argument("--merge-existing", action="store_true")
    args = parser.parse_args()

    temporal = {row["event_id"]: row for row in read_jsonl(args.temporal)}
    rows = read_jsonl(args.queue)
    if args.event_id:
        rows = [row for row in rows if row["event_id"] == args.event_id]

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
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
            generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
        response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        annotation = parse_json(response)
        results.append({
            "event_id": row["event_id"],
            "sequence": row["sequence"],
            "event_index": row["event_index"],
            "side": row["side"],
            "model": "Qwen3.5-2B",
            "annotation": annotation,
            "valid_choice": valid_choice(annotation, row),
            "raw_response": response,
            "status": "pending_human_review",
        })
        print(f"[qwen-temporal] {index + 1}/{len(rows)} {row['event_id']} valid={results[-1]['valid_choice']}", flush=True)

    if args.merge_existing and args.output.exists():
        merged = {row["event_id"]: row for row in read_jsonl(args.output)}
        merged.update({row["event_id"]: row for row in results})
        results = list(merged.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    print(json.dumps({"events": len(results), "valid": sum(bool(row["valid_choice"]) for row in results), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
