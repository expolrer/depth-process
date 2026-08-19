#!/usr/bin/env python3
"""Use a local Qwen3.5 VLM to resolve ambiguous grasp-target candidates."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from qwen_vl_utils import process_vision_info


TASK_INSTRUCTIONS = {
    "chengzhong": (
        "Use the left hand to pick up one metal sleeve from the box on the left and place it steadily "
        "on the weighing platform in the center. Then use the right hand to pick up the sleeve from the "
        "center platform and place it securely into the box on the right."
    ),
    "dajian": (
        "Use the right hand to pick up the automotive sheet-metal part from the center support stand "
        "and move it to the target support stand on the right."
    ),
    "zhoumian": (
        "Use the right hand to pick up the pin connector, seat belt, and cable from the tabletop one at "
        "a time, and place them in their designated positions in the empty box: from left to right, pin "
        "connector, seat belt, and cable."
    ),
    "leju_claw": "Pick up the intended toy from the toy pile and move it to the designated side of the tabletop.",
    "dex_hand": "Pick up the intended toy from the toy pile and move it to the designated side of the tabletop.",
}


def task_instruction(sequence: str) -> str:
    lowered = sequence.lower()
    for key, instruction in TASK_INSTRUCTIONS.items():
        if key in lowered:
            return instruction
    return "Identify the physical object grasped by the robot in this event."


def event_hint(row: dict[str, Any]) -> str:
    sequence = row["sequence"].lower()
    event_index = int(row["event_index"])
    if "zhoumian" in sequence:
        ordered = ("pin connector", "seat belt", "cable")
        if event_index < len(ordered):
            return f"This is grasp {event_index + 1} of 3. The expected target category is: {ordered[event_index]}."
    if "chengzhong" in sequence:
        if event_index == 0:
            return "This is the left-hand grasp of one metal sleeve from the left box."
        if event_index == 1:
            return "This is the right-hand grasp of the same metal sleeve from the center weighing platform."
    if "dajian" in sequence:
        return "The expected target category is the automotive sheet-metal part, not a support stand."
    return f"This is grasp event {event_index + 1}; identify the object physically enclosed by the active gripper."


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_json(text: str) -> dict[str, Any] | None:
    matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for match in reversed(matches):
        try:
            value = json.loads(match)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def valid_choice(choice: dict[str, Any] | None, row: dict[str, Any]) -> bool:
    if not choice:
        return False
    camera = str(choice.get("camera", ""))
    try:
        candidate_id = int(choice.get("candidate_id"))
    except (TypeError, ValueError):
        return False
    for view in row["views"]:
        if view["camera"] != camera:
            continue
        return any(int(candidate["candidate_id"]) == candidate_id for candidate in view["ranked_candidates"])
    return False


def prompt_for(row: dict[str, Any]) -> str:
    candidate_table = []
    for view in row["views"]:
        choices = ", ".join(
            f"ID {candidate['candidate_id']} ({candidate['label']}, score {candidate['interaction_score']:.3f})"
            for candidate in view["ranked_candidates"]
        )
        candidate_table.append(f"{view['camera']}: {choices or 'no object candidates'}")
    return f"""You are reviewing an offline robot manipulation demonstration.
Task instruction: {task_instruction(row['sequence'])}
Current event constraint: {event_hint(row)}
Current grasp uses the {row['side']} hand.

The image is a three-view collage. Candidate boxes are labeled RANK/ID and each camera has its own IDs.
Choose the single box that contains the physical object currently enclosed or contacted by the closing {row['side']} gripper.
Do not choose the gripper, robot hand, table, box, platform, or another same-class distractor.
Use visible gripper contact and enclosure as primary evidence. Use the task instruction only as supporting evidence.

Available candidates:
{chr(10).join(candidate_table)}

Return exactly one JSON object with no markdown:
{{"camera":"cam_h|cam_l|cam_r","candidate_id":0,"confidence":0.0,"reason":"short visual reason"}}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=Path("outputs/interaction_candidates/ambiguity_queue.jsonl"))
    parser.add_argument("--model", type=Path, default=Path("/ssd/hhw/models/internvla_a1_5/Qwen3.5-2B"))
    parser.add_argument("--output", type=Path, default=Path("outputs/interaction_candidates/vlm_resolutions.jsonl"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--all-events", action="store_true")
    parser.add_argument("--event-id")
    parser.add_argument("--merge-existing", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.queue)
    rows = [row for row in rows if args.all_events or row["needs_vlm"]]
    if args.event_id:
        rows = [row for row in rows if row["event_id"] == args.event_id]
    if args.max_events is not None:
        rows = rows[: args.max_events]

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    model.eval()

    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        image_path = str(Path(row["collage_path"]).resolve())
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt_for(row)},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(args.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=160, do_sample=False)
        trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
        response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        choice = parse_json(response)
        valid = valid_choice(choice, row)
        result = {
            "event_id": row["event_id"],
            "sequence": row["sequence"],
            "event_index": row["event_index"],
            "side": row["side"],
            "model": str(args.model),
            "choice": choice,
            "valid_choice": valid,
            "raw_response": response,
            "status": "pending_human_review",
        }
        results.append(result)
        print(f"[vlm] {index + 1}/{len(rows)} {row['event_id']} valid={valid} choice={choice}", flush=True)

    if args.merge_existing and args.output.exists():
        merged = {row["event_id"]: row for row in read_jsonl(args.output)}
        merged.update({row["event_id"]: row for row in results})
        results = list(merged.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    summary = {
        "events": len(results),
        "valid_choices": sum(bool(row["valid_choice"]) for row in results),
        "pending_human_review": len(results),
        "output": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
