from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from PIL import Image


class ConceptDetector(Protocol):
    def detect(self, image: Image.Image, phrases: list[str]) -> list[dict[str, Any]]: ...


class GroundingDinoDetector:
    def __init__(
        self,
        model_path: Path,
        device: str = "cuda:0",
        box_threshold: float = 0.20,
        text_threshold: float = 0.18,
    ) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32,
        ).to(device)
        self.model.eval()

    def detect(self, image: Image.Image, phrases: list[str]) -> list[dict[str, Any]]:
        prompt = " . ".join(phrase.strip(" .") for phrase in phrases) + " ."
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        model_inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self.torch.inference_mode():
            outputs = self.model(**model_inputs)
        kwargs = {
            "target_sizes": [(image.height, image.width)],
            "text_threshold": self.text_threshold,
        }
        try:
            result = self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids,
                box_threshold=self.box_threshold,
                **kwargs,
            )[0]
        except TypeError:
            result = self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids,
                threshold=self.box_threshold,
                **kwargs,
            )[0]
        labels = result.get("text_labels", result.get("labels", []))
        return [
            {
                "instance_id": index,
                "label": label if isinstance(label, str) else str(label),
                "box_xyxy": [float(value) for value in box],
                "detector_score": float(score),
            }
            for index, (box, score, label) in enumerate(
                zip(
                    result["boxes"].detach().float().cpu().tolist(),
                    result["scores"].detach().float().cpu().tolist(),
                    labels,
                )
            )
        ]


class Sam3Detector:
    def __init__(self, checkpoint: Path | None = None, device: str = "cuda:0") -> None:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        kwargs: dict[str, Any] = {"device": device}
        if checkpoint:
            kwargs.update({"checkpoint_path": str(checkpoint), "load_from_HF": False})
        self.processor = Sam3Processor(build_sam3_image_model(**kwargs))

    def detect(self, image: Image.Image, phrases: list[str]) -> list[dict[str, Any]]:
        state = self.processor.set_image(image)
        detections = []
        for phrase in phrases:
            output = self.processor.set_text_prompt(state=state, prompt=phrase)
            boxes = output["boxes"].detach().float().cpu().tolist()
            scores = output["scores"].detach().float().cpu().tolist()
            for box, score in zip(boxes, scores):
                detections.append(
                    {
                        "instance_id": len(detections),
                        "label": phrase,
                        "box_xyxy": [float(value) for value in box],
                        "detector_score": float(score),
                    }
                )
        return detections


def load_detector(
    backend: str,
    model_path: Path | None,
    device: str,
) -> ConceptDetector:
    if backend == "sam3":
        return Sam3Detector(model_path, device)
    if model_path is None:
        raise ValueError("GroundingDINO inventory requires --grounding-model")
    return GroundingDinoDetector(model_path, device)
