"""Plain-Python multi-model detector for YOLO .pt files."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


class PtDetector:
    """Run configured YOLO models without creating any ROS entities."""

    def __init__(
        self,
        perception: dict[str, Any],
        models: Sequence[dict[str, Any]],
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.image_size = int(perception.get("image_size", 640))
        self.device = str(perception.get("device", "cuda:0")).strip()
        if self.image_size <= 0:
            raise ValueError("perception.image_size must be positive")

        if model_factory is None:
            from ultralytics import YOLO

            model_factory = YOLO

        self.models: list[tuple[str, float, Any]] = []
        for index, item in enumerate(models):
            object_type = str(item.get("object_type", "")).strip().lower()
            model_path = Path(str(item.get("path", ""))).expanduser()
            confidence = float(item.get("confidence", 0.25))
            if not object_type:
                raise ValueError(f"models[{index}].object_type is required")
            if not model_path.is_file():
                raise RuntimeError(f"model does not exist: {model_path}")
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError(f"models[{index}].confidence must be in [0, 1]")
            self.models.append(
                (object_type, confidence, model_factory(str(model_path)))
            )
        if not self.models:
            raise ValueError("models must contain at least one enabled model")

        self.sequence = 0

    def process(
        self,
        rgb: np.ndarray,
        stamp_sec: int,
        stamp_nanosec: int,
    ) -> dict[str, Any]:
        """Run all models on one RGB image and return the original payload format."""
        if not isinstance(rgb, np.ndarray):
            raise TypeError("rgb must be a numpy array")
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must have shape [height, width, 3]")
        if rgb.dtype != np.uint8:
            raise ValueError("rgb must use uint8 pixels")

        # Ultralytics expects BGR when a numpy image is passed directly.
        bgr = rgb[:, :, ::-1].copy()
        detections: list[dict[str, Any]] = []

        for object_type, confidence, model in self.models:
            predict_args: dict[str, Any] = {
                "conf": confidence,
                "imgsz": self.image_size,
                "verbose": False,
            }
            if self.device:
                predict_args["device"] = self.device

            for result in model.predict(bgr, **predict_args):
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                coordinates = boxes.xyxy.detach().cpu().tolist()
                confidences = boxes.conf.detach().cpu().tolist()
                class_ids = boxes.cls.detach().cpu().tolist()
                for bbox, score, class_value in zip(
                    coordinates,
                    confidences,
                    class_ids,
                ):
                    class_id = int(class_value)
                    class_name = self._class_name(result.names, class_id)
                    x1, y1, x2, y2 = [float(value) for value in bbox]
                    detections.append(
                        {
                            "object_type": object_type,
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": float(score),
                            "bbox_xyxy": [x1, y1, x2, y2],
                            "center_xy": [
                                (x1 + x2) * 0.5,
                                (y1 + y2) * 0.5,
                            ],
                        }
                    )

        self.sequence += 1
        return {
            "sequence": self.sequence,
            "rgb_stamp": {
                "sec": int(stamp_sec),
                "nanosec": int(stamp_nanosec),
            },
            "rgb_image_size": {
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
            },
            "detection_count": len(detections),
            "detections": detections,
        }

    @staticmethod
    def _class_name(names: Any, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        return str(names[class_id])
