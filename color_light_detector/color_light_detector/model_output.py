"""One model-task contract and native-image axis-aligned boxes for all runners."""
import math

import numpy as np

from .task1_beacon_core import Detection


def load_model(path, task="auto"):
    from ultralytics import YOLO
    if task not in ("auto", "detect", "segment"):
        raise ValueError("model_task must be auto, detect or segment")
    if str(path).lower().endswith(".engine") and task == "auto":
        raise ValueError("Set model_task=detect or segment for an engine; its filename is not a task contract")
    model = YOLO(str(path), task=None if task == "auto" else task)
    if model.task not in ("detect", "segment") or (task != "auto" and model.task != task):
        raise ValueError(f"Model task {model.task!r} disagrees with configured {task!r}")
    if not str(path).lower().endswith(".engine") and dict(model.names) != {0: "beacon"}:
        # Old one-class checkpoints have other class names. Only the ID contract is required.
        if set(model.names) != {0}:
            raise ValueError("Task1 requires one beacon class at class ID 0")
    return model


def detections_from_prediction(prediction, shape, require_masks=False):
    """Enclose mask polygons tightly; x2/y2 are exclusive pixel bounds, never rotated boxes."""
    height, width = shape[:2]
    if prediction.boxes is None or len(prediction.boxes) == 0:
        return []
    boxes = prediction.boxes.cpu()
    polygons = prediction.masks.xy if prediction.masks is not None else None
    if require_masks and polygons is None:
        raise ValueError("Segmentation output has boxes but no masks; check model_task/runtime/export")
    if polygons is not None and len(polygons) != len(boxes):
        raise ValueError("Mask/box count mismatch")
    result = []
    for i, (xyxy, confidence, class_id) in enumerate(zip(
            boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())):
        if int(class_id) != 0:
            continue
        if polygons is not None:
            points = np.asarray(polygons[i], dtype=float).reshape(-1, 2)
            if len(points) < 3 or not np.isfinite(points).all():
                continue
            lo, hi = points.min(axis=0), points.max(axis=0)
            # Contours refer to pixel centers. Include their last occupied pixel.
            xyxy = [lo[0], lo[1], hi[0] + 1, hi[1] + 1]
        if not all(math.isfinite(v) for v in (*xyxy, confidence)):
            continue
        x1, y1 = max(0, math.floor(xyxy[0])), max(0, math.floor(xyxy[1]))
        x2, y2 = min(width, math.ceil(xyxy[2])), min(height, math.ceil(xyxy[3]))
        if x2 > x1 and y2 > y1:
            result.append(Detection((x1, y1, x2, y2), float(confidence), int(class_id)))
    return result
