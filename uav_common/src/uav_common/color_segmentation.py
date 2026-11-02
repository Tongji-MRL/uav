"""Small, ROS-independent helpers for detection colour segmentation."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Optional, Sequence

import cv2
import numpy as np


COLOR_HSV_RANGES = {
    "red": (((0, 80, 50), (10, 255, 255)), ((170, 80, 50), (179, 255, 255))),
    "green": (((35, 55, 40), (90, 255, 255)),),
    "blue": (((95, 70, 35), (135, 255, 255)),),
}


def stamp_to_nanoseconds(stamp: Any) -> Optional[int]:
    """Return a JSON/ROS-like stamp as integer nanoseconds."""
    if not isinstance(stamp, dict):
        return None
    try:
        sec = int(stamp["sec"])
        nanosec = int(stamp["nanosec"])
    except (KeyError, TypeError, ValueError):
        return None
    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        return None
    return sec * 1_000_000_000 + nanosec


def nearest_image(
    images: Sequence[tuple[int, np.ndarray]],
    target_stamp_ns: Optional[int],
    tolerance_ns: int,
) -> Optional[np.ndarray]:
    """Find the exact or nearest buffered image within the allowed time gap."""
    if target_stamp_ns is None or not images:
        return None
    stamp_ns, image = min(images, key=lambda item: abs(item[0] - target_stamp_ns))
    if abs(stamp_ns - target_stamp_ns) > tolerance_ns:
        return None
    return image


def classify_color(
    rgb: np.ndarray,
    bbox_xyxy: Any,
    roi_fraction: float,
    min_color_ratio: float,
) -> str:
    """Classify the dominant red, green, or blue colour in a box centre ROI."""
    bbox = _valid_bbox(bbox_xyxy)
    if bbox is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        return "None"

    image_height, image_width = rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    half_width = (x2 - x1) * roi_fraction * 0.5
    half_height = (y2 - y1) * roi_fraction * 0.5

    left = max(0, int(math.floor(center_x - half_width)))
    top = max(0, int(math.floor(center_y - half_height)))
    right = min(image_width, int(math.ceil(center_x + half_width)))
    bottom = min(image_height, int(math.ceil(center_y + half_height)))
    if left >= right or top >= bottom:
        return "None"

    hsv = cv2.cvtColor(rgb[top:bottom, left:right], cv2.COLOR_RGB2HSV)
    ratios: dict[str, float] = {}
    for color, ranges in COLOR_HSV_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(
                hsv,
                np.asarray(lower, dtype=np.uint8),
                np.asarray(upper, dtype=np.uint8),
            )
        ratios[color] = float(np.count_nonzero(mask)) / float(mask.size)

    color = max(ratios, key=ratios.get)
    return color if ratios[color] >= min_color_ratio else "None"


def enrich_detections(
    payload: dict[str, Any],
    rgb: Optional[np.ndarray],
    target_types: set[str],
    roi_fraction: float,
    min_color_ratio: float,
) -> dict[str, Any]:
    """Add per-frame IDs and colours while preserving all existing fields."""
    detections = payload.get("detections")
    if not isinstance(detections, list):
        raise ValueError("detections must be a list")
    if any(not isinstance(item, dict) for item in detections):
        raise ValueError("every detection must be a mapping")

    groups: dict[tuple[str, Any], list[int]] = defaultdict(list)
    for index, detection in enumerate(detections):
        object_type = str(detection.get("object_type", "")).strip().lower()
        class_id = detection.get("class_id")
        try:
            hash(class_id)
        except TypeError:
            class_id = repr(class_id)
        groups[(object_type, class_id)].append(index)

        detection["color"] = "None"
        if rgb is not None and object_type in target_types:
            detection["color"] = classify_color(
                rgb,
                detection.get("bbox_xyxy"),
                roi_fraction,
                min_color_ratio,
            )

    for indices in groups.values():
        ordered = sorted(
            indices,
            key=lambda index: (*_detection_center(detections[index]), index),
        )
        for count_id, index in enumerate(ordered):
            detections[index]["id"] = count_id

    return payload


def _valid_bbox(value: Any) -> Optional[tuple[float, float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in bbox):
        return None
    x1, y1, x2, y2 = bbox
    return bbox if x2 > x1 and y2 > y1 else None


def _detection_center(detection: dict[str, Any]) -> tuple[float, float]:
    center = detection.get("center_xy")
    if isinstance(center, (list, tuple)) and len(center) == 2:
        try:
            x, y = float(center[0]), float(center[1])
            if math.isfinite(x) and math.isfinite(y):
                return x, y
        except (TypeError, ValueError):
            pass
    bbox = _valid_bbox(detection.get("bbox_xyxy"))
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5
    return math.inf, math.inf
