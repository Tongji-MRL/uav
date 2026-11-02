"""Plain-Python multi-model RGB-D detector used by MapManagerNode."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - deployment dependency check
    cv2 = None


BUOY_HSV_RANGES = {
    "red": (((0, 80, 50), (10, 255, 255)), ((170, 80, 50), (179, 255, 255))),
    "green": (((35, 55, 40), (90, 255, 255)),),
    "blue": (((95, 70, 35), (135, 255, 255)),),
}


@dataclass(frozen=True)
class DepthFrame:
    values_m: np.ndarray
    raw_invalid_mask: np.ndarray
    stamp_sec: float
    encoding: str


@dataclass
class ModelSpec:
    object_type: str
    path: Path
    confidence: float
    model: Any


def image_stamp_sec(message: Any) -> float:
    return float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9


def decode_rgb_image(message: Any) -> np.ndarray:
    """Decode a sensor_msgs/Image-like object into an RGB uint8 array."""
    encoding = str(message.encoding).lower()
    channels = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
        "8uc3": 3,
        "8uc4": 4,
    }.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported RGB image encoding: {message.encoding}")

    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(
        int(message.height), int(message.step)
    )
    image = rows[:, : int(message.width) * channels].reshape(
        int(message.height), int(message.width), channels
    )
    if encoding in {"rgb8", "8uc3"}:
        return image.copy()
    if encoding == "bgr8":
        return image[:, :, ::-1].copy()
    if encoding in {"rgba8", "8uc4"}:
        return image[:, :, :3].copy()
    if encoding == "bgra8":
        return image[:, :, [2, 1, 0]].copy()
    return np.repeat(image, 3, axis=2)


def decode_depth_image(
    message: Any, unit: str, invalid_raw_values: Sequence[float]
) -> DepthFrame:
    """Decode a depth Image-like object and convert it to metres."""
    encoding = str(message.encoding).lower()
    if encoding in {"16uc1", "mono16"}:
        dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
        scale = {
            "m": 1.0,
            "meter": 1.0,
            "meters": 1.0,
            "cm": 0.01,
            "centimeter": 0.01,
            "centimeters": 0.01,
            "mm": 0.001,
            "millimeter": 0.001,
            "millimeters": 0.001,
        }.get(str(unit).strip().lower(), 0.001)
    elif encoding == "32fc1":
        dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
        scale = 1.0
    else:
        raise ValueError(f"unsupported depth image encoding: {message.encoding}")

    row_elements = int(message.step) // dtype.itemsize
    raw = np.frombuffer(message.data, dtype=dtype).reshape(
        int(message.height), row_elements
    )[:, : int(message.width)]
    raw_invalid = np.zeros(raw.shape, dtype=bool)
    for invalid_value in invalid_raw_values:
        raw_invalid |= raw == invalid_value
    return DepthFrame(
        values_m=raw.astype(np.float32) * scale,
        raw_invalid_mask=raw_invalid,
        stamp_sec=image_stamp_sec(message),
        encoding=str(message.encoding),
    )


class DepthProcessor:
    """Attach synchronized median depth values without loading YOLO models."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.depth_topic = str(config.get("depth_topic", "/oak/stereo/image_raw"))
        self.depth_aligned_to_rgb = bool(config.get("depth_aligned_to_rgb", True))
        self.depth_sync_tolerance_sec = float(
            config.get("depth_sync_tolerance_sec", 0.10)
        )
        self.depth_roi_fraction = float(config.get("depth_roi_fraction", 0.40))
        self.depth_min_m = float(config.get("depth_min_m", 0.05))
        self.depth_max_m = float(config.get("depth_max_m", 20.0))
        if not 0.0 < self.depth_roi_fraction <= 1.0:
            raise ValueError("perception.depth_roi_fraction must be in (0, 1]")
        if self.depth_min_m <= 0.0 or self.depth_max_m <= self.depth_min_m:
            raise ValueError("perception depth range is invalid")
        if (
            not math.isfinite(self.depth_sync_tolerance_sec)
            or self.depth_sync_tolerance_sec < 0.0
        ):
            raise ValueError(
                "perception.depth_sync_tolerance_sec must be non-negative"
            )

    def attach(
        self,
        detections: list[dict[str, Any]],
        rgb_shape: tuple[int, int],
        rgb_stamp: float,
        depth: Optional[DepthFrame],
    ) -> tuple[str, Optional[float]]:
        """Attach depth fields and return status plus RGB-depth time delta."""
        for detection in detections:
            detection["depth_median_m"] = None
            detection["depth_valid_pixel_count"] = 0

        if depth is None:
            return "depth_unavailable", None
        if not self.depth_aligned_to_rgb:
            return "depth_not_aligned_to_rgb", None
        sync_delta = abs(rgb_stamp - depth.stamp_sec)
        if sync_delta > self.depth_sync_tolerance_sec:
            return "depth_not_time_matched", sync_delta

        rgb_height, rgb_width = rgb_shape
        depth_height, depth_width = depth.values_m.shape[:2]
        scale_x = depth_width / float(rgb_width)
        scale_y = depth_height / float(rgb_height)
        for detection in detections:
            coordinates = detection.get("bbox_xyxy")
            if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 4:
                continue
            try:
                x1, y1, x2, y2 = [float(value) for value in coordinates]
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                continue
            if x2 <= x1 or y2 <= y1:
                continue

            center_x = (x1 + x2) * 0.5
            center_y = (y1 + y2) * 0.5
            half_width = max((x2 - x1) * self.depth_roi_fraction * 0.5, 1.0)
            half_height = max((y2 - y1) * self.depth_roi_fraction * 0.5, 1.0)
            dx1 = int(
                np.clip(
                    np.floor((center_x - half_width) * scale_x),
                    0,
                    depth_width - 1,
                )
            )
            dy1 = int(
                np.clip(
                    np.floor((center_y - half_height) * scale_y),
                    0,
                    depth_height - 1,
                )
            )
            dx2 = int(
                np.clip(
                    np.ceil((center_x + half_width) * scale_x),
                    dx1 + 1,
                    depth_width,
                )
            )
            dy2 = int(
                np.clip(
                    np.ceil((center_y + half_height) * scale_y),
                    dy1 + 1,
                    depth_height,
                )
            )
            roi = depth.values_m[dy1:dy2, dx1:dx2]
            invalid = depth.raw_invalid_mask[dy1:dy2, dx1:dx2]
            valid = (
                np.isfinite(roi)
                & ~invalid
                & (roi >= self.depth_min_m)
                & (roi <= self.depth_max_m)
            )
            values = roi[valid]
            detection["depth_valid_pixel_count"] = int(values.size)
            if values.size:
                detection["depth_median_m"] = float(np.median(values))
        return "depth_attached", sync_delta


class MultiRgbdTargetDetector:
    """Run configured YOLO models and return one JSON-compatible payload."""

    def __init__(
        self,
        config: dict[str, Any],
        models: Sequence[dict[str, Any]],
        *,
        model_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is required by the RGB-D detector")
        self.image_size = int(config.get("image_size", 640))
        self.device = str(config.get("device", "cuda:0")).strip()
        self.depth_processor = DepthProcessor(config)
        if self.image_size <= 0:
            raise ValueError("perception.image_size must be positive")

        if model_factory is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Ultralytics is required for configured models") from exc
            model_factory = YOLO
        self.models = self._load_models(models, model_factory)
        self.model_order = {
            spec.object_type: index for index, spec in enumerate(self.models)
        }
        self.sequence = 0
        self.failed_models: set[str] = set()

    @staticmethod
    def _load_models(
        model_configs: Sequence[dict[str, Any]], model_factory: Callable[[str], Any]
    ) -> list[ModelSpec]:
        if not isinstance(model_configs, (list, tuple)) or not model_configs:
            raise ValueError("models must contain at least one model definition")
        specs: list[ModelSpec] = []
        object_types: set[str] = set()
        for index, item in enumerate(model_configs):
            if not isinstance(item, dict):
                raise ValueError(f"models[{index}] must be a mapping")
            object_type = str(item.get("object_type", "")).strip().lower()
            path = Path(str(item.get("path", ""))).expanduser()
            confidence = float(item.get("confidence", 0.25))
            if not object_type or object_type in object_types:
                raise ValueError("model object_type values must be non-empty and unique")
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError(f"models[{index}].confidence must be in [0, 1]")
            if not path.is_file():
                raise RuntimeError(f"configured model path does not exist: {path}")
            specs.append(
                ModelSpec(
                    object_type=object_type,
                    path=path,
                    confidence=confidence,
                    model=model_factory(str(path)),
                )
            )
            object_types.add(object_type)
        return specs

    def process(
        self,
        rgb: np.ndarray,
        *,
        rgb_stamp_sec: float,
        stamp_sec: int,
        stamp_nanosec: int,
        depth: Optional[DepthFrame],
        enabled_object_types: Optional[set[str]] = None,
    ) -> tuple[dict[str, Any], list[str]]:
        """Run inference and return payload plus newly changed model errors."""
        if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must be an HxWx3 array")

        detections: list[dict[str, Any]] = []
        errors: list[str] = []
        for spec in self.models:
            # None表示运行全部模型；
            # 集合表示只运行集合中指定的模型。
            if (
                enabled_object_types is not None
                and spec.object_type not in enabled_object_types
            ):
                continue

            try:
                detections.extend(self._detect_with_model(spec, rgb))
                if spec.object_type in self.failed_models:
                    errors.append(f"{spec.object_type} model inference recovered")
                self.failed_models.discard(spec.object_type)
            except Exception as exc:  # noqa: BLE001
                if spec.object_type not in self.failed_models:
                    errors.append(f"{spec.object_type} model inference failed: {exc}")
                self.failed_models.add(spec.object_type)

        self._sort_and_assign_ids(detections)
        depth_status, sync_delta = self.depth_processor.attach(
            detections, rgb.shape[:2], rgb_stamp_sec, depth
        )
        self.sequence += 1
        return (
            {
                "sequence": self.sequence,
                "rgb_stamp": {"sec": int(stamp_sec), "nanosec": int(stamp_nanosec)},
                "rgb_image_size": {
                    "width": int(rgb.shape[1]),
                    "height": int(rgb.shape[0]),
                },
                "depth": {
                    "topic": self.depth_processor.depth_topic,
                    "aligned_to_rgb": self.depth_processor.depth_aligned_to_rgb,
                    "status": depth_status,
                    "sync_delta_sec": sync_delta,
                },
                "detection_count": len(detections),
                "detections": detections,
            },
            errors,
        )

    def _detect_with_model(
        self, spec: ModelSpec, rgb: np.ndarray
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "conf": spec.confidence,
            "imgsz": self.image_size,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device
        detections: list[dict[str, Any]] = []
        for result in spec.model.predict(rgb, **kwargs):
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            boxes_xyxy = boxes.xyxy.detach().cpu().tolist()
            confidences = boxes.conf.detach().cpu().tolist()
            for coordinates, confidence in zip(boxes_xyxy, confidences):
                bbox = self._clip_bbox(coordinates, rgb.shape[:2])
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                detection: dict[str, Any] = {
                    "object_type": spec.object_type,
                    "class_name": spec.object_type,
                    "id": None,
                    "confidence": float(confidence),
                    "color": "None",
                    "state": "static",
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "center_xy": [(x1 + x2) * 0.5, (y1 + y2) * 0.5],
                    "depth_median_m": None,
                    "depth_valid_pixel_count": 0,
                }
                if spec.object_type in {"tin", "circle"}:
                    color, ratio = self._classify_buoy_color(rgb, bbox)
                    detection["color"] = color
                    detection["color_match_ratio"] = ratio
                detections.append(detection)
        return detections

    def _sort_and_assign_ids(self, detections: list[dict[str, Any]]) -> None:
        detections.sort(
            key=lambda item: (
                self.model_order[str(item["object_type"])],
                float(item["center_xy"][0]),
                float(item["center_xy"][1]),
            )
        )
        next_id: dict[str, int] = defaultdict(int)
        for detection in detections:
            object_type = str(detection["object_type"])
            detection["id"] = next_id[object_type]
            next_id[object_type] += 1

    @staticmethod
    def _classify_buoy_color(
        rgb: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> tuple[str, float]:
        x1, y1, x2, y2 = bbox
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return "red", 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        ratios: dict[str, float] = {}
        for color, ranges in BUOY_HSV_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(
                    hsv,
                    np.asarray(lower, dtype=np.uint8),
                    np.asarray(upper, dtype=np.uint8),
                )
            ratios[color] = float(np.count_nonzero(mask)) / float(mask.size)
        color = max(BUOY_HSV_RANGES, key=lambda name: ratios[name])
        return color, ratios[color]

    def _attach_depth(
        self,
        detections: list[dict[str, Any]],
        rgb_shape: tuple[int, int],
        rgb_stamp: float,
        depth: Optional[DepthFrame],
    ) -> tuple[str, Optional[float]]:
        """Compatibility wrapper for the previous private method."""
        return self.depth_processor.attach(
            detections,
            rgb_shape,
            rgb_stamp,
            depth,
        )

    @staticmethod
    def _clip_bbox(
        coordinates: Sequence[float], image_shape: tuple[int, int]
    ) -> Optional[tuple[int, int, int, int]]:
        if len(coordinates) != 4:
            return None
        height, width = image_shape
        x1, y1, x2, y2 = (int(round(float(value))) for value in coordinates)
        x1 = int(np.clip(x1, 0, width - 1))
        y1 = int(np.clip(y1, 0, height - 1))
        x2 = int(np.clip(x2, 0, width))
        y2 = int(np.clip(y2, 0, height))
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None
