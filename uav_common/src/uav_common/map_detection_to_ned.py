"""Pure camera-to-PX4-local-NED coordinate transformations."""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - checked during construction when needed
    cv2 = None


def finite_vector(value: Any, length: int) -> Optional[list[float]]:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != length:
        return None
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return vector if all(math.isfinite(item) for item in vector) else None


def normalize_quaternion(value: Any) -> list[float]:
    quaternion = finite_vector(value, 4)
    if quaternion is None:
        raise ValueError("quaternion must contain four finite values")
    norm = math.sqrt(sum(item * item for item in quaternion))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("quaternion norm is invalid")
    return [item / norm for item in quaternion]


def yaw_from_quaternion(value: Any) -> float:
    """Return NED yaw from a PX4 body-FRD quaternion [w, x, y, z]."""
    w, x, y, z = normalize_quaternion(value)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def quat_rotate_frd_to_ned(
    quaternion: Sequence[float], vector: Sequence[float]
) -> list[float]:
    """Rotate a body-FRD vector into NED using PX4 q=[w, x, y, z]."""
    w, x, y, z = normalize_quaternion(quaternion)
    v = finite_vector(vector, 3)
    if v is None:
        raise ValueError("vector must contain three finite values")
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def required_transform_4x4(value: Any, field_name: str) -> list[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{field_name} must be a 4x4 homogeneous transform")
    transform: list[list[float]] = []
    for row in value:
        parsed = finite_vector(row, 4)
        if parsed is None:
            raise ValueError(f"{field_name} must contain finite numeric values")
        transform.append(parsed)

    expected = [0.0, 0.0, 0.0, 1.0]
    if any(abs(transform[3][index] - expected[index]) > 1e-9 for index in range(4)):
        raise ValueError(f"{field_name} last row must be [0, 0, 0, 1]")

    rotation = np.asarray([row[:3] for row in transform[:3]], dtype=float)
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
        raise ValueError(f"{field_name} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError(f"{field_name} rotation must have determinant +1")
    return transform


class MapDetectionToNed:
    """Convert a detector payload using a supplied PX4 pose snapshot."""

    def __init__(self, camera: dict[str, Any], transforms: dict[str, Any]) -> None:
        self.rgb_width = int(camera.get("rgb_width", 1280))
        self.rgb_height = int(camera.get("rgb_height", 720))
        self.fx = float(camera.get("fx", 900.0))
        self.fy = float(camera.get("fy", 900.0))
        self.cx = float(camera.get("cx", self.rgb_width * 0.5))
        self.cy = float(camera.get("cy", self.rgb_height * 0.5))
        self.distortion = [float(value) for value in camera.get("distortion", [])]
        self.depth_min_m = float(camera.get("depth_min_m", 0.05))
        self.depth_max_m = float(camera.get("depth_max_m", 20.0))
        self.t_body_camera = required_transform_4x4(
            transforms.get("T_body_camera"), "transforms.T_body_camera"
        )

        if self.rgb_width <= 0 or self.rgb_height <= 0:
            raise ValueError("camera image dimensions must be positive")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        if not all(
            math.isfinite(value) for value in (self.fx, self.fy, self.cx, self.cy)
        ):
            raise ValueError("camera intrinsics must be finite")
        if self.depth_min_m <= 0.0 or self.depth_max_m <= self.depth_min_m:
            raise ValueError("camera depth range is invalid")
        if self.distortion and cv2 is None:
            raise RuntimeError("OpenCV is required for distortion correction")

    def transform(
        self,
        payload: dict[str, Any],
        position_ned: Sequence[float],
        attitude_q: Sequence[float],
    ) -> dict[str, list[dict[str, Any]]]:
        position = finite_vector(position_ned, 3)
        if position is None:
            return {"objects": []}
        try:
            quaternion = normalize_quaternion(attitude_q)
        except ValueError:
            return {"objects": []}

        image_size = payload.get("rgb_image_size", {})
        if not isinstance(image_size, dict):
            image_size = {}
        image_width = self._positive_float(image_size.get("width"), self.rgb_width)
        image_height = self._positive_float(
            image_size.get("height"), self.rgb_height
        )
        detections = payload.get("detections", [])
        if not isinstance(detections, list):
            detections = []

        objects: list[dict[str, Any]] = []
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            transformed = self._transform_detection(
                detection, image_width, image_height, position, quaternion
            )
            if transformed is not None:
                objects.append(transformed)
        return {"objects": objects}

    def _transform_detection(
        self,
        detection: dict[str, Any],
        image_width: float,
        image_height: float,
        position_ned: Sequence[float],
        attitude_q: Sequence[float],
    ) -> Optional[dict[str, Any]]:
        center = finite_vector(detection.get("center_xy"), 2)
        if center is None:
            return None
        try:
            depth_m = float(detection.get("depth_median_m"))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(depth_m) or not self.depth_min_m <= depth_m <= self.depth_max_m:
            return None

        object_type = str(
            detection.get("object_type", detection.get("class_name", ""))
        ).strip()
        object_id = detection.get("id")
        if (
            not object_type
            or isinstance(object_id, bool)
            or not isinstance(object_id, int)
            or object_id < 0
        ):
            return None

        point_camera = self._pixel_depth_to_camera(
            center[0], center[1], depth_m, image_width, image_height
        )
        point_body = self._transform_point(self.t_body_camera, point_camera)
        offset_ned = quat_rotate_frd_to_ned(attitude_q, point_body)
        target_ned = [position_ned[index] + offset_ned[index] for index in range(3)]
        if not all(math.isfinite(value) for value in target_ned):
            return None
        return {
            "type": object_type,
            "color": str(detection.get("color", "None")),
            "state": str(detection.get("state", "static")),
            "id": object_id,
            "NED_position": target_ned,
        }

    def _pixel_depth_to_camera(
        self,
        u: float,
        v: float,
        depth_m: float,
        image_width: float,
        image_height: float,
    ) -> list[float]:
        scale_x = image_width / float(self.rgb_width)
        scale_y = image_height / float(self.rgb_height)
        fx = self.fx * scale_x
        fy = self.fy * scale_y
        cx = self.cx * scale_x
        cy = self.cy * scale_y
        if self.distortion:
            camera_matrix = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            dist_coeffs = np.asarray(self.distortion, dtype=np.float64).reshape(-1, 1)
            pixel = np.array([[[u, v]]], dtype=np.float64)
            normalized = cv2.undistortPoints(pixel, camera_matrix, dist_coeffs)
            x_norm = float(normalized[0, 0, 0])
            y_norm = float(normalized[0, 0, 1])
        else:
            x_norm = (u - cx) / fx
            y_norm = (v - cy) / fy
        return [x_norm * depth_m, y_norm * depth_m, depth_m]

    @staticmethod
    def _transform_point(
        transform: Sequence[Sequence[float]], point: Sequence[float]
    ) -> list[float]:
        return [
            transform[row][0] * point[0]
            + transform[row][1] * point[1]
            + transform[row][2] * point[2]
            + transform[row][3]
            for row in range(3)
        ]

    @staticmethod
    def _positive_float(value: Any, default: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return float(default)
        return result if math.isfinite(result) and result > 0.0 else float(default)
