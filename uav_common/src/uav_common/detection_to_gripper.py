"""Convert RGB-D detections into the gripper coordinate frame."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - deployment dependency
    cv2 = None

from uav_common.map_detection_to_ned import (
    finite_vector,
    required_transform_4x4,
)


class DetectionToGripper:
    """Transform detector objects from camera optical coordinates to gripper."""

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
        self.t_gripper_camera = required_transform_4x4(
            transforms.get("T_gripper_camera"),
            "transforms.T_gripper_camera",
        )

        if self.rgb_width <= 0 or self.rgb_height <= 0:
            raise ValueError("camera image dimensions must be positive")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        if self.distortion and cv2 is None:
            raise RuntimeError("OpenCV is required for distortion correction")

    def transform(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return detections expressed in the configured gripper frame."""
        image_size = payload.get("rgb_image_size", {})
        image_width = float(image_size.get("width", self.rgb_width))
        image_height = float(image_size.get("height", self.rgb_height))

        objects: list[dict[str, Any]] = []
        for detection in payload.get("detections", []):
            transformed = self._transform_detection(
                detection,
                image_width,
                image_height,
            )
            if transformed is not None:
                objects.append(transformed)

        return {"frame": "gripper", "objects": objects}

    def _transform_detection(
        self,
        detection: dict[str, Any],
        image_width: float,
        image_height: float,
    ) -> dict[str, Any] | None:
        center = finite_vector(detection.get("center_xy"), 2)
        if center is None:
            return None

        try:
            depth_m = float(detection.get("depth_median_m"))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(depth_m):
            return None
        if not self.depth_min_m <= depth_m <= self.depth_max_m:
            return None

        point_camera = self._pixel_depth_to_camera(
            center[0],
            center[1],
            depth_m,
            image_width,
            image_height,
        )
        point_gripper = self._transform_point(
            self.t_gripper_camera,
            point_camera,
        )

        return {
            "type": str(
                detection.get(
                    "object_type",
                    detection.get("class_name", ""),
                )
            ),
            "color": str(detection.get("color", "None")),
            "state": str(detection.get("state", "static")),
            "id": detection.get("id"),
            "gripper_position": point_gripper,
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
            coefficients = np.asarray(
                self.distortion,
                dtype=np.float64,
            ).reshape(-1, 1)
            pixel = np.array([[[u, v]]], dtype=np.float64)
            normalized = cv2.undistortPoints(
                pixel,
                camera_matrix,
                coefficients,
            )
            x_normalized = float(normalized[0, 0, 0])
            y_normalized = float(normalized[0, 0, 1])
        else:
            x_normalized = (u - cx) / fx
            y_normalized = (v - cy) / fy

        return [
            x_normalized * depth_m,
            y_normalized * depth_m,
            depth_m,
        ]

    @staticmethod
    def _transform_point(
        transform: Sequence[Sequence[float]],
        point: Sequence[float],
    ) -> list[float]:
        return [
            transform[row][0] * point[0]
            + transform[row][1] * point[1]
            + transform[row][2] * point[2]
            + transform[row][3]
            for row in range(3)
        ]
