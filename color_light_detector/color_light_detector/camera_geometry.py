"""Calibrated raw-image geometry; no neural network or GPU operations."""

from pathlib import Path

import cv2
import numpy as np
import yaml


class CameraCalibration:
    """Read the standard ROS monocular calibration YAML (raw K and D)."""

    def __init__(self, filename):
        path = Path(filename).expanduser()
        if not path.is_file():
            raise ValueError("Camera calibration file missing: " + str(path))
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise ValueError("Invalid camera calibration YAML") from error
        if not isinstance(data, dict):
            raise ValueError("Camera calibration must be a YAML mapping")
        self.approximate = data.get("calibration_is_approximate", False)
        if not isinstance(self.approximate, bool):
            raise ValueError("calibration_is_approximate must be true or false")
        self.source = str(data.get("calibration_source", "USER_CALIBRATION_FILE"))
        self.width = int(data["image_width"])
        self.height = int(data["image_height"])
        self.k = np.asarray(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
        self.d = np.asarray(data["distortion_coefficients"]["data"], dtype=np.float64).reshape(-1)
        self.model = str(data["distortion_model"])
        if self.width <= 0 or self.height <= 0 or not np.isfinite(self.k).all() or not np.isfinite(self.d).all():
            raise ValueError("Non-finite or invalid calibration dimensions")
        if self.k[0, 0] <= 0 or self.k[1, 1] <= 0:
            raise ValueError("Uncalibrated camera: focal lengths must be positive")
        if not np.allclose(self.k[2], [0, 0, 1]) or abs(self.k[0, 1]) > 1e-8 or abs(self.k[1, 0]) > 1e-8:
            raise ValueError("Only standard zero-skew camera matrices are supported")
        if not (0 <= self.k[0, 2] < self.width and 0 <= self.k[1, 2] < self.height):
            raise ValueError("Principal point outside calibrated image")
        sizes = {"plumb_bob": (4, 5), "rational_polynomial": (8, 12, 14), "equidistant": (4,)}
        if self.model not in sizes or self.d.size not in sizes[self.model]:
            raise ValueError("Unsupported distortion model or coefficient count")
        self.filename = str(path.resolve())

    def matrix(self, width, height, allow_resize=False):
        if (width, height) == (self.width, self.height):
            return self.k.copy()
        if not allow_resize:
            raise ValueError("CALIBRATION_SIZE_MISMATCH")
        sx, sy = width / self.width, height / self.height
        if width <= 0 or height <= 0 or abs(sx / sy - 1.0) > 0.002:
            raise ValueError("CALIBRATION_ASPECT_MISMATCH")
        # Matches cv2.resize pixel centers; use only a full-frame resize, never a crop.
        transform = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
        return transform @ self.k

    def rays(self, pixels, width, height, allow_resize=False):
        k = self.matrix(width, height, allow_resize)
        uv = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        if self.model == "equidistant":
            xy = cv2.fisheye.undistortPoints(uv, k, self.d)
        else:
            xy = cv2.undistortPoints(uv, k, self.d)
        rays = np.column_stack((xy.reshape(-1, 2), np.ones(len(uv))))
        if not np.isfinite(rays).all():
            raise ValueError("UNDISTORTION_FAILED")
        # Reject poorly inverted distortion rather than projecting a spurious ray.
        if self.model == "equidistant":
            reprojection, _ = cv2.fisheye.projectPoints(rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k, self.d)
        else:
            reprojection, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), k, self.d)
        if not np.isfinite(reprojection).all() or np.max(np.linalg.norm(reprojection.reshape(-1, 2) - uv.reshape(-1, 2), axis=1)) > 1.0:
            raise ValueError("UNDISTORTION_RESIDUAL_TOO_HIGH")
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def fill_info(self, message, width, height):
        k = self.matrix(width, height, allow_resize=True)
        message.distortion_model = self.model
        message.d = self.d.tolist()
        message.k = k.reshape(-1).tolist()
        message.r = np.eye(3).reshape(-1).tolist()
        message.p = np.column_stack((k, np.zeros(3))).reshape(-1).tolist()
