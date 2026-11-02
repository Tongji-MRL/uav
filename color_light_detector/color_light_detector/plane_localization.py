"""CPU ray/plane intersections with timestamped UAV poses.

All internal world vectors use ENU; body vectors use FLU. Camera vectors use
optical axes (right, down, forward). A height constraint provides metric scale;
this is model-based localization, not measured camera depth.
"""

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
import math
import threading

import numpy as np


DEFAULTS = {
    "calibration_file": "camera_intrinsics.yaml",
    "allow_provisional": False,
    "calibration_confirmed": False,
    "mount_confirmed": False,
    "height_reference_confirmed": False,
    "time_alignment_confirmed": False,
    "height_reference": "UNCONFIRMED",
    "pose_topic": "/mavros/local_position/pose",
    "pose_type": "pose_stamped",
    "pose_frame_id": "map",
    "output_frame_id": "map",
    "map_origin_id": "UNCONFIRMED",
    "body_frame_id": "base_link",
    "pose_body_frame_id": "base_link",
    "camera_frame_id": "UNCONFIRMED",
    "pose_convention": "ENU_FLU",
    "pose_has_xy": False,
    "body_from_optical_xyz_m": [0.0, 0.0, 0.0],
    "mount_rotation_mode": "quaternion",
    "camera_downward_angle_deg": 90.0,
    "body_from_optical_xyzw": [0.7071067811865476, -0.7071067811865476, 0.0, 0.0],
    "plane_z_m": 0.0,
    "pixel_anchor_xy": [0.5, 0.5],
    "allow_calibration_resize": False,
    "image_time_offset_seconds": 0.0,
    "pose_time_offset_seconds": 0.0,
    "max_image_age_seconds": 0.5,
    "max_pose_clock_skew_seconds": 0.5,
    "max_pose_gap_seconds": 0.12,
    "nearest_pose_seconds": 0.04,
    "min_motion_quality": 0.60,
    "min_bbox_side_pixels": 12.0,
    "min_downward_cosine": 0.25,
    "min_height_above_plane_m": 0.3,
    "max_range_m": 150.0,
    "pixel_sigma_px": 2.0,
    "anchor_sigma_fraction": 0.04,
    "pose_position_sigma_m": 0.5,
    "pose_angle_sigma_deg": 2.0,
    "plane_height_sigma_m": 0.15,
    "max_horizontal_sigma_m": 5.0,
    "max_pose_speed_m_s": 30.0,
    "max_pose_angular_speed_deg_s": 300.0,
}

NED_TO_ENU = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
FRD_TO_FLU = np.diag([1., -1., -1.])


def quaternion(value):
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("INVALID_QUATERNION")
    norm = float(np.linalg.norm(q))
    if not 0.9 <= norm <= 1.1:
        raise ValueError("INVALID_QUATERNION_NORM")
    return q / norm


def rotation(value):
    x, y, z, w = quaternion(value)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def slerp(a, b, fraction):
    a, b = quaternion(a), quaternion(b)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    dot = float(np.clip(dot, -1, 1))
    if dot > 0.9995:
        return quaternion(a + fraction * (b - a))
    angle = math.acos(dot)
    return (math.sin((1-fraction)*angle)*a + math.sin(fraction*angle)*b) / math.sin(angle)


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def mounting_rotation(parameters):
    """Optical to FLU; depression is 0 degrees forward, 90 straight down."""
    if parameters["mount_rotation_mode"] == "quaternion":
        return rotation(parameters["body_from_optical_xyzw"])
    angle = math.radians(parameters["camera_downward_angle_deg"])
    sine, cosine = math.sin(angle), math.cos(angle)
    # Image right is body right; at nadir image top is body forward.
    return np.array([[0., -sine, cosine], [-1., 0., 0.], [0., -cosine, -sine]])


def validate(values):
    if not isinstance(values, dict):
        raise ValueError("Position config must be a YAML mapping")
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown position parameters: " + ", ".join(sorted(unknown)))
    p = dict(DEFAULTS, **values)
    for key, default in DEFAULTS.items():
        if isinstance(default, bool) and not isinstance(p[key], bool):
            raise ValueError(key + " must be true or false")
        if isinstance(default, str) and (not isinstance(p[key], str) or not p[key].strip()):
            raise ValueError(key + " must be a nonempty string")
        if isinstance(default, float):
            if isinstance(p[key], bool) or not isinstance(p[key], (float, int)) or not math.isfinite(p[key]):
                raise ValueError(key + " must be a finite number")
            p[key] = float(p[key])
    for key, size in (("body_from_optical_xyz_m", 3), ("pixel_anchor_xy", 2)):
        a = np.asarray(p[key], dtype=float)
        if a.shape != (size,) or not np.isfinite(a).all():
            raise ValueError(key + " has invalid dimensions/values")
    quaternion(p["body_from_optical_xyzw"])
    if p["mount_rotation_mode"] not in ("quaternion", "downward_angle"):
        raise ValueError("mount_rotation_mode must be quaternion or downward_angle")
    if np.linalg.norm(p["body_from_optical_xyz_m"]) > 5:
        raise ValueError("Camera mounting translation exceeds 5 meters")
    if p["pose_type"] not in ("pose_stamped", "odometry"):
        raise ValueError("pose_type must be pose_stamped or odometry")
    if p["pose_convention"] not in ("ENU_FLU", "NED_FRD"):
        raise ValueError("pose_convention must be ENU_FLU or NED_FRD")
    if p["pose_convention"] == "NED_FRD" and p["pose_frame_id"] == p["output_frame_id"]:
        raise ValueError("NED input and ENU output must have different frame IDs")
    if p["pose_convention"] == "ENU_FLU" and p["pose_frame_id"] != p["output_frame_id"]:
        raise ValueError("ENU input/output must share a frame; this node does not transform map origins")
    if p["pose_convention"] == "ENU_FLU" and p["pose_body_frame_id"] != p["body_frame_id"]:
        raise ValueError("ENU_FLU input/output must share the same body frame")
    if p["pose_convention"] == "NED_FRD" and p["pose_body_frame_id"] == p["body_frame_id"]:
        raise ValueError("FRD input and FLU output must have different body frame IDs")
    if any(not 0 <= float(v) <= 1 for v in p["pixel_anchor_xy"]):
        raise ValueError("pixel_anchor_xy must be within [0, 1]")
    bounds = {
        "camera_downward_angle_deg": (0, 90),
        "image_time_offset_seconds": (-1, 1), "pose_time_offset_seconds": (-1, 1),
        "max_image_age_seconds": (0.01, 1), "max_pose_clock_skew_seconds": (0.01, 2),
        "max_pose_gap_seconds": (0.01, 0.5), "nearest_pose_seconds": (0, 0.10),
        "min_motion_quality": (0, 1), "min_bbox_side_pixels": (2, 200),
        "min_downward_cosine": (0.1, 1), "min_height_above_plane_m": (0.05, 100),
        "max_range_m": (1, 1000), "pixel_sigma_px": (0.1, 50),
        "anchor_sigma_fraction": (0, 0.5), "pose_position_sigma_m": (0.001, 100),
        "pose_angle_sigma_deg": (0.01, 45), "plane_height_sigma_m": (0.001, 10),
        "max_horizontal_sigma_m": (0.01, 100), "max_pose_speed_m_s": (0.1, 100),
        "max_pose_angular_speed_deg_s": (1, 1000),
    }
    for key, (lo, hi) in bounds.items():
        if not lo <= p[key] <= hi:
            raise ValueError(f"{key} must be within [{lo}, {hi}]")
    if p["nearest_pose_seconds"] > p["max_pose_gap_seconds"]:
        raise ValueError("nearest_pose_seconds exceeds max_pose_gap_seconds")
    return p


@dataclass(frozen=True)
class PoseSample:
    stamp: float
    position: np.ndarray
    q: np.ndarray


class PoseBuffer:
    """A bounded, locked buffer; never interpolate across a reset or long gap."""

    def __init__(self, parameters):
        self.p = parameters
        self.lock = threading.Lock()
        self.samples = deque(maxlen=512)
        self.last_error = "NO_POSE"
        self.reset_count = 0

    def reject(self, reason):
        with self.lock:
            if self.samples:
                self.reset_count += 1
            self.samples.clear()
            self.last_error = str(reason)

    def add(self, stamp, position, q):
        xyz, q = np.asarray(position, dtype=float), quaternion(q)
        if not math.isfinite(stamp) or stamp <= 0 or xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("INVALID_POSE")
        with self.lock:
            if self.samples:
                old = self.samples[-1]
                dt = stamp - old.stamp
                if -0.5 < dt <= 0:
                    return  # late/duplicate packet, not a new observation
                distance = np.linalg.norm(xyz - old.position)
                angle = 2 * math.acos(float(np.clip(abs(np.dot(q, old.q)), 0, 1)))
                if dt <= -0.5 or dt > 1.0 or (dt > 0 and (
                        distance / dt > self.p["max_pose_speed_m_s"] or
                        math.degrees(angle) / dt > self.p["max_pose_angular_speed_deg_s"])):
                    self.samples.clear()
                    self.reset_count += 1
                    self.last_error = "POSE_RESET_WAITING"
            self.samples.append(PoseSample(float(stamp), xyz.copy(), q.copy()))
            while self.samples and stamp - self.samples[0].stamp > 4:
                self.samples.popleft()
            if len(self.samples) >= 2:
                self.last_error = "OK"

    def at(self, stamp):
        with self.lock:
            items, error, resets = list(self.samples), self.last_error, self.reset_count
        if len(items) < 2:
            raise ValueError(error)
        index = bisect_left([s.stamp for s in items], stamp)
        if 0 < index < len(items):
            a, b = items[index-1], items[index]
            gap = b.stamp - a.stamp
            if gap > self.p["max_pose_gap_seconds"]:
                raise ValueError("POSE_GAP_TOO_LARGE")
            f = (stamp - a.stamp) / gap
            pos, q = a.position + f * (b.position - a.position), slerp(a.q, b.q, f)
            timing = {"method": "INTERPOLATED", "offset_seconds": 0.0, "bracket_seconds": gap}
        else:
            selected = items[0] if index == 0 else items[-1]
            offset = selected.stamp - stamp
            if abs(offset) > self.p["nearest_pose_seconds"]:
                raise ValueError("POSE_TIME_MISMATCH")
            # Also require fresh, contiguous history for the nearest-pose fallback.
            pair = items[:2] if index == 0 else items[-2:]
            if pair[1].stamp - pair[0].stamp > self.p["max_pose_gap_seconds"]:
                raise ValueError("POSE_GAP_TOO_LARGE")
            pos, q = selected.position.copy(), selected.q.copy()
            timing = {"method": "NEAREST", "offset_seconds": offset, "bracket_seconds": None}
        r = rotation(q)
        if self.p["pose_convention"] == "NED_FRD":
            pos, r = NED_TO_ENU @ pos, NED_TO_ENU @ r @ FRD_TO_FLU
        timing["reset_count"] = resets
        return pos, r, timing


class PlaneProjector:
    def __init__(self, parameters, calibration):
        self.p, self.camera = parameters, calibration
        self.r_bc = mounting_rotation(parameters)
        self.t_bc = np.asarray(parameters["body_from_optical_xyz_m"], dtype=float)

    def project(self, uv, box_side, width, height, pose, timing):
        p = self.p
        origin_body, r_wb = pose
        lever = r_wb @ self.t_bc
        origin = origin_body + lever
        height_delta = p["plane_z_m"] - origin[2]
        if -height_delta < p["min_height_above_plane_m"]:
            raise ValueError("CAMERA_NOT_ABOVE_PLANE")
        # Four neighbors supply a distortion-aware pixel Jacobian in one tiny CPU call.
        u, v = uv
        pixels = [(u, v), (u+.5, v), (u-.5, v), (u, v+.5), (u, v-.5)]
        rays_camera = self.camera.rays(pixels, width, height, p["allow_calibration_resize"])
        directions = rays_camera @ (r_wb @ self.r_bc).T
        if np.any(directions[:, 2] > -p["min_downward_cosine"]):
            raise ValueError("RAY_TOO_CLOSE_TO_HORIZON")
        distances = height_delta / directions[:, 2]
        if not np.isfinite(distances).all() or np.any(distances <= 0):
            raise ValueError("NO_FORWARD_INTERSECTION")
        if distances[0] > p["max_range_m"]:
            raise ValueError("RANGE_EXCEEDED")
        points = origin + distances[:, None] * directions
        if not np.isfinite(points).all():
            raise ValueError("NONFINITE_INTERSECTION")
        world = points[0]
        body = r_wb.T @ (world - origin_body)
        camera = self.r_bc.T @ (body - self.t_bc)

        # Approximate first-order covariance, driven by explicit configured error
        # assumptions, not by YOLO confidence or an asserted measured accuracy.
        direction, distance = directions[0], distances[0]
        jo = np.eye(3) - np.outer(direction, [0., 0., 1.]) / direction[2]
        jd = distance * jo
        jr = -jo @ skew(lever) - jd @ skew(direction)
        juv = np.column_stack((points[1]-points[2], points[3]-points[4]))
        jh = direction / direction[2]
        pixel_sigma = max(p["pixel_sigma_px"], p["anchor_sigma_fraction"] * box_side)
        # Include bounded timing error for a nearest-pose lookup. For interpolation,
        # the user-supplied pose error must include motion between pose samples.
        dt = abs(timing["offset_seconds"])
        pos_sigma = p["pose_position_sigma_m"] + dt * p["max_pose_speed_m_s"]
        angle_sigma = math.radians(p["pose_angle_sigma_deg"] + dt * p["max_pose_angular_speed_deg_s"])
        covariance = (juv @ juv.T * pixel_sigma**2 + jo @ jo.T * pos_sigma**2
                      + jr @ jr.T * angle_sigma**2 + np.outer(jh, jh) * p["plane_height_sigma_m"]**2)
        if not np.isfinite(covariance).all():
            raise ValueError("POSITION_UNCERTAINTY_TOO_HIGH")
        sigma_xy = math.sqrt(max(0., float(np.linalg.eigvalsh(covariance[:2, :2])[-1])))
        if sigma_xy > p["max_horizontal_sigma_m"]:
            raise ValueError("POSITION_UNCERTAINTY_TOO_HIGH")
        has_xy = p["pose_has_xy"]
        xyz = world if has_xy else body
        return {
            "valid": True, "status": "OK", "method": "RAY_PLANE",
            "frame_id": p["output_frame_id"] if has_xy else p["body_frame_id"],
            "xyz_m": xyz.tolist(), "world_valid": has_xy,
            "world_xyz_m": world.tolist() if has_xy else None,
            "body_xyz_m": body.tolist(), "camera_xyz_m": camera.tolist(),
            "range_m": float(distance), "downward_cosine": float(-direction[2]),
            "height_above_plane_m": float(-height_delta),
            "pixel_uv": [float(u), float(v)], "pixel_source": "TRACKED_ROI_ANCHOR",
            "plane_z_m": p["plane_z_m"], "height_reference": p["height_reference"],
            "horizontal_sigma_m": sigma_xy,
            "covariance_enu_m2": covariance.reshape(-1).tolist() if has_xy else None,
            "uncertainty_model": "CONFIGURED_FIRST_ORDER_APPROXIMATION",
            "pose_alignment": timing,
        }
