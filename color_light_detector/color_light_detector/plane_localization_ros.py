"""ROS pose input and per-frame CPU localization, independent of lamp state."""

import json
import math
from pathlib import Path
import time

import cv2
import yaml

from .camera_geometry import CameraCalibration
from .plane_localization import DEFAULTS, PlaneProjector, PoseBuffer, validate


def invalid(reason):
    return {"valid": False, "status": reason, "method": "RAY_PLANE",
            "frame_id": None, "xyz_m": None, "world_valid": False,
            "world_xyz_m": None, "body_xyz_m": None, "camera_xyz_m": None,
            "range_m": None, "height_above_plane_m": None, "pixel_uv": None, "horizontal_sigma_m": None,
            "covariance_enu_m2": None, "estimate": None}


class PlaneLocalizationROS:
    """Own a bounded pose buffer; processing never waits for ROS or flight data."""

    def __init__(self, node, enabled, config_file, result_topic):
        self.node, self.enabled = node, enabled
        self.p = dict(DEFAULTS)
        self.projector = None
        self.buffer = None
        self.subscription = None
        self.publisher = None
        self.config_error = "DISABLED"
        self.config_detail = ""
        self.last_log_status = None
        self.last_log_time = 0.0
        if not enabled:
            return
        from std_msgs.msg import String
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        self.publisher = node.create_publisher(String, result_topic, QoSProfile(
            depth=5, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE))
        stage = "CONFIG"
        try:
            if not config_file:
                from ament_index_python.packages import get_package_share_directory
                config_file = str(Path(get_package_share_directory("color_light_detector")) /
                                  "config/task1_position_b0495_provisional.yaml")
            path = Path(config_file).expanduser()
            if not path.is_file():
                raise ValueError("POSITION_CONFIG_MISSING")
            self.p = validate(yaml.safe_load(path.read_text(encoding="utf-8")))
            self.buffer = PoseBuffer(self.p)
            from geometry_msgs.msg import PoseStamped
            from nav_msgs.msg import Odometry
            message_type = PoseStamped if self.p["pose_type"] == "pose_stamped" else Odometry
            self.subscription = node.create_subscription(message_type, self.p["pose_topic"], self._pose,
                QoSProfile(depth=64, reliability=ReliabilityPolicy.BEST_EFFORT,
                           durability=DurabilityPolicy.VOLATILE))
            calibration_file = Path(self.p["calibration_file"]).expanduser()
            if not calibration_file.is_absolute():
                calibration_file = path.parent / calibration_file
            self.p["calibration_file"] = str(calibration_file.resolve())
            stage = "CALIBRATION"
            calibration = CameraCalibration(calibration_file)
            self.projector = PlaneProjector(self.p, calibration)
            self.config_error = ""
        except (ValueError, KeyError, TypeError, OSError, ImportError, yaml.YAMLError) as error:
            self.config_error = ("CALIBRATION_INVALID" if self.p["calibration_confirmed"] else "CALIBRATION_REQUIRED") if stage == "CALIBRATION" else "POSITION_CONFIG_INVALID"
            self.config_detail = str(error)
            node.get_logger().warning("3D position unavailable: " + str(error))
        self._log_status(self.readiness())

    def unconfirmed_parameters(self):
        missing = [key for key in ("calibration_confirmed", "mount_confirmed", "height_reference_confirmed",
                                  "time_alignment_confirmed")
                   if not self.p[key]]
        if self.projector is not None and self.projector.camera.approximate:
            missing.append("nominal_camera_intrinsics")
        if self.p["height_reference"] == "UNCONFIRMED":
            missing.append("height_reference")
        if self.p["pose_has_xy"] and self.p["map_origin_id"] == "UNCONFIRMED":
            missing.append("map_origin_id")
        return missing

    def readiness(self):
        if self.config_error:
            return self.config_error
        missing = self.unconfirmed_parameters()
        if missing:
            return "PROVISIONAL" if self.p["allow_provisional"] else "UNCONFIRMED:" + ",".join(missing)
        return "READY"

    def _pose(self, message):
        try:
            p = self.p
            if message.header.frame_id != p["pose_frame_id"]:
                raise ValueError("POSE_FRAME_MISMATCH")
            if p["pose_type"] == "odometry":
                if message.child_frame_id != p["pose_body_frame_id"]:
                    raise ValueError("POSE_BODY_FRAME_MISMATCH")
                pose = message.pose.pose
            else:
                pose = message.pose
            stamp = float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9
            stamp += p["pose_time_offset_seconds"]
            now = self.node.get_clock().now().nanoseconds * 1e-9
            if stamp <= 0 or abs(now-stamp) > p["max_pose_clock_skew_seconds"]:
                raise ValueError("POSE_CLOCK_MISMATCH")
            q, xyz = pose.orientation, pose.position
            # Never treat unknown x/y as actual map coordinates. In relative-only
            # mode only altitude and attitude enter the intersection geometry.
            position = [xyz.x, xyz.y, xyz.z] if p["pose_has_xy"] else [0., 0., xyz.z]
            self.buffer.add(stamp, position, [q.x, q.y, q.z, q.w])
        except (ValueError, TypeError, AttributeError) as error:
            self.buffer.reject(str(error))

    def _log_status(self, status):
        now = time.monotonic()
        if status != self.last_log_status and now - self.last_log_time >= 3:
            self.node.get_logger().info("3D: " + status)
            self.last_log_status, self.last_log_time = status, now

    def locate(self, observations, shape, header):
        started = time.perf_counter()
        height, width = shape[:2]
        stamp = float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9
        effective_stamp = stamp + self.p["image_time_offset_seconds"]
        status = self.readiness()
        summary = {"enabled": self.enabled, "status": status, "device": "cpu",
                   "configuration_detail": self.config_detail,
                   "input_image_size": [width, height], "effective_image_stamp": effective_stamp,
                   "height_reference": self.p["height_reference"], "plane_z_m": self.p["plane_z_m"],
                   "world_frame_id": self.p["output_frame_id"], "body_frame_id": self.p["body_frame_id"],
                   "map_origin_id": self.p["map_origin_id"],
                   "camera_frame_id": self.p["camera_frame_id"], "pose_has_xy": self.p["pose_has_xy"],
                   "pose_topic": self.p["pose_topic"], "pose_type": self.p["pose_type"],
                   "pose_convention": self.p["pose_convention"],
                   "world_convention": "ENU", "body_convention": "FLU",
                   "camera_convention": "OPTICAL_RIGHT_DOWN_FORWARD",
                   "pose_reset_detection": "LOCAL_BUFFER_DISCONTINUITY",
                   "geodetic_coordinates": False,
                   "unconfirmed_parameters": self.unconfirmed_parameters(),
                   "camera_model_source": self.projector.camera.source if self.projector else None,
                   "mount_rotation_mode": self.p["mount_rotation_mode"],
                   "camera_downward_angle_deg": (self.p["camera_downward_angle_deg"]
                       if self.p["mount_rotation_mode"] == "downward_angle" else None)}
        positions = {}
        if status in ("READY", "PROVISIONAL"):
            try:
                if header.frame_id != self.p["camera_frame_id"]:
                    raise ValueError("CAMERA_FRAME_MISMATCH")
                now = self.node.get_clock().now().nanoseconds * 1e-9
                age = now - effective_stamp
                if stamp <= 0 or not math.isfinite(age) or age < -0.05 or age > self.p["max_image_age_seconds"]:
                    raise ValueError("IMAGE_TIME_MISMATCH")
                pos, r, timing = self.buffer.at(effective_stamp)
                summary["pose_alignment"] = timing
                # The configured plane and map origin belong to the original
                # pose reference. New samples cannot reconfirm that reference.
                if timing["reset_count"] != 0:
                    raise ValueError("POSE_REFERENCE_RESET_RECONFIRM_REQUIRED")
                summary["body_height_above_plane_m"] = float(pos[2] - self.p["plane_z_m"])
                self.projector.camera.matrix(width, height, self.p["allow_calibration_resize"])
            except ValueError as error:
                status = str(error)
        for item in observations:
            result = invalid(status)
            if status in ("READY", "PROVISIONAL"):
                try:
                    if not item["visible"]:
                        raise ValueError("ROI_UNOBSERVED")
                    if item["motion_quality"] < self.p["min_motion_quality"]:
                        raise ValueError("ROI_MOTION_QUALITY_LOW")
                    x1, y1, x2, y2 = item["bbox"]
                    if not all(math.isfinite(float(v)) for v in (x1, y1, x2, y2, item["motion_quality"])):
                        raise ValueError("ROI_NONFINITE")
                    if min(x2-x1, y2-y1) < self.p["min_bbox_side_pixels"]:
                        raise ValueError("ROI_TOO_SMALL")
                    if x1 <= 1 or y1 <= 1 or x2 >= width-1 or y2 >= height-1:
                        raise ValueError("ROI_TRUNCATED_AT_IMAGE_EDGE")
                    if "pixel_uv" in item:
                        uv = item["pixel_uv"]
                        if uv is None:
                            raise ValueError("TARGET_ANCHOR_UNAVAILABLE")
                        if not isinstance(uv, (list, tuple)) or len(uv) != 2:
                            raise ValueError("TARGET_ANCHOR_INVALID")
                        try:
                            uv = tuple(float(value) for value in uv)
                        except (ValueError, TypeError, OverflowError) as error:
                            raise ValueError("TARGET_ANCHOR_INVALID") from error
                        if not all(math.isfinite(value) for value in uv) or not (0 <= uv[0] < width and 0 <= uv[1] < height):
                            raise ValueError("TARGET_ANCHOR_INVALID")
                    else:
                        cx, cy = self.p["pixel_anchor_xy"]
                        uv = (x1 + cx*(x2-x1), y1 + cy*(y2-y1))
                    result = self.projector.project(uv, max(x2-x1, y2-y1), width, height, (pos, r), timing)
                    result["pixel_source"] = item.get("pixel_source", "TRACKED_ROI_ANCHOR")
                    result["estimate"] = None
                    if status == "PROVISIONAL":
                        # Preserve the existing valid=false/null contract for planners.
                        # Numeric test estimates remain available in a separate field.
                        keys = ("frame_id", "xyz_m", "world_xyz_m", "body_xyz_m", "camera_xyz_m",
                                "range_m", "height_above_plane_m", "horizontal_sigma_m", "covariance_enu_m2")
                        result["estimate"] = {key: result[key] for key in keys}
                        result.update(valid=False, world_valid=False, status="PROVISIONAL",
                                      xyz_m=None, world_xyz_m=None, body_xyz_m=None,
                                      camera_xyz_m=None, covariance_enu_m2=None, range_m=None,
                                      height_above_plane_m=None, horizontal_sigma_m=None)
                    result["unconfirmed_parameters"] = summary["unconfirmed_parameters"]
                except (ValueError, cv2.error) as error:
                    result = invalid(str(error))
            result["effective_image_stamp"] = effective_stamp
            positions[item["track_id"]] = result
        summary["status"] = status
        summary["valid_count"] = sum(value["valid"] for value in positions.values())
        summary["estimate_count"] = sum(value.get("estimate") is not None for value in positions.values())
        summary["processing_ms"] = round((time.perf_counter()-started)*1000, 3)
        self._log_status(status)
        return positions, summary

    def publish(self, payload):
        if self.publisher is None:
            return
        from std_msgs.msg import String
        message = String()
        message.data = json.dumps({
            "schema_version": 2, "session_id": payload["session_id"],
            "algorithm_version": payload["algorithm_version"], "sequence": payload["sequence"],
            "stamp": payload["stamp"], "image_frame_id": payload["frame_id"],
            "timestamp_source": payload.get("timestamp_source", "unspecified"),
            "localization": payload["localization"],
            "tracks": [{key: track[key] for key in
                        ("track_id", "visible", "state", "state_valid", "position")}
                       for track in payload["tracks"]],
        }, separators=(",", ":"), allow_nan=False)
        self.publisher.publish(message)
