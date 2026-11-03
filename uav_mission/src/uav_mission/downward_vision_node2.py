#!/usr/bin/env python3
"""Mode-controlled RGB-D vision node for MAP and TASK3."""

from __future__ import annotations

from bisect import bisect_left
from collections import deque
import json
import math
from pathlib import Path
import time
from typing import Any

import rclpy
import yaml
from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import String

from uav_common.detection_to_gripper import DetectionToGripper
from uav_common.map_detection_to_ned import MapDetectionToNed
from uav_common.multi_rgbd_target_detector import (
    DepthFrame,
    MultiRgbdTargetDetector,
    decode_depth_image,
    decode_rgb_image,
    image_stamp_sec,
)


class DownwardVisionNode(Node):
    """Select a MAP or TASK3 RGB-D pipeline without controlling the UAV."""

    VALID_MODES = {"OFF", "MAP", "TASK3"}

    def __init__(self) -> None:
        super().__init__("downward_vision")
        self.declare_parameter("config_file", "")

        config = self._load_config()
        runtime = config.get("runtime", {})
        topics = config.get("topics", {})
        perception = config.get("perception", {})
        models = config.get("models", {})
        camera = config.get("camera", {})
        transforms = config.get("transforms", {})

        self.mode = "OFF"
        # None表示运行当前模式下的全部模型
        # 空集合表示不运行任何模型
        self.active_object_types: set[str] | None = set()

        self.latest_rgb: Image | None = None
        self.latest_depth: DepthFrame | None = None
        self.position_ned: list[float] | None = None
        self.attitude_q: list[float] | None = None
        self.last_processed_stamp: tuple[int, int] | None = None

        # PX4位姿按照timestamp_sample缓存，用于匹配RGB拍摄时间。
        self.position_buffer: deque[
            tuple[float, list[float]]
        ] = deque(maxlen=1000)
        self.attitude_buffer: deque[
            tuple[float, list[float]]
        ] = deque(maxlen=2000)
        self.pose_buffer_duration_sec = float(
            perception.get("pose_buffer_duration_sec", 2.0)
        )
        self.pose_sync_tolerance_sec = float(
            perception.get("pose_sync_tolerance_sec", 0.05)
        )
        if (
            not math.isfinite(self.pose_buffer_duration_sec)
            or self.pose_buffer_duration_sec <= 0.0
        ):
            raise ValueError(
                "perception.pose_buffer_duration_sec must be positive"
            )
        if (
            not math.isfinite(self.pose_sync_tolerance_sec)
            or self.pose_sync_tolerance_sec < 0.0
        ):
            raise ValueError(
                "perception.pose_sync_tolerance_sec must be non-negative"
            )
        self.last_pose_sync_warning_time = 0.0

        self.depth_unit = str(perception.get("depth_unit", "mm"))
        self.invalid_depth_values = list(
            perception.get("invalid_depth_values", [0])
        )

        # Both models are loaded once. Mode selection decides which one runs.
        self.map_detector = MultiRgbdTargetDetector(
            perception,
            models.get("map", []),
        )
        self.task3_detector = MultiRgbdTargetDetector(
            perception,
            models.get("task3", []),
        )

        self.map_to_ned = MapDetectionToNed(camera, transforms)
        self.task3_to_gripper = DetectionToGripper(camera, transforms)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.control_sub = self.create_subscription(
            String,
            str(
                topics.get(
                    "vision_control",
                    "/tongji_mrl/tsky/downward_vision_control",
                )
            ),
            self._on_control,
            10,
        )
        self.rgb_sub = self.create_subscription(
            Image,
            str(topics.get("rgb_image", "/camera/color/image_raw")),
            self._on_rgb_image,
            sensor_qos,
        )
        self.depth_sub = self.create_subscription(
            Image,
            str(
                topics.get(
                    "depth_image",
                    "/camera/aligned_depth_to_color/image_raw",
                )
            ),
            self._on_depth_image,
            sensor_qos,
        )
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            str(
                topics.get(
                    "vehicle_local_position",
                    "/fmu/out/vehicle_local_position_v1",
                )
            ),
            self._on_local_position,
            px4_qos,
        )
        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            str(
                topics.get(
                    "vehicle_attitude",
                    "/fmu/out/vehicle_attitude",
                )
            ),
            self._on_attitude,
            px4_qos,
        )

        self.map_result_pub = self.create_publisher(
            String,
            str(
                topics.get(
                    "map_result",
                    "/tongji_mrl/tsky/map_targets_ned",
                )
            ),
            10,
        )

        
        # 很多消息的发布
        self.map_debug_pub = self.create_publisher(
            String,
            str(
                topics.get(
                    "map_debug_result",
                    "/tongji_mrl/tsky/map_detection_debug",
                )
            ),
            10,
        )

        self.task3_debug_pub = self.create_publisher(
            String,
            str(
                topics.get(
                    "task3_debug_result",
                    "/tongji_mrl/tsky/task3_detection_debug",
                )
            ),
            10,
        )

        self.task3_result_pub = self.create_publisher(
            String,
            str(
                topics.get(
                    "task3_result",
                    "/tongji_mrl/tsky/task3_target_gripper",
                )
            ),
            10,
        )

        inference_rate_hz = float(runtime.get("inference_rate_hz", 3.0))
        if inference_rate_hz <= 0.0:
            raise ValueError("runtime.inference_rate_hz must be positive")
        self.inference_timer = self.create_timer(
            1.0 / inference_rate_hz,
            self._process_latest_frame,
        )

        self.get_logger().info(
            "Downward vision is ready; mode is OFF"
        )

    def _load_config(self) -> dict[str, Any]:
        config_file = str(self.get_parameter("config_file").value).strip()
        if not config_file:
            raise RuntimeError("config_file parameter is required")

        config_path = Path(config_file).expanduser()
        if not config_path.is_file():
            raise RuntimeError(f"Config file does not exist: {config_path}")

        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        config = loaded.get("downward_vision", loaded)
        if not isinstance(config, dict):
            raise ValueError("downward_vision config must be a mapping")
        return config

    def _on_control(self, msg: String) -> None:
        command = json.loads(msg.data)

        mode = str(command["mode"]).strip().upper()
        if mode not in self.VALID_MODES:
            self.get_logger().warning(
                f"Ignored unknown vision mode: {mode}"
            )
            return

        requested_types = {
            str(item).strip().lower()
            for item in command.get("object_types", [])
        }

        if mode == "OFF":
            active_types = set()
        elif "all" in requested_types:
            active_types = None
        else:
            active_types = requested_types

        if (
            mode != self.mode
            or active_types != self.active_object_types
        ):
            self.mode = mode
            self.active_object_types = active_types
            self.last_processed_stamp = None

            self.get_logger().info(
                f"Vision mode={mode}, "
                f"objects={active_types or 'ALL/NONE'}"
            )

    def _on_rgb_image(self, msg: Image) -> None:
        self.latest_rgb = msg

    def _on_depth_image(self, msg: Image) -> None:
        try:
            self.latest_depth = decode_depth_image(
                msg,
                self.depth_unit,
                self.invalid_depth_values,
            )
        except ValueError as exc:
            self.get_logger().warning(f"Ignored depth image: {exc}")

    def _on_local_position(self, msg: VehicleLocalPosition) -> None:
        position = [float(msg.x), float(msg.y), float(msg.z)]
        self.position_ned = position
        self._append_pose_sample(
            self.position_buffer,
            self._px4_sample_time_sec(msg),
            position,
        )

    def _on_attitude(self, msg: VehicleAttitude) -> None:
        attitude = [float(value) for value in msg.q]
        self.attitude_q = attitude
        self._append_pose_sample(
            self.attitude_buffer,
            self._px4_sample_time_sec(msg),
            attitude,
        )

    def _process_latest_frame(self) -> None:
        # OFF模式不推理；没有RGB图像也无法推理
        if self.mode == "OFF" or self.latest_rgb is None:
            return

        # 空集合表示当前不运行任何模型
        # None表示运行当前模式下的全部模型
        if (
            self.active_object_types is not None
            and not self.active_object_types
        ):
            return

        # 获取当前RGB图像的时间戳
        stamp = (
            int(self.latest_rgb.header.stamp.sec),
            int(self.latest_rgb.header.stamp.nanosec),
        )

        # 避免重复处理同一张图像
        if stamp == self.last_processed_stamp:
            return

        rgb_stamp_sec = image_stamp_sec(self.latest_rgb)
        matched_position, position_sync = (
            self._sample_pose_buffer(
                self.position_buffer,
                rgb_stamp_sec,
                quaternion=False,
            )
        )
        matched_attitude, attitude_sync = (
            self._sample_pose_buffer(
                self.attitude_buffer,
                rgb_stamp_sec,
                quaternion=True,
            )
        )
        pose_sync = {
            "status": (
                "matched"
                if matched_position is not None
                and matched_attitude is not None
                else "unavailable"
            ),
            "rgb_stamp_sec": rgb_stamp_sec,
            "position": position_sync,
            "attitude": attitude_sync,
        }

        try:
            # ROS Image转换成NumPy RGB图像
            rgb = decode_rgb_image(self.latest_rgb)

            # mode只负责决定使用哪组模型
            if self.mode == "MAP":
                detector = self.map_detector
            elif self.mode == "TASK3":
                detector = self.task3_detector
            else:
                return

            # 运行选中的一个、多个或全部模型
            detections, errors = detector.process(
                rgb,
                rgb_stamp_sec=rgb_stamp_sec,
                stamp_sec=stamp[0],
                stamp_nanosec=stamp[1],
                depth=self.latest_depth,
                enabled_object_types=self.active_object_types,
            )

        except Exception as exc:
            self.get_logger().error(
                f"Vision processing failed: {exc}"
            )
            self.last_processed_stamp = stamp
            return

        # 记录当前图像已经处理
        self.last_processed_stamp = stamp

        # 输出模型或深度处理错误
        for error in errors:
            self.get_logger().warning(error)

        # mode负责决定坐标转换方式
        if self.mode == "MAP":
            # 转换为local NED并发布
            self._publish_map_result(
                detections,
                matched_position,
                matched_attitude,
                pose_sync,
            )

        elif self.mode == "TASK3":
            # 转换为gripper坐标并发布
            self._publish_task3_result(
                detections,
                matched_position,
                matched_attitude,
                pose_sync,
            )

    def _publish_map_result(
        self,
        detections: dict[str, Any],
        matched_position_ned: list[float] | None,
        matched_attitude_q: list[float] | None,
        pose_sync: dict[str, Any],
    ) -> None:
        if (
            matched_position_ned is None
            or matched_attitude_q is None
        ):
            self._publish_pose_sync_warning(pose_sync)
            self._publish_json(
                self.map_debug_pub,
                self._build_debug_result(
                    "MAP_DEBUG",
                    "local_ned",
                    detections,
                    matched_position_ned,
                    matched_attitude_q,
                    pose_sync,
                    [],
                ),
            )
            return

        transformed = self.map_to_ned.transform(
            detections,
            matched_position_ned,
            matched_attitude_q,
        )

        # 保持原来的MAP输出格式，避免影响主节点
        map_result = {
            "task": "MAP",
            "frame": "local_ned",
            "objects": [
                {
                    "type": item["type"],
                    "color": item["color"],
                    "state": item["state"],
                    "id": item["id"],
                    "NED_position": item["NED_position"],
                }
                for item in transformed["objects"]
            ],
        }

        self._publish_json(
            self.map_result_pub,
            map_result,
        )

        # 完整调试信息
        debug_result = self._build_debug_result(
            "MAP_DEBUG",
            "local_ned",
            detections,
            matched_position_ned,
            matched_attitude_q,
            pose_sync,
            transformed.get("objects", []),
        )

        self._publish_json(
            self.map_debug_pub,
            debug_result,
        )

    def _publish_task3_result(
        self,
        detections: dict[str, Any],
        matched_position_ned: list[float] | None,
        matched_attitude_q: list[float] | None,
        pose_sync: dict[str, Any],
    ) -> None:
        transformed = self.task3_to_gripper.transform(
            detections
        )

        # 保持原有TASK3输出格式
        task3_result = {
            "task": "TASK3",
            "frame": "gripper",
            "objects": [
                {
                    "type": item["type"],
                    "color": item["color"],
                    "state": item["state"],
                    "id": item["id"],
                    "gripper_position": (
                        item["gripper_position"]
                    ),
                }
                for item in transformed.get("objects", [])
            ],
        }

        self._publish_json(
            self.task3_result_pub,
            task3_result,
        )

        # TASK3独立调试输出
        debug_result = self._build_debug_result(
            "TASK3_DEBUG",
            "gripper",
            detections,
            matched_position_ned,
            matched_attitude_q,
            pose_sync,
            transformed.get("objects", []),
        )

        self._publish_json(
            self.task3_debug_pub,
            debug_result,
        )

    @staticmethod
    def _px4_sample_time_sec(msg: Any) -> float:
        timestamp_us = int(
            getattr(msg, "timestamp_sample", 0)
            or getattr(msg, "timestamp", 0)
        )
        return timestamp_us * 1e-6

    def _append_pose_sample(
        self,
        buffer: deque[tuple[float, list[float]]],
        timestamp_sec: float,
        value: list[float],
    ) -> None:
        if (
            timestamp_sec <= 0.0
            or not math.isfinite(timestamp_sec)
            or not all(math.isfinite(item) for item in value)
        ):
            return

        # PX4重启导致时间回退时，丢弃旧时间域的数据。
        if buffer and timestamp_sec < buffer[-1][0]:
            if buffer[-1][0] - timestamp_sec > 1.0:
                buffer.clear()
            else:
                return

        buffer.append((timestamp_sec, list(value)))
        cutoff = timestamp_sec - self.pose_buffer_duration_sec
        while buffer and buffer[0][0] < cutoff:
            buffer.popleft()

    def _sample_pose_buffer(
        self,
        buffer: deque[tuple[float, list[float]]],
        target_time_sec: float,
        *,
        quaternion: bool,
    ) -> tuple[list[float] | None, dict[str, Any]]:
        if not buffer:
            return None, {"status": "empty"}

        samples = list(buffer)
        times = [item[0] for item in samples]
        index = bisect_left(times, target_time_sec)

        # 缓存边缘没有前后两帧时，允许使用足够接近的最近样本。
        if index == 0 or index == len(samples):
            nearest = samples[0] if index == 0 else samples[-1]
            delta = abs(nearest[0] - target_time_sec)
            info = {
                "status": (
                    "matched" if delta <= self.pose_sync_tolerance_sec
                    else "outside_tolerance"
                ),
                "method": "nearest",
                "sample_time_sec": nearest[0],
                "delta_sec": delta,
            }
            if delta > self.pose_sync_tolerance_sec:
                return None, info
            return list(nearest[1]), info

        before_time, before_value = samples[index - 1]
        after_time, after_value = samples[index]
        before_delta = target_time_sec - before_time
        after_delta = after_time - target_time_sec
        info = {
            "status": "matched",
            "method": "interpolated",
            "before_time_sec": before_time,
            "after_time_sec": after_time,
            "before_delta_sec": before_delta,
            "after_delta_sec": after_delta,
        }

        if (
            before_delta > self.pose_sync_tolerance_sec
            or after_delta > self.pose_sync_tolerance_sec
        ):
            info["status"] = "outside_tolerance"
            return None, info

        interval = after_time - before_time
        alpha = (
            0.0
            if interval <= 1e-9
            else before_delta / interval
        )
        if quaternion:
            try:
                value = self._slerp(
                    before_value,
                    after_value,
                    alpha,
                )
            except ValueError as exc:
                info["status"] = "invalid_sample"
                info["error"] = str(exc)
                return None, info
        else:
            value = [
                before_value[i]
                + alpha * (after_value[i] - before_value[i])
                for i in range(len(before_value))
            ]
        return value, info

    @staticmethod
    def _slerp(
        first: list[float],
        second: list[float],
        alpha: float,
    ) -> list[float]:
        first_norm = math.sqrt(sum(value * value for value in first))
        second_norm = math.sqrt(sum(value * value for value in second))
        if first_norm <= 1e-9 or second_norm <= 1e-9:
            raise ValueError("Cannot interpolate an invalid quaternion")

        q0 = [value / first_norm for value in first]
        q1 = [value / second_norm for value in second]
        dot = sum(q0[i] * q1[i] for i in range(4))
        if dot < 0.0:
            q1 = [-value for value in q1]
            dot = -dot
        dot = max(-1.0, min(1.0, dot))

        if dot > 0.9995:
            result = [
                q0[i] + alpha * (q1[i] - q0[i])
                for i in range(4)
            ]
        else:
            theta = math.acos(dot)
            sin_theta = math.sin(theta)
            first_scale = math.sin((1.0 - alpha) * theta) / sin_theta
            second_scale = math.sin(alpha * theta) / sin_theta
            result = [
                first_scale * q0[i] + second_scale * q1[i]
                for i in range(4)
            ]

        norm = math.sqrt(sum(value * value for value in result))
        return [value / norm for value in result]

    @staticmethod
    def _build_debug_result(
        task: str,
        frame: str,
        detections: dict[str, Any],
        matched_position_ned: list[float] | None,
        matched_attitude_q: list[float] | None,
        pose_sync: dict[str, Any],
        transformed_objects: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "task": task,
            "frame": frame,
            "sequence": detections.get("sequence"),
            "rgb_stamp": detections.get("rgb_stamp"),
            "rgb_image_size": detections.get("rgb_image_size"),
            "depth": detections.get("depth"),
            "pose_sync": pose_sync,
            "raw_detections": detections.get("detections", []),
            "vehicle_position_ned": matched_position_ned,
            "vehicle_attitude_q": matched_attitude_q,
            "transformed_objects": transformed_objects,
        }

    def _publish_pose_sync_warning(
        self,
        pose_sync: dict[str, Any],
    ) -> None:
        now = time.monotonic()
        if now - self.last_pose_sync_warning_time < 1.0:
            return
        self.last_pose_sync_warning_time = now
        self.get_logger().warning(
            "No PX4 pose matched the RGB timestamp: "
            f"{pose_sync}"
        )

    @staticmethod
    def _publish_json(publisher: Any, payload: dict[str, Any]) -> None:
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False)
        publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DownwardVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
