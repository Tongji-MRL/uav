#!/usr/bin/env python3
"""Mode-controlled RGB-D vision node for MAP and TASK3."""

from __future__ import annotations

import json
from pathlib import Path
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

    VALID_MODES = {"OFF", "MAP", "TASK1", "TASK3"}
    MODE_OBJECT_TYPES = {
        "OFF": frozenset(),
        "MAP": frozenset({"platform"}),
        "TASK1": frozenset(),
        "TASK3": frozenset({"platform", "tin", "circle"}),
    }

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

        self.depth_unit = str(perception.get("depth_unit", "mm"))
        self.invalid_depth_values = list(
            perception.get("invalid_depth_values", [0])
        )

        # Both models are loaded once. Mode selection decides which one runs.
        self.map_detector = MultiRgbdTargetDetector(
            perception,
            [
                item for item in models.get("map", [])
                if str(item.get("object_type", "")).strip().lower()
                in self.MODE_OBJECT_TYPES["MAP"]
            ],
        )
        self.task3_detector = MultiRgbdTargetDetector(
            perception,
            [
                item for item in models.get("task3", [])
                if str(item.get("object_type", "")).strip().lower()
                in self.MODE_OBJECT_TYPES["TASK3"]
            ],
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

        control_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
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
            control_qos,
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
        mode = command["mode"]
        requested_types = set(command["object_types"])

        allowed = self.MODE_OBJECT_TYPES[mode]
        active_types = set(allowed) if "all" in requested_types else allowed & requested_types
        if mode != self.mode or active_types != self.active_object_types:
            self.mode = mode
            self.active_object_types = set(active_types)
            self.last_processed_stamp = None
            self.get_logger().info(
                f"Vision mode={mode}, objects={sorted(active_types)}"
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
        self.position_ned = [float(msg.x), float(msg.y), float(msg.z)]

    def _on_attitude(self, msg: VehicleAttitude) -> None:
        self.attitude_q = [float(value) for value in msg.q]

    def _process_latest_frame(self) -> None:
        # TASK1 uses the separate beacon detector, so down-looking inference stops.
        if self.mode not in {"MAP", "TASK3"} or self.latest_rgb is None:
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
                rgb_stamp_sec=image_stamp_sec(self.latest_rgb),
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
            self._publish_map_result(detections)

        elif self.mode == "TASK3":
            # 转换为gripper坐标并发布
            self._publish_task3_result(detections)

    def _publish_map_result(
        self,
        detections: dict[str, Any],
    ) -> None:
        if (
            self.position_ned is None
            or self.attitude_q is None
        ):
            return

        transformed = self.map_to_ned.transform(
            detections,
            self.position_ned,
            self.attitude_q,
        )

        # 保持原来的MAP输出格式，避免影响主节点
        map_result = {
            "task": "MAP",
            "frame": "local_ned",
            "stamp": detections.get("rgb_stamp"),
            "rgb_stamp": detections.get("rgb_stamp"),
            "sequence": detections.get("sequence"),
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
        debug_result = {
            "task": "MAP_DEBUG",
            "sequence": detections.get("sequence"),
            "rgb_stamp": detections.get("rgb_stamp"),
            "rgb_image_size": detections.get(
                "rgb_image_size"
            ),
            "depth": detections.get("depth"),

            # 原始YOLO和深度结果
            "raw_detections": detections.get(
                "detections",
                [],
            ),

            # 本次转换使用的无人机位姿
            "vehicle_position_ned": list(
                self.position_ned
            ),
            "vehicle_attitude_q": list(
                self.attitude_q
            ),

            # 各坐标转换阶段
            "transformed_objects": transformed.get(
                "objects",
                [],
            ),
        }

        self._publish_json(
            self.map_debug_pub,
            debug_result,
        )

    def _publish_task3_result(
        self,
        detections: dict[str, Any],
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
        debug_result = {
            "task": "TASK3_DEBUG",
            "frame": "gripper",

            "sequence": detections.get("sequence"),
            "rgb_stamp": detections.get("rgb_stamp"),
            "rgb_image_size": detections.get(
                "rgb_image_size"
            ),
            "depth": detections.get("depth"),

            # YOLO原始结果及深度
            "raw_detections": detections.get(
                "detections",
                [],
            ),

            # 当前PX4位姿，仅用于排查时间同步
            "vehicle_position_ned": (
                list(self.position_ned)
                if self.position_ned is not None
                else None
            ),
            "vehicle_attitude_q": (
                list(self.attitude_q)
                if self.attitude_q is not None
                else None
            ),

            # 相机坐标转夹爪坐标的结果
            "transformed_objects": transformed.get(
                "objects",
                [],
            ),
        }

        self._publish_json(
            self.task3_debug_pub,
            debug_result,
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
