"""Publish sampled PX4 NED/FRD odometry and TF for perception geometry."""

from __future__ import annotations

from pathlib import Path

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition
from sensor_msgs.msg import CameraInfo
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
import yaml

from uav_common.px4_pose_bridge import DEFAULTS, Px4PoseBuffer, camera_transform


WORLD_FRAME = "map_ned"
BODY_FRAME = "base_link_frd"


def _stamp(message_stamp, seconds):
    total = round(seconds * 1_000_000_000)
    message_stamp.sec = total // 1_000_000_000
    message_stamp.nanosec = total % 1_000_000_000


def _point(destination, values):
    destination.x, destination.y, destination.z = values


def _quaternion(destination, values):
    destination.x, destination.y, destination.z, destination.w = values


class Px4PerceptionPoseNode(Node):
    """Supply the same sampled pose to beacon odometry and dock TF consumers."""

    def __init__(self):
        super().__init__("px4_perception_pose")
        self.declare_parameter("config_file", "")
        self.declare_parameter("downward_config_file", "")
        filename = str(self.get_parameter("config_file").value)
        config = (yaml.safe_load(Path(filename).expanduser().read_text(encoding="utf-8"))["px4_pose"]
                  if filename else {})
        parameters = {}
        for name, default in DEFAULTS.items():
            self.declare_parameter(name, config.get(name, default))
            parameters[name] = self.get_parameter(name).value
        self.buffer = Px4PoseBuffer(parameters)
        self._last_reset_count = 0
        topics = config.get("topics", {})
        self.declare_parameter("downward_camera_frame_id", config.get(
            "downward_camera_frame_id", ""))
        self.camera_frame = str(self.get_parameter("downward_camera_frame_id").value).strip()
        self.declare_parameter("camera_info_topic", topics.get("camera_info", "/oak/rgb/camera_info"))
        px4_qos = QoSProfile(depth=64, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE,
                             history=HistoryPolicy.KEEP_LAST)
        pose_qos = QoSProfile(depth=64, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.VOLATILE)
        self.odom_pub = self.create_publisher(
            Odometry, topics.get("odometry", "/tongji_mrl/tsky/perception/odometry"), pose_qos)
        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        downward_file = str(self.get_parameter("downward_config_file").value)
        downward = yaml.safe_load(Path(downward_file).expanduser().read_text(encoding="utf-8"))["downward_vision"]
        self.camera_translation, self.camera_rotation = camera_transform(downward["transforms"]["T_body_camera"])
        if self.camera_frame:
            self._publish_camera_transform()
        self.camera_info_sub = self.create_subscription(
            CameraInfo, str(self.get_parameter("camera_info_topic").value), self._camera_info, px4_qos)
        self.position_sub = self.create_subscription(
            VehicleLocalPosition,
            topics.get("vehicle_local_position", "/fmu/out/vehicle_local_position_v1"),
            self._position, px4_qos)
        self.attitude_sub = self.create_subscription(
            VehicleAttitude, topics.get("vehicle_attitude", "/fmu/out/vehicle_attitude"),
            self._attitude, px4_qos)

    def _camera_info(self, message):
        if not self.camera_frame and message.header.frame_id:
            self.camera_frame = message.header.frame_id
            self._publish_camera_transform()

    def _publish_camera_transform(self):
        if self.camera_frame in {WORLD_FRAME, BODY_FRAME}:
            raise ValueError("camera optical frame must be distinct from world/body frames")
        transform = TransformStamped()
        # Static calibration is time-invariant. No map-to-camera shortcut is created.
        transform.header.frame_id = BODY_FRAME
        transform.child_frame_id = self.camera_frame
        _point(transform.transform.translation, self.camera_translation)
        _quaternion(transform.transform.rotation, self.camera_rotation)
        self.static_tf.sendTransform(transform)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _position(self, message):
        now = self._now()
        if self.buffer.add_position(message, now):
            self._publish_ready(now)
        self._log_reset()

    def _attitude(self, message):
        now = self._now()
        if self.buffer.add_attitude(message, now):
            self._publish_ready(now)
        self._log_reset()

    def _publish_ready(self, now):
        for sample in self.buffer.take_ready(now):
            message = Odometry()
            _stamp(message.header.stamp, sample.stamp_seconds)
            message.header.frame_id = WORLD_FRAME
            message.child_frame_id = BODY_FRAME
            _point(message.pose.pose.position, sample.position_ned)
            _quaternion(message.pose.pose.orientation, sample.quaternion_xyzw)
            # Pose is assembled from sample timestamps, not callback receipt time.
            self.odom_pub.publish(message)
            transform = TransformStamped()
            _stamp(transform.header.stamp, sample.stamp_seconds)
            transform.header.frame_id = WORLD_FRAME
            transform.child_frame_id = BODY_FRAME
            _point(transform.transform.translation, sample.position_ned)
            _quaternion(transform.transform.rotation, sample.quaternion_xyzw)
            self.tf.sendTransform(transform)

    def _log_reset(self):
        if self.buffer.reset_count != self._last_reset_count:
            self._last_reset_count = self.buffer.reset_count
            self.get_logger().warning("PX4 estimator reset: cleared perception pose buffer")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Px4PerceptionPoseNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
