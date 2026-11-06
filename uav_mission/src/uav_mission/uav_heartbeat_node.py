#!/usr/bin/env python3
"""Publish RobotX UAV heartbeat telemetry as JSON over ROS 2."""

from __future__ import annotations

import json
import math
import time
from typing import Any

import rclpy
from px4_msgs.msg import (
    VehicleAttitude,
    VehicleGlobalPosition,
    VehicleLandDetected,
    VehicleLocalPosition,
    VehicleStatus,
)
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String


class UavHeartbeatNode(Node):
    """Collect PX4 telemetry and publish a RobotX heartbeat at 5 Hz."""

    STATE_UNKNOWN = "STATE_UNKNOWN"
    STATE_KILLED = "STATE_KILLED"
    STATE_MANUAL = "STATE_MANUAL"
    STATE_AUTO = "STATE_AUTO"

    TASK_UNKNOWN = "TASK_UNKNOWN"
    TASK_NONE = "TASK_NONE"
    TASK_SAFE_PASSAGE = "TASK_SAFE_PASSAGE"
    TASK_COORDINATED_LOGISTICS = "TASK_COORDINATED_LOGISTICS"
    TASK_DYNAMIC_INCIDENT = "TASK_DYNAMIC_INCIDENT"

    TYPE_UAV = "TYPE_UAV"

    FLIGHT_PHASE_UNKNOWN = "FLIGHT_PHASE_UNKNOWN"
    FLIGHT_PHASE_GROUNDED = "FLIGHT_PHASE_GROUNDED"
    FLIGHT_PHASE_AIRBORNE = "FLIGHT_PHASE_AIRBORNE"

    MISSION_TO_RX_TASK = {
        "MAP": TASK_SAFE_PASSAGE,
        "TASK1": TASK_SAFE_PASSAGE,
        "TASK3": TASK_COORDINATED_LOGISTICS,
        "TASK4": TASK_DYNAMIC_INCIDENT,
        "FINISHED": TASK_NONE,
        "ERROR": TASK_NONE,
    }

    AUTO_NAV_STATE_NAMES = (
        "NAVIGATION_STATE_OFFBOARD",
        "NAVIGATION_STATE_AUTO_MISSION",
        "NAVIGATION_STATE_AUTO_LOITER",
        "NAVIGATION_STATE_AUTO_RTL",
        "NAVIGATION_STATE_AUTO_TAKEOFF",
        "NAVIGATION_STATE_AUTO_LAND",
        "NAVIGATION_STATE_AUTO_PRECLAND",
        "NAVIGATION_STATE_ORBIT",
    )

    def __init__(self) -> None:
        super().__init__("uav_heartbeat")

        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("telemetry_timeout_s", 1.0)
        self.declare_parameter(
            "heartbeat_topic",
            "/tongji_mrl/tsky/heartbeat",
        )
        self.declare_parameter(
            "progress_topic",
            "/tongji_mrl/tsky/progress",
        )
        self.declare_parameter(
            "vehicle_status_topic",
            "/fmu/out/vehicle_status_v1",
        )
        self.declare_parameter(
            "vehicle_global_position_topic",
            "/fmu/out/vehicle_global_position",
        )
        self.declare_parameter(
            "vehicle_local_position_topic",
            "/fmu/out/vehicle_local_position_v1",
        )
        self.declare_parameter(
            "vehicle_attitude_topic",
            "/fmu/out/vehicle_attitude",
        )
        self.declare_parameter(
            "vehicle_land_detected_topic",
            "/fmu/out/vehicle_land_detected",
        )

        self.publish_rate_hz = float(
            self.get_parameter("publish_rate_hz").value
        )
        self.telemetry_timeout_s = float(
            self.get_parameter("telemetry_timeout_s").value
        )
        if self.publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be positive")
        if self.telemetry_timeout_s <= 0.0:
            raise ValueError("telemetry_timeout_s must be positive")

        self.robot_state = self.STATE_UNKNOWN
        self.current_task = self.TASK_UNKNOWN
        self.flight_phase = self.FLIGHT_PHASE_UNKNOWN

        self.latitude: float | None = None
        self.longitude: float | None = None
        self.altitude_hae_m: float | None = None
        self.spd_mps: float | None = None
        self.heading_deg: float | None = None
        self.roll_deg: float | None = None
        self.pitch_deg: float | None = None

        self.last_global_position_time: float | None = None
        self.last_local_position_time: float | None = None
        self.last_attitude_time: float | None = None
        self.last_warning_time = 0.0

        self.auto_nav_states = {
            int(getattr(VehicleStatus, name))
            for name in self.AUTO_NAV_STATE_NAMES
            if hasattr(VehicleStatus, name)
        }

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus,
            str(self.get_parameter("vehicle_status_topic").value),
            self._on_vehicle_status,
            px4_qos,
        )
        self.global_position_sub = self.create_subscription(
            VehicleGlobalPosition,
            str(
                self.get_parameter(
                    "vehicle_global_position_topic"
                ).value
            ),
            self._on_global_position,
            px4_qos,
        )
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            str(
                self.get_parameter(
                    "vehicle_local_position_topic"
                ).value
            ),
            self._on_local_position,
            px4_qos,
        )
        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            str(self.get_parameter("vehicle_attitude_topic").value),
            self._on_attitude,
            px4_qos,
        )
        self.land_detected_sub = self.create_subscription(
            VehicleLandDetected,
            str(
                self.get_parameter(
                    "vehicle_land_detected_topic"
                ).value
            ),
            self._on_land_detected,
            px4_qos,
        )
        self.progress_sub = self.create_subscription(
            String,
            str(self.get_parameter("progress_topic").value),
            self._on_progress,
            10,
        )

        heartbeat_topic = str(
            self.get_parameter("heartbeat_topic").value
        )
        self.heartbeat_pub = self.create_publisher(
            String,
            heartbeat_topic,
            10,
        )
        self.heartbeat_timer = self.create_timer(
            1.0 / self.publish_rate_hz,
            self._publish_heartbeat,
        )

        self.get_logger().info(
            f"UAV heartbeat publisher started: topic={heartbeat_topic}, "
            f"rate={self.publish_rate_hz:g} Hz"
        )

    def _on_vehicle_status(self, msg: VehicleStatus) -> None:
        armed = (
            int(msg.arming_state)
            == int(VehicleStatus.ARMING_STATE_ARMED)
        )
        if not armed:
            self.robot_state = self.STATE_KILLED
        elif int(msg.nav_state) in self.auto_nav_states:
            self.robot_state = self.STATE_AUTO
        else:
            self.robot_state = self.STATE_MANUAL

    def _on_global_position(
        self,
        msg: VehicleGlobalPosition,
    ) -> None:
        if not bool(msg.lat_lon_valid) or not bool(msg.alt_valid):
            self.latitude = None
            self.longitude = None
            self.altitude_hae_m = None
            return

        latitude = float(msg.lat)
        longitude = float(msg.lon)
        altitude_hae_m = float(msg.alt_ellipsoid)
        if not all(
            math.isfinite(value)
            for value in (latitude, longitude, altitude_hae_m)
        ):
            return

        self.latitude = latitude
        self.longitude = longitude
        self.altitude_hae_m = altitude_hae_m
        self.last_global_position_time = time.monotonic()

    def _on_local_position(
        self,
        msg: VehicleLocalPosition,
    ) -> None:
        if not bool(msg.v_xy_valid):
            self.spd_mps = None
            return

        vx = float(msg.vx)
        vy = float(msg.vy)
        if not math.isfinite(vx) or not math.isfinite(vy):
            self.spd_mps = None
            return

        self.spd_mps = math.hypot(vx, vy)
        self.last_local_position_time = time.monotonic()

    def _on_attitude(self, msg: VehicleAttitude) -> None:
        w, x, y, z = (float(value) for value in msg.q)
        if not all(
            math.isfinite(value) for value in (w, x, y, z)
        ):
            return

        roll_rad = math.atan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x * x + y * y),
        )
        pitch_input = max(
            -1.0,
            min(1.0, 2.0 * (w * y - z * x)),
        )
        pitch_rad = math.asin(pitch_input)
        yaw_rad = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

        self.roll_deg = math.degrees(roll_rad)
        self.pitch_deg = math.degrees(pitch_rad)
        self.heading_deg = (math.degrees(yaw_rad) + 360.0) % 360.0
        self.last_attitude_time = time.monotonic()

    def _on_land_detected(
        self,
        msg: VehicleLandDetected,
    ) -> None:
        self.flight_phase = (
            self.FLIGHT_PHASE_GROUNDED
            if bool(msg.landed)
            else self.FLIGHT_PHASE_AIRBORNE
        )

    def _on_progress(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError) as exc:
            self.get_logger().warning(
                f"Ignored invalid progress JSON: {exc}"
            )
            return

        if not isinstance(payload, dict):
            self.current_task = self.TASK_UNKNOWN
            return

        mission = str(payload.get("task", "")).strip().upper()
        self.current_task = self.MISSION_TO_RX_TASK.get(
            mission,
            self.TASK_UNKNOWN,
        )

    def _telemetry_is_ready(self) -> bool:
        return all(
            value is not None
            for value in (
                self.latitude,
                self.longitude,
                self.altitude_hae_m,
                self.spd_mps,
                self.heading_deg,
                self.roll_deg,
                self.pitch_deg,
            )
        )

    def _telemetry_is_fresh(self, now: float) -> bool:
        timestamps = (
            self.last_global_position_time,
            self.last_local_position_time,
            self.last_attitude_time,
        )
        return all(
            timestamp is not None
            and now - timestamp <= self.telemetry_timeout_s
            for timestamp in timestamps
        )

    def _publish_heartbeat(self) -> None:
        now = time.monotonic()
        if not self._telemetry_is_ready() or not self._telemetry_is_fresh(now):
            if now - self.last_warning_time >= 5.0:
                self.last_warning_time = now
                self.get_logger().warning(
                    "Heartbeat not published: PX4 telemetry is "
                    "unavailable, invalid, or stale"
                )
            return

        payload: dict[str, Any] = {
            "state": self.robot_state,
            "position": {
                "latitude": self.latitude,
                "longitude": self.longitude,
            },
            "spd_mps": self.spd_mps,
            "heading_deg": self.heading_deg,
            "roll_deg": self.roll_deg,
            "pitch_deg": self.pitch_deg,
            "altitude_hae_m": self.altitude_hae_m,
            "current_task": self.current_task,
            "vehicle_type": self.TYPE_UAV,
            "flight_phase": self.flight_phase,
        }

        message = String()
        message.data = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.heartbeat_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UavHeartbeatNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
