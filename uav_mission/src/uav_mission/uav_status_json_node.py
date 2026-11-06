#!/usr/bin/env python3
"""Publish the RobotX UAV status as JSON over ROS 2."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import rclpy
import yaml
from px4_msgs.msg import (
    BatteryStatus,
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


class UavStatusJsonNode(Node):
    """Collect PX4 state and publish one JSON status object."""

    STATE_UNKNOWN = "STATE_UNKNOWN"
    STATE_KILLED = "STATE_KILLED"
    STATE_MANUAL = "STATE_MANUAL"
    STATE_AUTO = "STATE_AUTO"

    FLIGHT_PHASE_UNKNOWN = "FLIGHT_PHASE_UNKNOWN"
    FLIGHT_PHASE_GROUNDED = "FLIGHT_PHASE_GROUNDED"
    FLIGHT_PHASE_AIRBORNE = "FLIGHT_PHASE_AIRBORNE"

    # JSON-stage representation of the protobuf VehicleType UAV member.
    VEHICLE_TYPE = "TYPE_UAV"

    AUTO_NAV_STATE_NAMES = (
        "NAVIGATION_STATE_DESCEND",
        "NAVIGATION_STATE_OFFBOARD",
        "NAVIGATION_STATE_AUTO_MISSION",
        "NAVIGATION_STATE_AUTO_LOITER",
        "NAVIGATION_STATE_AUTO_RTL",
        "NAVIGATION_STATE_AUTO_TAKEOFF",
        "NAVIGATION_STATE_AUTO_LAND",
        "NAVIGATION_STATE_AUTO_FOLLOW_TARGET",
        "NAVIGATION_STATE_AUTO_PRECLAND",
        "NAVIGATION_STATE_AUTO_VTOL_TAKEOFF",
        "NAVIGATION_STATE_ORBIT",
    )

    BATTERY_WARNING_NAMES = (
        "WARNING_NONE",
        "WARNING_LOW",
        "WARNING_CRITICAL",
        "WARNING_EMERGENCY",
        "WARNING_FAILED",
        "STATE_UNHEALTHY",
        "STATE_CHARGING",
    )

    def __init__(self) -> None:
        super().__init__("uav_status_json")

        self.declare_parameter("config_file", "")
        config = self._load_config()
        identity = config.get("identity", {})
        runtime = config.get("runtime", {})
        topics = config.get("topics", {})

        self.vehicle_id = str(identity.get("vehicle_id", "")).strip()
        if not self.vehicle_id:
            raise ValueError("identity.vehicle_id must not be empty")

        self.publish_rate_hz = float(runtime.get("publish_rate_hz", 5.0))
        self.telemetry_timeout_s = float(
            runtime.get("telemetry_timeout_s", 1.0)
        )
        if self.publish_rate_hz <= 0.0:
            raise ValueError("runtime.publish_rate_hz must be positive")
        if self.telemetry_timeout_s <= 0.0:
            raise ValueError("runtime.telemetry_timeout_s must be positive")

        self.robot_state = self.STATE_UNKNOWN
        self.flight_phase = self.FLIGHT_PHASE_UNKNOWN
        self.latitude: float | None = None
        self.longitude: float | None = None
        self.altitude_hae_m: float | None = None
        self.speed_mps: float | None = None
        self.heading_deg: float | None = None
        self.battery_status: dict[str, Any] | None = None

        self.last_vehicle_status_time: float | None = None
        self.last_global_position_time: float | None = None
        self.last_local_position_time: float | None = None
        self.last_attitude_time: float | None = None
        self.last_land_detected_time: float | None = None
        self.last_battery_time: float | None = None
        self.last_warning_time = 0.0

        self.auto_nav_states = {
            int(getattr(VehicleStatus, name))
            for name in self.AUTO_NAV_STATE_NAMES
            if hasattr(VehicleStatus, name)
        }
        self.termination_nav_state = int(
            getattr(VehicleStatus, "NAVIGATION_STATE_TERMINATION", -1)
        )
        self.kill_disarm_reasons = {
            int(getattr(VehicleStatus, name))
            for name in (
                "ARM_DISARM_REASON_KILL_SWITCH",
                "ARM_DISARM_REASON_LOCKDOWN",
            )
            if hasattr(VehicleStatus, name)
        }
        self.battery_warning_names = {
            int(getattr(BatteryStatus, name)): name
            for name in self.BATTERY_WARNING_NAMES
            if hasattr(BatteryStatus, name)
        }

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus,
            self._topic(topics, "vehicle_status"),
            self._on_vehicle_status,
            px4_qos,
        )
        self.global_position_sub = self.create_subscription(
            VehicleGlobalPosition,
            self._topic(topics, "vehicle_global_position"),
            self._on_global_position,
            px4_qos,
        )
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            self._topic(topics, "vehicle_local_position"),
            self._on_local_position,
            px4_qos,
        )
        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            self._topic(topics, "vehicle_attitude"),
            self._on_attitude,
            px4_qos,
        )
        self.land_detected_sub = self.create_subscription(
            VehicleLandDetected,
            self._topic(topics, "vehicle_land_detected"),
            self._on_land_detected,
            px4_qos,
        )
        self.battery_sub = self.create_subscription(
            BatteryStatus,
            self._topic(topics, "battery_status"),
            self._on_battery_status,
            px4_qos,
        )

        output_topic = self._topic(topics, "output")
        self.status_pub = self.create_publisher(String, output_topic, 10)
        self.status_timer = self.create_timer(
            1.0 / self.publish_rate_hz,
            self._publish_status,
        )

        self.get_logger().info(
            "UAV JSON status publisher started: "
            f"vehicle_id={self.vehicle_id}, topic={output_topic}, "
            f"rate={self.publish_rate_hz:g} Hz"
        )

    def _load_config(self) -> dict[str, Any]:
        config_file = str(self.get_parameter("config_file").value).strip()
        if not config_file:
            raise RuntimeError("config_file parameter is required")

        config_path = Path(config_file).expanduser()
        if not config_path.is_file():
            raise RuntimeError(f"Config file does not exist: {config_path}")

        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        config = loaded.get("uav_status", loaded)
        if not isinstance(config, dict):
            raise ValueError("uav_status config must be a mapping")
        return config

    @staticmethod
    def _topic(topics: dict[str, Any], name: str) -> str:
        value = str(topics.get(name, "")).strip()
        if not value:
            raise ValueError(f"topics.{name} must not be empty")
        return value

    @staticmethod
    def _finite_or_none(value: Any, minimum: float | None = None) -> float | None:
        result = float(value)
        if not math.isfinite(result):
            return None
        if minimum is not None and result < minimum:
            return None
        return result

    def _on_vehicle_status(self, msg: VehicleStatus) -> None:
        nav_state = int(msg.nav_state)
        armed = (
            int(msg.arming_state)
            == int(VehicleStatus.ARMING_STATE_ARMED)
        )
        killed = nav_state == self.termination_nav_state or (
            not armed
            and int(msg.latest_disarming_reason) in self.kill_disarm_reasons
        )

        if killed:
            self.robot_state = self.STATE_KILLED
        elif nav_state in self.auto_nav_states:
            self.robot_state = self.STATE_AUTO
        else:
            self.robot_state = self.STATE_MANUAL
        self.last_vehicle_status_time = time.monotonic()

    def _on_global_position(self, msg: VehicleGlobalPosition) -> None:
        if not bool(msg.lat_lon_valid) or not bool(msg.alt_valid):
            self.latitude = None
            self.longitude = None
            self.altitude_hae_m = None
            return

        values = (
            float(msg.lat),
            float(msg.lon),
            float(msg.alt_ellipsoid),
        )
        if not all(math.isfinite(value) for value in values):
            return

        self.latitude, self.longitude, self.altitude_hae_m = values
        self.last_global_position_time = time.monotonic()

    def _on_local_position(self, msg: VehicleLocalPosition) -> None:
        if not bool(msg.v_xy_valid):
            self.speed_mps = None
            return

        vx = float(msg.vx)
        vy = float(msg.vy)
        if not math.isfinite(vx) or not math.isfinite(vy):
            self.speed_mps = None
            return

        self.speed_mps = math.hypot(vx, vy)
        self.last_local_position_time = time.monotonic()

    def _on_attitude(self, msg: VehicleAttitude) -> None:
        w, x, y, z = (float(value) for value in msg.q)
        if not all(math.isfinite(value) for value in (w, x, y, z)):
            return

        yaw_rad = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        self.heading_deg = (math.degrees(yaw_rad) + 360.0) % 360.0
        self.last_attitude_time = time.monotonic()

    def _on_land_detected(self, msg: VehicleLandDetected) -> None:
        self.flight_phase = (
            self.FLIGHT_PHASE_GROUNDED
            if bool(msg.landed)
            else self.FLIGHT_PHASE_AIRBORNE
        )
        self.last_land_detected_time = time.monotonic()

    def _on_battery_status(self, msg: BatteryStatus) -> None:
        remaining = self._finite_or_none(msg.remaining, 0.0)
        if remaining is not None:
            remaining = min(remaining, 1.0)

        self.battery_status = {
            "connected": bool(msg.connected),
            "remaining": remaining,
            "remaining_percent": (
                round(remaining * 100.0, 1)
                if remaining is not None
                else None
            ),
            "voltage_v": self._finite_or_none(msg.voltage_v, 0.0),
            "current_a": self._finite_or_none(msg.current_a, 0.0),
            "time_remaining_s": self._finite_or_none(
                msg.time_remaining_s,
                0.0,
            ),
            "warning": self.battery_warning_names.get(
                int(msg.warning),
                f"UNKNOWN_{int(msg.warning)}",
            ),
        }
        self.last_battery_time = time.monotonic()

    def _telemetry_is_ready(self) -> bool:
        return (
            self.robot_state != self.STATE_UNKNOWN
            and self.flight_phase != self.FLIGHT_PHASE_UNKNOWN
            and all(
                value is not None
                for value in (
                    self.latitude,
                    self.longitude,
                    self.altitude_hae_m,
                    self.speed_mps,
                    self.heading_deg,
                    self.battery_status,
                )
            )
        )

    def _telemetry_is_fresh(self, now: float) -> bool:
        timestamps = (
            self.last_vehicle_status_time,
            self.last_global_position_time,
            self.last_local_position_time,
            self.last_attitude_time,
            self.last_land_detected_time,
            self.last_battery_time,
        )
        return all(
            timestamp is not None
            and now - timestamp <= self.telemetry_timeout_s
            for timestamp in timestamps
        )

    def _publish_status(self) -> None:
        now = time.monotonic()
        if not self._telemetry_is_ready() or not self._telemetry_is_fresh(now):
            if now - self.last_warning_time >= 5.0:
                self.last_warning_time = now
                self.get_logger().warning(
                    "UAV status not published: PX4 telemetry is "
                    "unavailable, invalid, or stale"
                )
            return

        payload: dict[str, Any] = {
            "vehicle_id": self.vehicle_id,
            "vehicle_type": self.VEHICLE_TYPE,
            "state": self.robot_state,
            "flight_phase": self.flight_phase,
            "position": {
                "latitude": self.latitude,
                "longitude": self.longitude,
            },
            "altitude_hae_m": self.altitude_hae_m,
            "spd_mps": self.speed_mps,
            "heading_deg": self.heading_deg,
            "battery_status": self.battery_status,
        }

        message = String()
        message.data = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.status_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UavStatusJsonNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
