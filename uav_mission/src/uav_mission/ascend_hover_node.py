#!/usr/bin/env python3
"""Ascend to a relative altitude, hover, and yield to Position mode."""

from __future__ import annotations

import math
import time
from enum import Enum, auto

import rclpy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleAttitude,
    VehicleCommand,
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


class FlightState(Enum):
    """States of the autonomous ascend-and-hover mission."""

    WAIT_FOR_LOCAL_POSITION = auto()
    WAIT_FOR_ARM = auto()
    PREPARE_OFFBOARD = auto()
    ASCEND = auto()
    HOVER = auto()
    MANUAL_CONTROL = auto()
    ERROR = auto()


class AscendHoverNode(Node):
    """Fly vertically to a relative altitude and hold until pilot takeover."""

    def __init__(self) -> None:
        super().__init__("uav_ascend_hover")

        # Mission parameters.
        self.declare_parameter("target_altitude_m", 10.0)
        self.declare_parameter("altitude_tolerance_m", 0.20)
        self.declare_parameter("arrival_stable_s", 1.0)
        self.declare_parameter("update_rate_hz", 20.0)
        self.declare_parameter("local_position_timeout_s", 1.0)
        self.declare_parameter("prestream_duration_s", 1.0)
        self.declare_parameter("offboard_request_interval_s", 0.5)

        # Topic defaults match the existing UAV mission package.
        self.declare_parameter(
            "topics.vehicle_status", "/fmu/out/vehicle_status_v1"
        )
        self.declare_parameter(
            "topics.vehicle_local_position",
            "/fmu/out/vehicle_local_position_v1",
        )
        self.declare_parameter(
            "topics.vehicle_attitude", "/fmu/out/vehicle_attitude"
        )
        self.declare_parameter(
            "topics.offboard_control_mode",
            "/fmu/in/offboard_control_mode",
        )
        self.declare_parameter(
            "topics.trajectory_setpoint", "/fmu/in/trajectory_setpoint"
        )
        self.declare_parameter(
            "topics.vehicle_command", "/fmu/in/vehicle_command"
        )

        self.target_altitude_m = float(
            self.get_parameter("target_altitude_m").value
        )
        self.altitude_tolerance_m = float(
            self.get_parameter("altitude_tolerance_m").value
        )
        self.arrival_stable_s = float(
            self.get_parameter("arrival_stable_s").value
        )
        self.update_rate_hz = float(
            self.get_parameter("update_rate_hz").value
        )
        self.local_position_timeout_s = float(
            self.get_parameter("local_position_timeout_s").value
        )
        self.prestream_duration_s = float(
            self.get_parameter("prestream_duration_s").value
        )
        self.offboard_request_interval_s = float(
            self.get_parameter("offboard_request_interval_s").value
        )
        self._validate_parameters()

        self.state = FlightState.WAIT_FOR_LOCAL_POSITION
        self.local_position: list[float] | None = None
        self.last_local_position_time: float | None = None
        self.current_yaw: float | None = None
        self.home_ned: list[float] | None = None
        self.target_ned: list[float] | None = None
        self.hold_yaw: float | None = None

        self.vehicle_armed = False
        self.vehicle_in_offboard = False
        self.vehicle_in_position = False
        self.offboard_was_active = False

        self.prestream_start_time: float | None = None
        self.last_offboard_request_time: float | None = None
        self.arrival_start_time: float | None = None

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus,
            str(self.get_parameter("topics.vehicle_status").value),
            self._on_vehicle_status,
            px4_qos,
        )
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            str(self.get_parameter("topics.vehicle_local_position").value),
            self._on_local_position,
            px4_qos,
        )
        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            str(self.get_parameter("topics.vehicle_attitude").value),
            self._on_vehicle_attitude,
            px4_qos,
        )

        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode,
            str(self.get_parameter("topics.offboard_control_mode").value),
            px4_qos,
        )
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            str(self.get_parameter("topics.trajectory_setpoint").value),
            px4_qos,
        )
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand,
            str(self.get_parameter("topics.vehicle_command").value),
            px4_qos,
        )

        self.update_timer = self.create_timer(
            1.0 / self.update_rate_hz,
            self._tick,
        )

        self.get_logger().info(
            "Ascend-hover node started; waiting for local position and attitude"
        )

    def _validate_parameters(self) -> None:
        positive_values = {
            "target_altitude_m": self.target_altitude_m,
            "altitude_tolerance_m": self.altitude_tolerance_m,
            "update_rate_hz": self.update_rate_hz,
            "local_position_timeout_s": self.local_position_timeout_s,
            "offboard_request_interval_s": self.offboard_request_interval_s,
        }
        for name, value in positive_values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")

        nonnegative_values = {
            "arrival_stable_s": self.arrival_stable_s,
            "prestream_duration_s": self.prestream_duration_s,
        }
        for name, value in nonnegative_values.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a non-negative finite number")

    # ------------------------------------------------------------------
    # Subscriber callbacks only cache the newest PX4 state.
    # ------------------------------------------------------------------
    def _on_vehicle_status(self, msg: VehicleStatus) -> None:
        self.vehicle_armed = (
            msg.arming_state == VehicleStatus.ARMING_STATE_ARMED
        )
        self.vehicle_in_offboard = (
            msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        )
        self.vehicle_in_position = (
            msg.nav_state == VehicleStatus.NAVIGATION_STATE_POSCTL
        )

        if self.vehicle_in_offboard:
            self.offboard_was_active = True

    def _on_local_position(self, msg: VehicleLocalPosition) -> None:
        position = [float(msg.x), float(msg.y), float(msg.z)]
        if not all(math.isfinite(value) for value in position):
            return

        self.local_position = position
        self.last_local_position_time = time.monotonic()

    def _on_vehicle_attitude(self, msg: VehicleAttitude) -> None:
        quaternion = [float(value) for value in msg.q]
        if len(quaternion) != 4 or not all(
            math.isfinite(value) for value in quaternion
        ):
            return

        w, x, y, z = quaternion
        self.current_yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    # ------------------------------------------------------------------
    # State machine.
    # ------------------------------------------------------------------
    def _tick(self) -> None:
        # Position before the first Offboard entry is a normal startup mode.
        # Position after Offboard has been active means the pilot took over.
        if (
            self.offboard_was_active
            and self.vehicle_in_position
            and self.state
            not in {FlightState.MANUAL_CONTROL, FlightState.ERROR}
        ):
            self._change_state(
                FlightState.MANUAL_CONTROL,
                "Position mode detected; stopped Offboard output, pilot has control",
            )

        if self.state == FlightState.WAIT_FOR_LOCAL_POSITION:
            self._wait_for_local_position()
        elif self.state == FlightState.WAIT_FOR_ARM:
            self._wait_for_arm()
        elif self.state == FlightState.PREPARE_OFFBOARD:
            self._prepare_offboard()
        elif self.state == FlightState.ASCEND:
            self._ascend()
        elif self.state == FlightState.HOVER:
            self._hover()
        elif self.state in {FlightState.MANUAL_CONTROL, FlightState.ERROR}:
            # Critical handover rule: publish no Offboard heartbeat or setpoint.
            return

    def _wait_for_local_position(self) -> None:
        now = time.monotonic()
        if not self._local_position_is_fresh(now) or self.current_yaw is None:
            return

        self.home_ned = list(self.local_position)
        self.target_ned = [
            self.home_ned[0],
            self.home_ned[1],
            # PX4 local NED: moving upward makes Z more negative.
            self.home_ned[2] - self.target_altitude_m,
        ]
        self.hold_yaw = self.current_yaw

        self.get_logger().info(
            f"Home NED recorded: {self.home_ned}; "
            f"target NED: {self.target_ned}"
        )
        self._change_state(
            FlightState.WAIT_FOR_ARM,
            "Waiting for pilot to arm the vehicle",
        )

    def _wait_for_arm(self) -> None:
        if not self._local_position_is_fresh(time.monotonic()):
            return
        if not self.vehicle_armed:
            return

        self.prestream_start_time = time.monotonic()
        self._change_state(
            FlightState.PREPARE_OFFBOARD,
            "Vehicle armed; pre-streaming the hold setpoint",
        )

    def _prepare_offboard(self) -> None:
        now = time.monotonic()
        if not self._require_fresh_position(now):
            return

        if not self.vehicle_armed:
            self._set_error("Vehicle disarmed while preparing Offboard")
            return

        if self.home_ned is None:
            self._set_error("Home position is unavailable")
            return

        self._publish_position_setpoint(self.home_ned)

        if self.vehicle_in_offboard:
            self._change_state(
                FlightState.ASCEND,
                f"Offboard confirmed; ascending to {self.target_altitude_m:g} m",
            )
            return

        if self.prestream_start_time is None:
            self.prestream_start_time = now

        if now - self.prestream_start_time < self.prestream_duration_s:
            return

        if (
            self.last_offboard_request_time is None
            or now - self.last_offboard_request_time
            >= self.offboard_request_interval_s
        ):
            self._request_offboard()
            self.last_offboard_request_time = now

    def _ascend(self) -> None:
        now = time.monotonic()
        if not self._active_control_is_safe(now):
            return
        if self.target_ned is None or self.local_position is None:
            self._set_error("Target or local position is unavailable")
            return

        self._publish_position_setpoint(self.target_ned)

        altitude_error = abs(self.local_position[2] - self.target_ned[2])
        if altitude_error > self.altitude_tolerance_m:
            self.arrival_start_time = None
            return

        if self.arrival_start_time is None:
            self.arrival_start_time = now
            return

        if now - self.arrival_start_time >= self.arrival_stable_s:
            self._change_state(
                FlightState.HOVER,
                f"Reached {self.target_altitude_m:g} m; holding position",
            )

    def _hover(self) -> None:
        if not self._active_control_is_safe(time.monotonic()):
            return
        if self.target_ned is None:
            self._set_error("Hover target is unavailable")
            return

        self._publish_position_setpoint(self.target_ned)

    # ------------------------------------------------------------------
    # Safety and state helpers.
    # ------------------------------------------------------------------
    def _active_control_is_safe(self, now: float) -> bool:
        if not self._require_fresh_position(now):
            return False
        if not self.vehicle_armed:
            self._set_error("Vehicle disarmed during autonomous control")
            return False
        if not self.vehicle_in_offboard:
            self._set_error("Offboard mode was lost without Position takeover")
            return False
        return True

    def _require_fresh_position(self, now: float) -> bool:
        if self._local_position_is_fresh(now):
            return True
        self._set_error("PX4 local position is unavailable or stale")
        return False

    def _local_position_is_fresh(self, now: float) -> bool:
        return (
            self.local_position is not None
            and self.last_local_position_time is not None
            and now - self.last_local_position_time
            <= self.local_position_timeout_s
        )

    def _change_state(self, new_state: FlightState, message: str) -> None:
        if new_state == self.state:
            return
        old_state = self.state
        self.state = new_state
        self.get_logger().info(
            f"State: {old_state.name} -> {new_state.name}; {message}"
        )

    def _set_error(self, message: str) -> None:
        if self.state == FlightState.ERROR:
            return
        old_state = self.state
        self.state = FlightState.ERROR
        self.get_logger().error(
            f"State: {old_state.name} -> ERROR; {message}. "
            "Offboard output has stopped; PX4 failsafe is responsible for recovery."
        )

    # ------------------------------------------------------------------
    # PX4 output helpers.
    # ------------------------------------------------------------------
    def _request_offboard(self) -> None:
        msg = VehicleCommand()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.param1 = 1.0
        msg.param2 = 6.0
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.vehicle_command_pub.publish(msg)

    def _publish_position_setpoint(self, position: list[float]) -> None:
        timestamp_us = self.get_clock().now().nanoseconds // 1000

        control_mode = OffboardControlMode()
        control_mode.timestamp = timestamp_us
        control_mode.position = True
        control_mode.velocity = False
        control_mode.acceleration = False
        control_mode.attitude = False
        control_mode.body_rate = False
        self.offboard_control_mode_pub.publish(control_mode)

        setpoint = TrajectorySetpoint()
        setpoint.timestamp = timestamp_us
        setpoint.position = [
            float(position[0]),
            float(position[1]),
            float(position[2]),
        ]
        setpoint.velocity = [math.nan, math.nan, math.nan]
        setpoint.acceleration = [math.nan, math.nan, math.nan]
        if hasattr(setpoint, "jerk"):
            setpoint.jerk = [math.nan, math.nan, math.nan]
        setpoint.yaw = (
            float(self.hold_yaw) if self.hold_yaw is not None else math.nan
        )
        setpoint.yawspeed = math.nan
        self.trajectory_setpoint_pub.publish(setpoint)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AscendHoverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
