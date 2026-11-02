"""ROS 2 node that controls a PWM gripper through PX4.

PX4 output setup (example for MAIN 5):
    PWM_MAIN_FUNC6 = 301  # Peripheral via Actuator Set 1

The node does not move the gripper at startup. Publish a JSON string to
``/task3/gripper/command``. Only its ``command`` field is used:
``{"command": "close"}`` closes and ``{"command": "open"}`` opens.
"""

import json
import math

import rclpy
from px4_msgs.msg import VehicleCommand, VehicleCommandAck, VehicleStatus
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String


GRIPPER_COMMAND_TOPIC = "/task3/gripper/command"


class Px4GripperControl(Node):
    """Send MAV_CMD_DO_SET_ACTUATOR to one PX4 actuator-set function."""

    def __init__(self):
        super().__init__("px4_gripper_control")

        self.declare_parameter("actuator_number", 1)
        self.declare_parameter("open_value", 1.0)
        self.declare_parameter("close_value", -1.0)
        self.declare_parameter("require_fmu_status", True)
        self.declare_parameter("fmu_status_timeout_s", 2.0)
        self.declare_parameter("repeat_count", 1)
        self.declare_parameter("repeat_interval_s", 0.10)
        self.declare_parameter("target_system", 1)
        self.declare_parameter("target_component", 1)
        self.declare_parameter("source_system", 1)
        self.declare_parameter("source_component", 1)

        self.actuator_number = int(
            self.get_parameter("actuator_number").value
        )
        self.open_value = float(self.get_parameter("open_value").value)
        self.close_value = float(self.get_parameter("close_value").value)
        self.require_fmu_status = bool(
            self.get_parameter("require_fmu_status").value
        )
        self.fmu_status_timeout_s = float(
            self.get_parameter("fmu_status_timeout_s").value
        )
        self.repeat_count = int(self.get_parameter("repeat_count").value)
        self.repeat_interval_s = float(
            self.get_parameter("repeat_interval_s").value
        )
        self.target_system = int(self.get_parameter("target_system").value)
        self.target_component = int(
            self.get_parameter("target_component").value
        )
        self.source_system = int(self.get_parameter("source_system").value)
        self.source_component = int(
            self.get_parameter("source_component").value
        )

        self._validate_parameters()

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.command_pub = self.create_publisher(
            VehicleCommand,
            "/fmu/in/vehicle_command",
            px4_qos,
        )
        self.create_subscription(
            VehicleCommandAck,
            "/fmu/out/vehicle_command_ack",
            self.command_ack_callback,
            px4_qos,
        )
        # VehicleStatus has MESSAGE_VERSION=1 in PX4/px4_msgs v1.17, so the
        # DDS bridge exposes the version suffix in the ROS topic name.
        self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status_v1",
            self.vehicle_status_callback,
            px4_qos,
        )
        self.create_subscription(
            String,
            GRIPPER_COMMAND_TOPIC,
            self.gripper_command_callback,
            10,
        )

        self.last_fmu_status_rx_s = None
        self.pending_value = None
        self.pending_label = None
        self.pending_transmissions = 0
        self.next_transmission_s = 0.0
        self.repeat_timer = self.create_timer(0.02, self.repeat_timer_callback)

        self.get_logger().warning(
            "No startup movement is sent. Remove propellers and calibrate the "
            "gripper PWM endpoints in QGroundControl before the first command."
        )
        self.get_logger().info(
            f"Ready: {GRIPPER_COMMAND_TOPIC} (JSON command=open|close), actuator set "
            f"{self.actuator_number}, open={self.open_value:.3f}, "
            f"close={self.close_value:.3f}"
        )

    def _validate_parameters(self):
        if not 1 <= self.actuator_number <= 6:
            raise ValueError("actuator_number must be in the range 1..6")
        for name, value in (
            ("open_value", self.open_value),
            ("close_value", self.close_value),
        ):
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [-1.0, 1.0]")
        if self.open_value == self.close_value:
            raise ValueError("open_value and close_value must be different")
        if not 1 <= self.repeat_count <= 20:
            raise ValueError("repeat_count must be in the range 1..20")
        if not 0.02 <= self.repeat_interval_s <= 2.0:
            raise ValueError("repeat_interval_s must be in the range 0.02..2.0")
        if not 0.1 <= self.fmu_status_timeout_s <= 30.0:
            raise ValueError("fmu_status_timeout_s must be in 0.1..30.0")

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def timestamp_us(self):
        return self.get_clock().now().nanoseconds // 1000

    def vehicle_status_callback(self, _msg):
        self.last_fmu_status_rx_s = self.now_s()

    def fmu_status_is_fresh(self):
        return (
            self.last_fmu_status_rx_s is not None
            and self.now_s() - self.last_fmu_status_rx_s
            <= self.fmu_status_timeout_s
        )

    def gripper_command_callback(self, msg):
        try:
            payload = json.loads(msg.data)

        except json.JSONDecodeError as error:
            self.get_logger().error(
                f"Command rejected: invalid JSON on {GRIPPER_COMMAND_TOPIC}: {error}"
            )
            return

        if not isinstance(payload, dict):
            self.get_logger().error(
                "Command rejected: JSON payload must be an object with a command field."
            )
            return

        command = payload.get("command")

        if not isinstance(command, str):
            self.get_logger().error(
                "Command rejected: JSON field command must be the string open or close."
            )
            return

        command = command.strip().lower()

        if command not in ("open", "close"):
            self.get_logger().error(
                f"Command rejected: unsupported command={command!r}; use open or close."
            )
            return

        if self.require_fmu_status and not self.fmu_status_is_fresh():
            self.get_logger().error(
                "Command rejected: no fresh /fmu/out/vehicle_status_v1. "
                "Check the Micro XRCE-DDS Agent and the Orin-FMU link."
            )
            return

        self.pending_label = command.upper()
        self.pending_value = self.close_value if command == "close" else self.open_value
        self.pending_transmissions = self.repeat_count
        self.next_transmission_s = self.now_s()
        self.get_logger().info(
            f"Requested {self.pending_label}: normalized actuator value "
            f"{self.pending_value:.3f}"
        )

    def repeat_timer_callback(self):
        if self.pending_transmissions <= 0:
            return
        if self.now_s() < self.next_transmission_s:
            return

        self.publish_actuator_command(self.pending_value)
        self.pending_transmissions -= 1
        self.next_transmission_s = self.now_s() + self.repeat_interval_s

    def publish_actuator_command(self, value):
        msg = VehicleCommand()
        msg.timestamp = self.timestamp_us()

        # MAV_CMD_DO_SET_ACTUATOR carries actuator sets 1..6 in params 1..6.
        # NaN leaves every other actuator-set function unchanged. PX4 v1.17's
        # FunctionActuatorSet currently accepts set index 0 in param7.
        msg.param1 = math.nan
        msg.param2 = math.nan
        msg.param3 = math.nan
        msg.param4 = math.nan
        msg.param5 = math.nan
        msg.param6 = math.nan
        setattr(msg, f"param{self.actuator_number}", float(value))
        msg.param7 = 0.0
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_ACTUATOR
        msg.target_system = self.target_system
        msg.target_component = self.target_component
        msg.source_system = self.source_system
        msg.source_component = self.source_component
        msg.confirmation = 0
        msg.from_external = True
        self.command_pub.publish(msg)

    def command_ack_callback(self, msg):
        if msg.command != VehicleCommand.VEHICLE_CMD_DO_SET_ACTUATOR:
            return

        results = {
            VehicleCommandAck.VEHICLE_CMD_RESULT_ACCEPTED: "ACCEPTED",
            VehicleCommandAck.VEHICLE_CMD_RESULT_TEMPORARILY_REJECTED:
                "TEMPORARILY_REJECTED",
            VehicleCommandAck.VEHICLE_CMD_RESULT_DENIED: "DENIED",
            VehicleCommandAck.VEHICLE_CMD_RESULT_UNSUPPORTED: "UNSUPPORTED",
            VehicleCommandAck.VEHICLE_CMD_RESULT_FAILED: "FAILED",
            VehicleCommandAck.VEHICLE_CMD_RESULT_IN_PROGRESS: "IN_PROGRESS",
            VehicleCommandAck.VEHICLE_CMD_RESULT_CANCELLED: "CANCELLED",
        }
        result = results.get(msg.result, f"UNKNOWN({msg.result})")
        log = self.get_logger().info if msg.result == 0 else self.get_logger().error
        log(
            f"PX4 DO_SET_ACTUATOR acknowledgement: {result}. "
            "ACK confirms command handling, not physical gripper motion."
        )


def main(args=None):
    rclpy.init(args=args)
    node = Px4GripperControl()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
