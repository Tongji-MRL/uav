#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

from px4_msgs.msg import LandingGear
from std_msgs.msg import String


class LandingGearControlNode(Node):

    def __init__(self) -> None:
        super().__init__("landing_gear_control")

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.gear_pub = self.create_publisher(
            LandingGear,
            "/fmu/in/landing_gear",
            px4_qos,
        )

        self.command_sub = self.create_subscription(
            String,
            "/landing_gear/command",
            self._on_command,
            10,
        )

        self.get_logger().info(
            "Landing gear control node started"
        )

    def _on_command(self, msg: String) -> None:
        command = msg.data.strip().lower()

        if command == "down":
            gear_state = LandingGear.GEAR_DOWN
        elif command == "up":
            gear_state = LandingGear.GEAR_UP
        else:
            self.get_logger().warning(
                "Unknown command; use 'up' or 'down'"
            )
            return

        gear_msg = LandingGear()
        gear_msg.timestamp = (
            self.get_clock().now().nanoseconds // 1000
        )
        gear_msg.landing_gear = gear_state

        self.gear_pub.publish(gear_msg)

        self.get_logger().info(
            f"Landing gear command sent once: {command}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)

    node = LandingGearControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()