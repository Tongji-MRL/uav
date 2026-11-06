"""Launch the PX4 ascend-hover state-machine node."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("uav_mission"))
    config_file = package_share / "config" / "ascend_hover.yaml"

    return LaunchDescription(
        [
            Node(
                package="uav_mission",
                executable="ascend_hover",
                name="uav_ascend_hover",
                output="screen",
                parameters=[str(config_file)],
            )
        ]
    )
