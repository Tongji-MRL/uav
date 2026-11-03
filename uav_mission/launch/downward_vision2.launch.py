import os

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory(
        "uav_mission"
    )

    config_file = os.path.join(
        package_share,
        "config",
        "downward_vision.yaml",
    )

    return LaunchDescription(
        [
            Node(
                package="uav_mission",
                executable="downward_vision2",
                name="downward_vision2",
                output="screen",
                parameters=[
                    {"config_file": config_file}
                ],
            )
        ]
    )