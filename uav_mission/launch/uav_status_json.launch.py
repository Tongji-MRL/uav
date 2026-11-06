import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("uav_mission")
    config_file = os.path.join(
        package_share,
        "config",
        "uav_status.yaml",
    )

    return LaunchDescription(
        [
            Node(
                package="uav_mission",
                executable="uav_status_json",
                name="uav_status_json",
                output="screen",
                parameters=[{"config_file": config_file}],
            )
        ]
    )
