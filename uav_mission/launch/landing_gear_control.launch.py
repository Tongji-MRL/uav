from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="uav_mission",
                executable="landing_gear_control",
                name="landing_gear_control",
                output="screen",
            )
        ]
    )