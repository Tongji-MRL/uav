from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="uav_mission",
                executable="px4_gripper_control",
                name="px4_gripper_control",
                output="screen",
                parameters=[
                    {
                        "actuator_number": 1,
                        "open_value": 1.0,
                        "close_value": -1.0,
                        "require_fmu_status": True,
                        "fmu_status_timeout_s": 2.0,
                        "repeat_count": 3,
                        "repeat_interval_s": 0.10,
                        "target_system": 1,
                        "target_component": 1,
                        "source_system": 1,
                        "source_component": 1,
                    }
                ],
            )
        ]
    )
