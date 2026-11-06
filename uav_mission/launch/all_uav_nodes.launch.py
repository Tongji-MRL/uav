import os

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)

# 独立 launch 统一启动；识别模型由 mission 的视觉模式指令启停。
def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory(
        "uav_mission"
    )
    launch_dir = os.path.join(
        package_share,
        "launch",
    )

    launch_files = [
        "mission_perception.launch.py",
        "downward_vision2.launch.py",
        "px4_gripper_control.launch.py",
        "landing_gear_control.launch.py",
        "mission.launch.py",
        "uav_heartbeat.launch.py",
        "uav_status_json.launch.py",
    ]

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(launch_dir, filename)
                )
            )
            for filename in launch_files
        ]
    )
