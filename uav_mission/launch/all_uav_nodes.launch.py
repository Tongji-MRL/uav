import os

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)

#这里直接复用了原来的4份 launch，所以以后调整某个节点的参数，只需要修改对应的独立 launch 文件。
def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory(
        "uav_mission"
    )
    launch_dir = os.path.join(
        package_share,
        "launch",
    )

    launch_files = [
        "downward_vision2.launch.py",
        "px4_gripper_control.launch.py",
        "landing_gear_control.launch.py",
        "mission.launch.py",
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