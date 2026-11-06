from glob import glob
from setuptools import find_packages, setup


package_name = "uav_mission"


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="RobotX Team",
    maintainer_email="robotx@example.invalid",
    description=(
        "Single-node RobotX UAV mission state machine for MAP, TASK1, "
        "TASK3, and TASK4."
    ),
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "uav_mission_manager = "
            "uav_mission.uav_mission_node:main",
            "downward_vision = "
            "uav_mission.downward_vision_node:main",
            "downward_vision2 = "
            "uav_mission.downward_vision_node2:main",
            "mission_perception = uav_mission.mission_perception_node:main",
            "px4_perception_pose = uav_mission.px4_perception_pose_node:main",
            "px4_gripper_control = "
            "uav_mission.px4_gripper_control_node:main",
            "landing_gear_control = "
            "uav_mission.landing_gear_control_node:main",
            "uav_heartbeat = "
            "uav_mission.uav_heartbeat_node:main",
            "uav_status_json = "
            "uav_mission.uav_status_json_node:main",
        ],
    },
)

