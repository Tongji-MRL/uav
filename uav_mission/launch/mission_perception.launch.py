"""Start buoy/dock workers and the mission map recorder, initially inactive."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    mission = Path(get_package_share_directory("uav_mission"))
    beacon = Path(get_package_share_directory("color_light_detector"))
    dock = Path(get_package_share_directory("robotx_uav_pontoon_bridge"))
    defaults = {
        "perception_config": str(mission / "config/mission_perception.yaml"),
        "buoy_position_config": str(beacon / "config/task1_position_px4.yaml"),
        "buoy_detector_config": str(beacon / "config/task1_arducam_45.yaml"),
        "buoy_camera_config": str(beacon / "config/arducam_current_mode.yaml"),
        "buoy_model_path": str(beacon / "weights/best.pt"),
        "buoy_image_topic": "/arducam/image_native",
        "start_buoy_camera": "true",
        "dock_config": str(dock / "config/pontoon.yaml"),
        "dock_image_topic": "/oak/rgb/image_raw",
        "dock_camera_info_topic": "/oak/rgb/camera_info",
        "dock_depth_topic": "/oak/stereo/image_raw",
        "dock_input_rectified": "false",
        "downward_config": str(mission / "config/downward_vision.yaml"),
    }
    return LaunchDescription([
        *(DeclareLaunchArgument(name, default_value=value) for name, value in defaults.items()),
        Node(
            package="uav_mission", executable="px4_perception_pose", name="px4_perception_pose", output="screen",
            parameters=[{"config_file": LaunchConfiguration("perception_config"),
                         "downward_config_file": LaunchConfiguration("downward_config"),
                         "camera_info_topic": LaunchConfiguration("dock_camera_info_topic")}],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(beacon / "launch/arducam_task1.launch.py")),
            launch_arguments={
                "position_config": LaunchConfiguration("buoy_position_config"),
                "detector_config": LaunchConfiguration("buoy_detector_config"),
                "camera_config": LaunchConfiguration("buoy_camera_config"),
                "model_path": LaunchConfiguration("buoy_model_path"),
                "image_topic": LaunchConfiguration("buoy_image_topic"),
                "start_camera": LaunchConfiguration("start_buoy_camera"),
                "enable_position": "true",
                "start_map": "false",  # The unified recorder below owns mission MAP output.
                "mission_control_enabled": "true",
                "mission_control_topic": "/tongji_mrl/tsky/downward_vision_control",
            }.items(),
        ),
        Node(
            package="robotx_uav_pontoon_bridge", executable="pontoon_node",
            name="uav_pontoon_v37_bridge", output="screen",
            parameters=[LaunchConfiguration("dock_config"), {
                "mission_control.enabled": True,
                "mission_control.topic": "/tongji_mrl/tsky/downward_vision_control",
                "image_topic": LaunchConfiguration("dock_image_topic"),
                "camera_info_topic": LaunchConfiguration("dock_camera_info_topic"),
                "aligned_depth_topic": LaunchConfiguration("dock_depth_topic"),
                "input_rectified": ParameterValue(LaunchConfiguration("dock_input_rectified"), value_type=bool),
            }],
        ),
        Node(
            package="uav_mission", executable="mission_perception", name="mission_perception", output="screen",
            parameters=[{"config_file": LaunchConfiguration("perception_config")}],
        ),
    ])
