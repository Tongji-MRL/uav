"""Arducam B0495 45-degree perception; no camera parameter controls."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def required_exit(event, context):
    if context.is_shutdown:
        return []
    raise RuntimeError(f"Required Task1 process exited (code {event.returncode}); stopping the launch.")


def generate_launch_description():
    share = Path(get_package_share_directory("color_light_detector"))
    camera = Node(
        package="color_light_detector", executable="arducam_camera",
        name="arducam_camera", output="screen",
        prefix=[LaunchConfiguration("python_executable")],
        parameters=[LaunchConfiguration("camera_config"), {
            "camera_mode": ParameterValue(LaunchConfiguration("camera_mode"), value_type=str),
            "native_image_topic": ParameterValue(LaunchConfiguration("image_topic"), value_type=str),
        }], condition=IfCondition(LaunchConfiguration("start_camera")),
    )
    detector = Node(
        package="color_light_detector", executable="task1_beacon",
        name="task1_beacon", output="screen",
        prefix=[LaunchConfiguration("python_executable")],
        parameters=[LaunchConfiguration("detector_config"), {
            "image_topic": ParameterValue(LaunchConfiguration("image_topic"), value_type=str),
            "model_path": ParameterValue(LaunchConfiguration("model_path"), value_type=str),
            "model_task": ParameterValue(LaunchConfiguration("model_task"), value_type=str),
            "position.enabled": ParameterValue(LaunchConfiguration("enable_position"), value_type=bool),
            "position.config_file": ParameterValue(LaunchConfiguration("position_config"), value_type=str),
        }],
    )
    mapping = Node(
        package="color_light_detector", executable="task1_beacon_map",
        name="task1_beacon_map", output="screen",
        prefix=[LaunchConfiguration("python_executable")],
        parameters=[{"map_config": ParameterValue(
            LaunchConfiguration("map_config"), value_type=str)}],
        condition=IfCondition(LaunchConfiguration("start_map")),
    )
    return LaunchDescription([
        SetEnvironmentVariable("YOLO_AUTOINSTALL", "false"),
        SetEnvironmentVariable("YOLO_OFFLINE", "true"),
        DeclareLaunchArgument("image_topic", default_value="/arducam/image_native"),
        DeclareLaunchArgument("camera_config", default_value=str(share / "config/arducam_current_mode.yaml")),
        DeclareLaunchArgument("start_camera", default_value="true"),
        DeclareLaunchArgument("camera_mode", default_value="max",
                              description="max native resolution/FPS, or current to keep the existing device mode"),
        DeclareLaunchArgument("model_path", default_value=str(share / "weights/best.pt")),
        DeclareLaunchArgument("model_task", default_value="segment",
                              description="segment for included weights; detect for a replacement detect model"),
        DeclareLaunchArgument("detector_config", default_value=str(
            share / "config/task1_arducam_45.yaml")),
        DeclareLaunchArgument("position_config", default_value=str(
            share / "config/task1_position_b0495_provisional.yaml")),
        DeclareLaunchArgument("map_config", default_value=str(
            share / "config/task1_beacon_map_unconfigured.yaml")),
        DeclareLaunchArgument("python_executable", default_value="python3",
                              description="Python with ROS 2, torch and ultralytics"),
        DeclareLaunchArgument("enable_position", default_value="true",
                              description="Enable calibrated ray-plane range; missing calibration stays invalid"),
        DeclareLaunchArgument("start_map", default_value="false",
                              description="Optional map prototype; not field validated"),
        RegisterEventHandler(OnProcessExit(target_action=camera, on_exit=required_exit)),
        RegisterEventHandler(OnProcessExit(target_action=detector, on_exit=required_exit)),
        RegisterEventHandler(OnProcessExit(target_action=mapping, on_exit=required_exit)),
        camera,
        detector,
        mapping,
    ])
