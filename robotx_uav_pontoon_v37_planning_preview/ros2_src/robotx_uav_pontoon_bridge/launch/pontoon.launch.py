from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory('robotx_uav_pontoon_bridge')
    return LaunchDescription([
        DeclareLaunchArgument('source_dir', default_value=''),
        DeclareLaunchArgument('mission_control_enabled', default_value='false'),
        DeclareLaunchArgument('mission_control_topic', default_value='/tongji_mrl/tsky/downward_vision_control'),
        Node(package='robotx_uav_pontoon_bridge', executable='pontoon_node',
             name='uav_pontoon_v37_bridge', output='screen',
             parameters=[os.path.join(pkg, 'config', 'pontoon.yaml'), {
                 'source_dir': ParameterValue(LaunchConfiguration('source_dir'), value_type=str),
                 'mission_control.enabled': ParameterValue(LaunchConfiguration('mission_control_enabled'), value_type=bool),
                 'mission_control.topic': ParameterValue(LaunchConfiguration('mission_control_topic'), value_type=str),
             }]),
    ])
