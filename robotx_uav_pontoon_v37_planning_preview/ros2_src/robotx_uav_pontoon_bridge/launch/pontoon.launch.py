from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory('robotx_uav_pontoon_bridge')
    return LaunchDescription([
        Node(package='robotx_uav_pontoon_bridge', executable='pontoon_node',
             name='uav_pontoon_v37_bridge', output='screen',
             parameters=[os.path.join(pkg, 'config', 'pontoon.yaml')]),
    ])
