from setuptools import setup
from glob import glob
import os
package_name = 'robotx_uav_pontoon_bridge'
setup(
    name=package_name, version='0.1.0', packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='RobotX UAV Team', maintainer_email='robotx@example.invalid',
    description='RobotX UAV Pontoon V3.7 ROS2 bridge', license='Apache-2.0',
    entry_points={'console_scripts': ['pontoon_node = robotx_uav_pontoon_bridge.node:main']},
)
