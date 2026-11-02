from glob import glob

from setuptools import find_packages, setup

package_name = 'color_light_detector'

setup(
    name=package_name,
    version='0.2.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/weights', glob('weights/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aurora',
    maintainer_email='aurora@todo.todo',
    description='RobotX new-beacon plate/aperture color and blink perception.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'arducam_camera = color_light_detector.arducam_camera_node:main',
            'task1_beacon = color_light_detector.task1_beacon_node:main',
            'task1_beacon_map = color_light_detector.task1_beacon_map_node:main',
        ],
    },
)
