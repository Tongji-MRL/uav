from setuptools import find_packages, setup


package_name = "uav_common"


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "shapely>=1.7,<3"],
    zip_safe=True,
    maintainer="RobotX Team",
    maintainer_email="robotx@example.com",
    description="Shared Python helpers for RobotX UAV ROS 2 nodes.",
    license="Apache-2.0",
)

