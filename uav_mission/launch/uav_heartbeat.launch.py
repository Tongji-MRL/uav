from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="uav_mission",
                executable="uav_heartbeat",
                name="uav_heartbeat",
                output="screen",
                parameters=[
                    {
                        "publish_rate_hz": 5.0,
                        "telemetry_timeout_s": 1.0,
                        "heartbeat_topic": (
                            "/tongji_mrl/tsky/heartbeat"
                        ),
                        "progress_topic": (
                            "/tongji_mrl/tsky/progress"
                        ),
                        "vehicle_status_topic": (
                            "/fmu/out/vehicle_status_v1"
                        ),
                        "vehicle_global_position_topic": (
                            "/fmu/out/vehicle_global_position"
                        ),
                        "vehicle_local_position_topic": (
                            "/fmu/out/vehicle_local_position_v1"
                        ),
                        "vehicle_attitude_topic": (
                            "/fmu/out/vehicle_attitude"
                        ),
                        "vehicle_land_detected_topic": (
                            "/fmu/out/vehicle_land_detected"
                        ),
                    }
                ],
            )
        ]
    )