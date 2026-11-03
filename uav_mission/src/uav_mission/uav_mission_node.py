#!/usr/bin/env python3
"""Single ROS 2 node for the complete RobotX UAV mission."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from sensor_msgs.msg import Range

import rclpy
import yaml
from px4_msgs.msg import (
    BatteryStatus,
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleAttitude,
    VehicleLocalPosition,
    VehicleStatus,
    VehicleCommand,
)
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from std_msgs.msg import String

from uav_common.safety_monitor import SafetyMonitor

from uav_common.waypoint_planner import (
    WaypointOrderPlanner,
    WaypointPlanner,
)

from uav_common.map_detection_to_ned import (
    quat_rotate_frd_to_ned,
)

from uav_common.position_setpoint_interpolator import (
    PositionSetpointInterpolator,
)

def select_flashing_blue_buoy(payload: Any) -> dict[str, Any] | None:
    """Return the first flashing blue buoy from an objects payload."""
    if not isinstance(payload, dict):
        return None

    objects = payload.get("objects")
    if not isinstance(objects, list):
        return None

    for item in objects:
        if not isinstance(item, dict):
            continue
        if (
            str(item.get("type", "")).strip().lower() == "buoy"
            and str(item.get("color", "")).strip().lower() == "blue"
            and str(item.get("state", "")).strip().lower() == "flashing"
        ):
            return item

    return None


class UavMissionNode(Node):
    """Run MAP, TASK1, TASK3, and TASK4 through one ROS 2 node."""

    def __init__(self) -> None:
        super().__init__("uav_mission_manager")

        # ==============================================================
        # 1. 读取配置
        # ==============================================================
        self.declare_parameter("config_file", "")
        config = self._load_config()
        runtime = config.get("runtime", {})
        topics = config.get("topics", {})
        target_cfg = config.get("target", {})
        motion_cfg = config.get("motion", {})
        follow_cfg = config.get("follow", {})
        safety_cfg = config.get("safety", {})
        map_cfg = config.get("map", {})
        task3_cfg = config.get("task3", {})
        task4_cfg = config.get("task4", {})

        # 创建安全监控对象
        self.safety_monitor = SafetyMonitor(safety_cfg)

        self.position_interpolator = (
            PositionSetpointInterpolator(
                max_speed_m_s=0.5,
                time_constant_s=0.6,
                max_tracking_error_m=0.8,
            )
        )

        self.update_rate_hz = float(runtime.get("update_rate_hz", 10.0))
        self.progress_rate_hz = float(runtime.get("progress_rate_hz", 1.0))
        if self.update_rate_hz <= 0.0 or self.progress_rate_hz <= 0.0:
            raise ValueError("update_rate_hz and progress_rate_hz must be positive")

        self.receive_timeout_s = float(target_cfg.get("receive_timeout_s", 15.0))
        self.x_tolerance_m = float(target_cfg.get("x_tolerance_m", 0.30))
        self.y_tolerance_m = float(target_cfg.get("y_tolerance_m", 0.30))
        self.z_tolerance_m = float(target_cfg.get("z_tolerance_m", 0.20))
        self.local_position_timeout_s = float(
            target_cfg.get("local_position_timeout_s", 1.0)
        )
        self.fly_speed_m_s = float(motion_cfg.get("fly_speed_m_s", 1.0))
        self.follow_speed_m_s = float(follow_cfg.get("speed_m_s", 1.0))
        self.follow_xy_tolerance_m = float(
            follow_cfg.get("xy_tolerance_m", 0.30)
        )
        self.follow_position_timeout_s = float(
            follow_cfg.get("position_timeout_s", 1.0)
        )

        self.search_height_m = float(
            map_cfg.get("search_height_m", 1.5)
        )
        self.ascend_speed_m_s = float(
            map_cfg.get("ascend_speed_m_s", 0.3)
        )

        self.route_speed_m_s = float(
            map_cfg.get("route_speed_m_s", 0.8)
        )
        self.waypoint_tolerance_m = float(
            map_cfg.get("waypoint_tolerance_m", 0.30)
        )

        self.task3_fly_speed_m_s = float(
            task3_cfg.get("fly_speed_m_s", 0.5)
        )
        self.task3_platform_tolerance_m = float(
            task3_cfg.get("platform_tolerance_m", 0.30)
        )

        self.task3_visual_move_speed_m_s = float(
            task3_cfg.get("visual_move_speed_m_s", 0.20)
        )
        self.task3_gripper_xy_tolerance_m = float(
            task3_cfg.get("gripper_xy_tolerance_m", 0.05)
        )
        self.task3_vision_timeout_s = float(
            task3_cfg.get("vision_timeout_s", 1.0)
        )

        self.task3_pick_speed_m_s = float(
            task3_cfg.get("pick_speed_m_s", 0.10)
        )

        self.task3_pick_tolerance_m = float(
            task3_cfg.get("pick_tolerance_m", 0.05)
        )

        self.task3_return_speed_m_s = float(
            task3_cfg.get("return_speed_m_s", 0.20)
        )
        self.task3_return_tolerance_m = float(
            task3_cfg.get("return_tolerance_m", 0.05)
        )
        self.task3_gripper_close_wait_s = float(
            task3_cfg.get("gripper_close_wait_s", 5.0)
        )

        self.task3_verify_duration_s = float(
            task3_cfg.get("verify_duration_s", 2.0)
        )

        self.task3_verify_max_retries = int(
            task3_cfg.get("verify_max_retries", 3)
        )

        self.task4_fly_speed_m_s = float(
            task4_cfg.get("fly_speed_m_s", 0.5)
        )

        self.task4_xy_tolerance_m = float(
            task4_cfg.get("xy_tolerance_m", 0.30)
        )

        self.task3_xy_small_tolerance_m = float(
            task3_cfg.get("xy_small_tolerance_m", 0.03)
        )
        self.task3_xy_large_tolerance_m = float(
            task3_cfg.get("xy_large_tolerance_m", 0.10)
        )
        self.task3_xy_servo_kp = float(
            task3_cfg.get("xy_servo_kp", 0.8)
        )
        self.task3_align_stable_duration_s = float(
            task3_cfg.get("align_stable_duration_s", 0.5)
        )

        self.task3_range_timeout_s = float(
            task3_cfg.get("range_timeout_s", 0.5)
        )
        self.task3_sensor_to_gripper_m = float(
            task3_cfg.get("sensor_to_gripper_m", 0.24)
        )
        self.task3_grasp_distance_m = float(
            task3_cfg.get("grasp_distance_m", 0.02)
        )
        self.task3_descend_to_tin_speed_m_s = float(
            task3_cfg.get("descend_to_tin_speed_m_s", 0.05)
        )
        self.task3_recover_max_retries = int(
            task3_cfg.get("recover_max_retries", 3)
        )
        # ==============================================================
        # 2. 整体任务状态和公共数据
        # ==============================================================
        self.mission = "MAP"
        self.state = "WAIT_BEGIN"
        self.error: str | None = None
        self.control_started = False

        self.begin_requests = {
            "MAP": False,
            "TASK1": False,
            "TASK3": False,
            "TASK4": False,
        }
        self.finish_requests = {
            "MAP": False,
            "TASK1": False,
            "TASK3": False,
            "TASK4": False,
        }

        self.local_position: list[float] | None = None
        self.last_local_position_time: float | None = None

        self.waypoint_planner = WaypointPlanner()
        self.map_corners: list[list[float]] | None = None
        self.map_waypoints: list[list[float]] | None = None

        self.waypoint_order_planner = WaypointOrderPlanner()
        self.map_route: list[list[float]] = []
        self.map_route_index = 0

        self.vehicle_armed = False
        self.vehicle_in_offboard = False

        self.home_ned: list[float] | None = None

        self.vehicle_attitude_q: list[float] | None = None
        self.current_yaw: float | None = None
        self.fixed_mission_yaw: float | None = None

        self.landing_gear_wait_s = 5.0
        self.landing_gear_wait_start_time: float | None = None

        self.vehicle_in_position = False
        self.error_handover_complete = False
        self.error_mode_request_time: float | None = None
        self.offboard_was_active = False

        self.vertical_move_xy: list[float] | None = None
        self.hold_position_ned: list[float] | None = None
        # ==============================================================
        # TEST 数据
        # ==============================================================
        self.map_debug_last_log_time = 0.0
        self.map_finish_log_written = False

        self.last_published_position_sp = None
        self.vehicle_nav_state = None
        # ==============================================================
        # 3. TASK1 数据
        # ==============================================================
        self.task1_begin_time: float | None = None
        self.task1_buoy: dict[str, Any] | None = None
        self.task1_target: list[float] | None = None
        self.task1_usv_at_start = False
        self.task1_usv_at_end = False
        self.usv_ned_position: list[float] | None = None
        self.last_usv_position_time: float | None = None
        # ==============================================================
        # 3. TASK3 数据
        # ==============================================================
        self.task3_resource_color: str | None = None
        self.task3_destination_color: str | None = None
        self.task3_color_sequence_received = False
        
        self.task3_platform_1_position: list[float] | None = None
        self.task3_platform_2_position: list[float] | None = None
        self.task3_platforms_received = False
        
        self.latest_task3_objects = []
        self.task3_target_platform: list[float] | None = None
        
        self.last_task3_vision_time: float | None = None

        self.task3_detect_tin_start_time: float | None = None
        self.task3_tin_objects: list[dict[str, Any]] = []
        self.task3_tin_count = 0

        self.task3_place_target_ned: list[float] | None = None

        # 最开始到达的平台，以及另一个平台
        self.task3_original_platform_position: list[float] | None = None
        self.task3_other_platform_position: list[float] | None = None

        # 当前飞往/所在的平台身份：ORIGINAL或OTHER
        self.task3_target_platform_role: str | None = None
        self.task3_current_platform_role: str | None = None

        # 找到的resource tin
        self.task3_selected_resource_tin: dict[str, Any] | None = None

        # 平台精确对准后应该进入哪个状态
        self.task3_after_alignment_state = "DETECT_TIN"

        self.task3_pick_target_ned: list[float] | None = None

        self.task3_platform_center_ned: list[float] | None = None
        self.task3_gripper_close_start_time: float | None = None

        self.task3_tin_count_after_pick = 0
        self.task3_verify_start_time: float | None = None

        self.task3_verify_retry_count = 0

        self.task3_place_start_time: float | None = None

        self.task3_finished_start_time: float | None = None

        self.task3_align_stable_start_time: float | None = None

        self.latest_height_range_m: float | None = None
        self.last_height_range_time: float | None = None
        
        self.task3_recover_retry_count = 0

        self.task3_descent_hold_z = None
        # ==============================================================
        # 3. TASK4 数据
        # ==============================================================
        self.task4_target_vehicle: str | None = None
        self.task4_is_executor = False
        self.task4_command_seq: int | None = None
        self.task4_target_position: dict[str, float] | None = None
        self.task4_checkpoint: dict[str, Any] | None = None
        self.task4_ack_start_time: float | None = None

        # TASK4恢复状态映射
        # key：进入TASK4前的状态
        # value：收到resume后应该进入的状态
        # None：该状态不允许进入TASK4，无法安全恢复
        self.task4_resume_state_map = {
            # ==========================================================
            # MAP
            # ==========================================================
            "MAP": {
                # MAP尚未正式开始，不建议执行TASK4
                #"WAIT_BEGIN": None,

                # home_ned尚未完成记录，TASK4无法安全计算飞行高度
                #"RECORD_HOME_COORDINATE": None,

                # UAV可能尚未解锁或进入Offboard
                #"PREPARE_TAKEOFF": None,

                "WAIT_GEAR_UP": "WAIT_GEAR_UP",

                # 继续上升到搜索高度
                "ASCEND": "ASCEND",

                # 从当前位置继续飞向当前航点
                "FOLLOW_ROUTE": "FOLLOW_ROUTE",

                # MAP已经完成，继续等待TASK1
                "MAP_FINISH": "MAP_FINISH",
            },

            # ==========================================================
            # TASK1
            # ==========================================================
            "TASK1": {
                # 继续等待TASK1开始条件
                "WAIT_BEGIN": "WAIT_BEGIN",

                # 重新开始接收目标，并重置接收超时计时器
                "RECEIVE_TARGET": "RECEIVE_TARGET",

                # 从当前位置重新飞向task1_target
                "FLY": "FLY",

                # 不直接恢复HOVER，先重新确认已经到达目标
                "HOVER": "FLY",

                # 直接追踪USV当前坐标，不返回USV过去的位置
                "FOLLOW": "FOLLOW",

                # TASK1已经完成，继续等待TASK3
                "TASK1_FINISH": "TASK1_FINISH",
            },

            # ==========================================================
            # TASK3
            # ==========================================================
            "TASK3": {
                # 继续等待颜色和平台信息
                "WAIT_BEGIN": "WAIT_BEGIN",

                # 从当前位置继续前往目标平台
                "FLY_TO_PLATFORM": "FLY_TO_PLATFORM",

                # 先重新粗略飞回平台，再进行视觉精确对准
                "MOVE_TO_PLATFORM": "FLY_TO_PLATFORM",

                # 重新返回平台、对准并重新检测TIN
                "DETECT_TIN": "FLY_TO_PLATFORM",

                # 不使用旧的检测结果，重新对准并检测TIN
                "CHECK_PLATFORM": "FLY_TO_PLATFORM",

                "ALIGN_TIN_XY": "FLY_TO_PLATFORM",
                "DESCEND_TO_TIN": "FLY_TO_PLATFORM",
                "RECOVER_ASCEND": "FLY_TO_PLATFORM",

                # close命令尚未发送，重新检测并规划抓取////觉得时间太短，应该不会在这个状态发出task4的事情
                #"PICK": "FLY_TO_PLATFORM",

                # close命令已经发送，返回平台后重新验证
                "WAIT_GRIPPER_CLOSE": "RETURN_SEARCH_HEIGHT",

                # 返回平台重新对准，然后验证抓取结果
                "RETURN_SEARCH_HEIGHT": "RETURN_SEARCH_HEIGHT",

                # 清除旧验证画面，返回平台重新验证
                "VERIFY": "RETURN_SEARCH_HEIGHT",

                # 从当前位置继续前往另一个平台
                "FLY_TO_OTHER_PLATFORM": "FLY_TO_OTHER_PLATFORM",

                # 返回目标平台重新对准，并重新识别目标circle
                "PLACE_RESOURCE": "FLY_TO_PLATFORM",

                # 旧的绝对放置坐标可能失效，重新识别circle
                "MOVE_TO_PLACE_POSITION": "FLY_TO_PLATFORM",

                # open命令尚未发送，重新对准并规划放置///上下文应该对让此时知道，是place阶段还是pick阶段///时间太短，不考虑
                #"PLACE": "FLY_TO_PLATFORM",

                # open命令已经发送，认为放置动作已经执行
                "WAIT_FOR_PLACE": "ASCEND_AFTER_PLACE",

                # 继续上升到搜索高度///就算不是在platform上面，也可以这样做，先上升，然后进入fly to home
                "ASCEND_AFTER_PLACE": "ASCEND_AFTER_PLACE",

                # 从当前位置继续返航
                "FLY_TO_HOME": "FLY_TO_HOME",

                "WAIT_GEAR_DOWN": "FLY_TO_HOME",

                # 不能直接从TASK4位置下降，先重新飞到Home上方
                "DESCEND": "FLY_TO_HOME",

                # 已经发送LAND命令，不能安全执行TASK4///不考虑
                #"LAND": None,

                # PX4正在降落或可能已经解除武装，不能执行TASK4///不考虑
                #"WAIT_FOR_LAND": None,

                # TASK3已经完成并落地，不能执行TASK4///不考虑
                #"FINISHED": None,
            },
        }

        # TASK3、TASK4 暂时只缓存最近事件，后续添加具体数据。
        self.task3_last_event: str | None = None
        self.task4_last_event: str | None = None

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ==============================================================
        # 4. ROS 订阅者
        # ==============================================================
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus,
            str(
                topics.get(
                    "vehicle_status",
                    "/fmu/out/vehicle_status_v1",
                )
            ),
            self._on_vehicle_status,
            px4_qos,
        )

        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            str(
                topics.get(
                    "vehicle_attitude",
                    "/fmu/out/vehicle_attitude",
                )
            ),
            self._on_vehicle_attitude,
            px4_qos,
        )

        self.battery_status_sub = self.create_subscription(
            BatteryStatus,
            str(
                topics.get(
                    "battery_status",
                    "/fmu/out/battery_status",
                )
            ),
            self._on_battery_status,
            px4_qos,
        )
        
        self.task_command_sub = self.create_subscription(
            String,
            str(topics.get("task_command", "/tongji_mrl/tsky/task_cmd")),
            self._on_task_command,
            10,
        )

        self.course_config_sub = self.create_subscription(
            String,
            str(
                topics.get(
                    "course_config",
                    "/tongji_mrl/tsky/course_config",
                )
            ),
            self._on_course_config,
            10,
        )

        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            str(
                topics.get(
                    "vehicle_local_position",
                    "/fmu/out/vehicle_local_position_v1",
                )
            ),
            self._on_local_position,
            px4_qos,
        )
        self.task1_buoy_sub = self.create_subscription(
            String,
            str(topics.get("task1_buoy", "/tongji_mrl/twave/task1_buoy")),
            self._on_task1_buoy,
            10,
        )
        self.task1_reply_sub = self.create_subscription(
            String,
            str(topics.get("task1_reply", "/tongji_mrl/twave/task1")),
            self._on_task1_reply,
            10,
        )
        self.task1_position_sub = self.create_subscription(
            String,
            str(
                topics.get(
                    "task1_position",
                    "/tongji_mrl/twave/task1_position",
                )
            ),
            self._on_task1_position,
            10,
        )
        self.task3_event_sub = self.create_subscription(
            String,
            str(topics.get("task3_event", "/tongji_mrl/twave/task3")),
            self._on_task3_event,
            10,
        )

        self.task3_platform_sub = self.create_subscription(
            String,
            str(
                topics.get(
                    "task3_platform",
                    "/tongji_mrl/twave/task3_platform",
                )
            ),
            self._on_task3_platform,
            10,
        )

        self.task3_vision_sub = self.create_subscription(
            String,
            str(
                topics.get(
                    "task3_vision_result",
                    "/tongji_mrl/tsky/task3_target_gripper",
                )
            ),
            self._on_task3_vision,
            10,
        )

        self.task4_event_sub = self.create_subscription(
            String,
            str(topics.get("task4_event", "/tongji_mrl/twave/task4")),
            self._on_task4_event,
            10,
        )

        self.task3_home_xy_tolerance_m = float(
            task3_cfg.get("home_xy_tolerance_m", 0.20)
        )
        self.task3_descend_speed_m_s = float(
            task3_cfg.get("descend_speed_m_s", 0.20)
        )

        self.height_range_sub = self.create_subscription(
            Range,
            str(
                topics.get(
                    "height_sensor",
                    "/height_sensor/range",
                )
            ),
            self._on_height_range,
            qos_profile_sensor_data,
        )

        # ==============================================================
        # 5. ROS 发布者：整份代码只有这里创建 PX4 控制发布器
        # ==============================================================
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode,
            str(
                topics.get(
                    "offboard_control_mode",
                    "/fmu/in/offboard_control_mode",
                )
            ),
            px4_qos,
        )
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            str(
                topics.get(
                    "trajectory_setpoint",
                    "/fmu/in/trajectory_setpoint",
                )
            ),
            px4_qos,
        )
        self.progress_pub = self.create_publisher(
            String,
            str(topics.get("progress", "/tongji_mrl/tsky/progress")),
            10,
        )
        self.task1_notify_pub = self.create_publisher(
            String,
            str(topics.get("task1_notify", "/tongji_mrl/tsky/task1")),
            10,
        )

        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand,
            str(
                topics.get(
                    "vehicle_command",
                    "/fmu/in/vehicle_command",
                )
            ),
            px4_qos,
        )

        # 视觉控制发布器
        self.vision_control_pub = self.create_publisher(
            String,
            str(
                topics.get(
                    "vision_control",
                    "/tongji_mrl/tsky/downward_vision_control",
                )
            ),
            10,
        )
        
        # 控制夹爪的发布器
        self.gripper_command_pub = self.create_publisher(
            String,
            str(
                topics.get(
                    "gripper_command",
                    "/task3/gripper/command",
                )
            ),
            10,
        )

        # 控制起落架的发布器
        self.landing_gear_command_pub = self.create_publisher(
            String,
            str(
                topics.get(
                    "landing_gear_command",
                    "/landing_gear/command",
                )
            ),
            10,
        )

        self.checkpoint_pub = self.create_publisher(
            String,
            str(
                topics.get(
                    "checkpoint",
                    "/tongji_mrl/tsky/checkpoint",
                )
            ),
            10,
        )

        # ==============================================================
        # 6. 唯一的状态机定时器
        # ==============================================================
        self.update_timer = self.create_timer(
            1.0 / self.update_rate_hz,
            self._tick,
        )
        self.progress_timer = self.create_timer(
            1.0 / self.progress_rate_hz,
            self._publish_progress,
        )

        self.get_logger().info("UAV mission manager is waiting for MAP begin")

    # ==================================================================
    # 配置
    # ==================================================================
    def _load_config(self) -> dict[str, Any]:
        config_file = str(self.get_parameter("config_file").value).strip()
        if not config_file:
            raise RuntimeError("config_file parameter is required")

        config_path = Path(config_file).expanduser()
        if not config_path.is_file():
            raise RuntimeError(f"Config file does not exist: {config_path}")

        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Mission config must be a mapping")

        config = loaded.get("uav_mission_manager", loaded)
        if not isinstance(config, dict):
            raise ValueError("uav_mission_manager config must be a mapping")
        return config

    # ==================================================================
    # 所有订阅回调：只解析并缓存数据，不直接控制飞行
    # ==================================================================
    def _on_vehicle_status(self, msg: VehicleStatus) -> None:
        # -------------
        new_nav_state = int(msg.nav_state)

        if new_nav_state != self.vehicle_nav_state:
            self.get_logger().warning(
                f"PX4 nav_state changed: "
                f"{self.vehicle_nav_state} -> {new_nav_state}; "
                f"position={self.local_position}"
            )

        self.vehicle_nav_state = new_nav_state
        #-----------------
        if self.vehicle_in_offboard:
            self.offboard_was_active = True

        self.vehicle_armed = (
            msg.arming_state
            == VehicleStatus.ARMING_STATE_ARMED
        )

        self.vehicle_in_offboard = (
            msg.nav_state
            == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        )

        self.vehicle_in_position = (
            msg.nav_state
            == VehicleStatus.NAVIGATION_STATE_POSCTL
        )

        self.safety_monitor.update_vehicle_status(
            msg,
            time.monotonic(),
        )

    def _on_vehicle_attitude(
        self,
        msg: VehicleAttitude,
    ) -> None:
        self.vehicle_attitude_q = [
            float(msg.q[0]),
            float(msg.q[1]),
            float(msg.q[2]),
            float(msg.q[3]),
        ]

        w, x, y, z = self.vehicle_attitude_q

        self.current_yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

        self.safety_monitor.update_attitude(
            msg,
            time.monotonic(),
        )

    def _on_battery_status(
        self,
        msg: BatteryStatus,
    ) -> None:
        self.safety_monitor.update_battery_status(
            msg,
            time.monotonic(),
        )
    
    def _on_task_command(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(
                f"Ignored invalid task command JSON: {exc}"
            )
            return

        if not isinstance(payload, dict):
            return

        # 这份节点只处理发给UAV的消息
        vehicle = str(
            payload.get("vehicle", "")
        ).strip().lower()

        if vehicle != "uav":
            return

        task = str(
            payload.get("task", "")
        ).strip().upper()

        raw_command = payload.get("command")

        # TASK4恢复命令
        if (
            isinstance(raw_command, str)
            and raw_command.strip().lower() == "resume"
        ):
            self._handle_task4_resume(
                task=task,
                requested_state=payload.get("progress"),
            )
            return

        # ==========================================================
        # TASK4命令
        # ==========================================================
        if task == "TASK4" and isinstance(raw_command, dict):
            command_type = str(
                raw_command.get("type", "")
            ).strip().upper()

            if command_type != "TASK4_BEGIN":
                return

            command_seq = int(
                raw_command.get("command_seq", 0)
            )

            # 防止同一个TASK4命令被重复处理
            if (
                self.task4_command_seq == command_seq
                and self.mission == "TASK4"
            ):
                return

            # 必须在切换到TASK4之前记录原任务
            self.task4_checkpoint = {
                "vehicle": "uav",
                "task": self.mission,
                "progress": self.state,
                "error": self.error,
                "command": "checkpoint",
            }

            self.task4_command_seq = command_seq

            self.task4_target_vehicle = str(
                raw_command.get(
                    "target_vehicle",
                    "",
                )
            ).strip().lower()

            self.task4_is_executor = (
                self.task4_target_vehicle == "uav"
            )

            # 只有UAV是执行者时才保存目标经纬度
            if self.task4_is_executor:
                position = raw_command["position"]

                self.task4_target_position = {
                    "latitude": float(
                        position["latitude"]
                    ),
                    "longitude": float(
                        position["longitude"]
                    ),
                }
            else:
                # 避免保留上一次TASK4的位置
                self.task4_target_position = None

            # UAV和USV谁执行都要标记TASK4开始
            self.begin_requests["TASK4"] = True
            self.finish_requests["TASK4"] = False

            # 发布切换前的任务进度
            self._publish_checkpoint()

            # 暂停原任务使用的视觉检测
            self._publish_vision_mode("OFF", [])

            # 切换总状态机到TASK4
            self._change_mission(
                "TASK4",
                "WAIT_BEGIN",
            )

            self.get_logger().info(
                f"TASK4 started: "
                f"target_vehicle={self.task4_target_vehicle}, "
                f"is_executor={self.task4_is_executor}, "
                f"position={self.task4_target_position}"
            )
            return

        # ==========================================================
        # 原来的普通任务命令
        # ==========================================================
        if task not in self.begin_requests:
            return

        if not isinstance(raw_command, str):
            return

        command = raw_command.strip().lower()

        if command == "begin":
            self.begin_requests[task] = True
            self.get_logger().info(
                f"Received {task} begin command"
            )

        elif command in {
            "finish",
            "finished",
            "end",
        }:
            self.finish_requests[task] = True
            self.get_logger().info(
                f"Received {task} finish command"
            )

    def _on_course_config(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            corners = payload["corners"]

            self.map_corners = corners
            self.map_waypoints = self.waypoint_planner.plan(corners)

            self.get_logger().info(
                f"Generated MAP waypoints: {self.map_waypoints}"
            )

        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            self.get_logger().warning(
                f"Ignored invalid course config: {exc}"
            )

    def _on_local_position(self, msg: VehicleLocalPosition) -> None:
        now = time.monotonic()
        self.local_position = [float(msg.x), float(msg.y), float(msg.z)]
        self.last_local_position_time = now

        self.safety_monitor.update_local_position(
            msg,
            now,
        )

    def _on_task1_buoy(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Ignored invalid buoy JSON: {exc}")
            return

        selected = select_flashing_blue_buoy(payload)
        if selected is not None:
            self.task1_buoy = selected
            self.get_logger().info("Cached TASK1 flashing blue buoy")

    def _on_task1_reply(self, msg: String) -> None:
        reply = msg.data.strip()
        if reply == "usv_arrive_start":
            self.task1_usv_at_start = True
        elif reply == "usv_arrive_end":
            self.task1_usv_at_end = True

    def _on_task1_position(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            position = payload["NED_POSITION"]
            self.usv_ned_position = [
                float(position[0]),
                float(position[1]),
                float(position[2]),
            ]
            self.last_usv_position_time = time.monotonic()
        except (
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            self.get_logger().warning(
                f"Ignored invalid TASK1 position message: {exc}"
            )

    def _on_task3_event(self, msg: String) -> None:
        message = json.loads(msg.data)
        payload = message["payload"]

        self.task3_resource_color = payload["resource_color"]
        self.task3_destination_color = payload["destination_color"]
        self.task3_color_sequence_received = True

        self.get_logger().info(
            f"TASK3 colors: "
            f"resource={self.task3_resource_color}, "
            f"destination={self.task3_destination_color}"
        )

    def _on_task3_platform(self, msg: String) -> None:
        message = json.loads(msg.data)
        platforms = message["objects"]

        self.task3_platform_1_position = list(
            platforms[0]["NED_position"]
        )
        self.task3_platform_2_position = list(
            platforms[1]["NED_position"]
        )

        self.task3_platforms_received = True

    def _on_task3_vision(self, msg: String) -> None:
        result = json.loads(msg.data)
        self.latest_task3_objects = result["objects"]
        self.last_task3_vision_time = time.monotonic()

    def _on_task4_event(self, msg: String) -> None:
        self.task4_last_event = msg.data

    def _on_height_range(self, msg: Range) -> None:
        distance = float(msg.range)

        if not math.isfinite(distance) or distance <= 0.0:
            return

        self.latest_height_range_m = distance
        self.last_height_range_time = time.monotonic()

    # ==================================================================
    # 总状态机：只负责选择当前任务
    # ==================================================================
    def _tick(self) -> None:

        # if (
        #     self.control_started
        #     and self.mission not in {"ERROR", "FINISHED"}
        # ):
        #     safety = self.safety_monitor.evaluate(
        #         time.monotonic()
        #     )

        #     if not safety.safe:
        #         self._publish_velocity(0.0, 0.0, 0.0)
        #         self._set_error(safety.reason)
        #         return

        if (
            self.control_started
            and self.offboard_was_active
            and not self.vehicle_in_offboard
            and self.mission not in {"ERROR", "FINISHED"}
            and self.state not in {
                "LAND",
                "WAIT_FOR_LAND",
                "FINISHED",
            }
        ):
            self.error = "Pilot left Offboard mode"
            self.error_handover_complete = True

            self._publish_vision_mode("OFF", [])
            self._change_mission("ERROR", "ERROR")
            return

        if self.mission == "MAP":
            self._run_map_state_machine()
        elif self.mission == "TASK1":
            self._run_task1_state_machine()
        elif self.mission == "TASK3":
            self._run_task3_state_machine()
        elif self.mission == "TASK4":
            self._run_task4_state_machine()
        elif self.mission == "FINISHED":
            # 已落地并Disarm，不再发送任何飞控指令
            pass
        elif self.mission == "ERROR":
            self._run_error_handover()
        else:
            self._set_error(f"Unknown mission: {self.mission}")

    # ==================================================================
    # MAP 状态机
    # ==================================================================
    def _run_map_state_machine(self) -> None:
        if self.state == "WAIT_BEGIN":
            self._map_wait_begin()
        elif self.state == "RECORD_HOME_COORDINATE":
            self._map_record_home_coordinate()
        elif self.state == "PREPARE_TAKEOFF":
            self._map_prepare_takeoff()
        elif self.state == "ASCEND":
            self._map_ascend()
        elif self.state == "WAIT_GEAR_UP":
            self._map_wait_gear_up()
        elif self.state == "FOLLOW_ROUTE":
            self._map_follow_route()
        elif self.state == "MAP_FINISH":
            self._map_finished()
        else:
            self._set_error(f"Unknown MAP state: {self.state}")

    def _map_wait_begin(self) -> None:
        self._hold_if_control_started()
        if self.begin_requests["MAP"]:
            self.control_started = True
            self._change_state("RECORD_HOME_COORDINATE")

    def _map_record_home_coordinate(self) -> None:
        # 等待有效的本地NED位置
        if not self._local_position_is_fresh(
            time.monotonic()
        ):
            return

        # 等待有效的姿态数据
        if self.current_yaw is None:
            return

        # 记录起点NED位置
        self.home_ned = list(self.local_position)

        # 整个任务只记录一次固定偏航角
        if self.fixed_mission_yaw is None:
            self.fixed_mission_yaw = self.current_yaw

        self.get_logger().info(
            f"Recorded home NED: {self.home_ned}, "
            f"fixed yaw: "
            f"{math.degrees(self.fixed_mission_yaw):.1f} deg"
        )

        self._change_state("PREPARE_TAKEOFF")

    def _map_prepare_takeoff(self) -> None:
        self._publish_velocity(0.0, 0.0, 0.0)

        if not self.vehicle_armed:
            return

        if not self.vehicle_in_offboard:
            self._request_offboard()
            return

        self._change_state("ASCEND")

    def _map_ascend(self) -> None:
        if (
            self.home_ned is None
            or not self._local_position_is_fresh(time.monotonic())
        ):
            return

        # NED中向上是负方向
        target_z = self.home_ned[2] - self.search_height_m
        current_z = self.local_position[2]

        # 已经到达搜索高度
        if abs(current_z - target_z) <= self.z_tolerance_m:
            self._publish_position_hold()

            if not self.map_waypoints:
                self._set_error("MAP waypoints are unavailable")
                return

            self.map_route = self.waypoint_order_planner.order(
                self.local_position,
                self.map_waypoints,
            )
            self.map_route_index = 0

            self.get_logger().info(
                f"Ordered MAP route: {self.map_route}"
            )

            # 同时发送起落架收起和夹爪打开命令
            self._publish_landing_gear_command("up")
            self._publish_gripper_command("open")
            # 开始5秒等待
            self.landing_gear_wait_start_time = time.monotonic()
            self._change_state("WAIT_GEAR_UP")
            return
            
        # 还未到达搜索高度，继续上升
        target_position = [
            self.home_ned[0],
            self.home_ned[1],
            target_z,
        ]

        self._publish_interpolated_position(
            target_position,
            max_speed_m_s=self.ascend_speed_m_s,
        )

    def _map_wait_gear_up(self) -> None:
        # 等待期间保持悬停并持续发送Offboard消息
        self._publish_position_hold()

        if self.landing_gear_wait_start_time is None:
            self.landing_gear_wait_start_time = time.monotonic()
            return

        elapsed = (
            time.monotonic()
            - self.landing_gear_wait_start_time
        )

        if elapsed < self.landing_gear_wait_s:
            return

        self.landing_gear_wait_start_time = None
        self._change_state("FOLLOW_ROUTE")

    def _map_follow_route(self) -> None:
        self._publish_vision_mode("MAP", ["all"])
        if not self._local_position_is_fresh(time.monotonic()):
            self._publish_position_hold()
            self._set_error("PX4 local position is unavailable or stale")
            return

        if self.map_route_index >= len(self.map_route):
            self._publish_position_hold()
            self._change_state("MAP_FINISH")
            return

        target = self.map_route[self.map_route_index]

        dx = target[0] - self.local_position[0]
        dy = target[1] - self.local_position[1]
        distance = math.hypot(dx, dy)

        # ---------
        now = time.monotonic()

        if now - self.map_debug_last_log_time >= 1.0:
            self.map_debug_last_log_time = now

            self.get_logger().info(
                f"MAP DEBUG: "
                f"index={self.map_route_index}/{len(self.map_route)}, "
                f"current_ned={self.local_position}, "
                f"target_xy={target}, "
                f"distance={distance:.3f}, "
                f"published_sp={self.last_published_position_sp}"
            )
        # ---------

        # 到达当前航点，切换到下一个航点
        if distance <= self.waypoint_tolerance_m:
            # ----------
            self.get_logger().info(
                f"MAP waypoint reached: "
                f"index={self.map_route_index}, "
                f"current_ned={self.local_position}, "
                f"target_xy={target}, "
                f"distance={distance:.3f}"
            )
            # ----------
        
            self.map_route_index += 1

            # 所有航点已经完成
            if self.map_route_index >= len(
                self.map_route
            ):
                self._publish_position_hold()
                self._change_state("MAP_FINISH")
                return

            # 立即读取下一个航点
            target = self.map_route[
                self.map_route_index
            ]

        target_position = [
            float(target[0]),
            float(target[1]),
            (
                self.home_ned[2]
                - self.search_height_m
            ),
        ]

        self._publish_interpolated_position(
            target_position,
            max_speed_m_s=self.route_speed_m_s,
        )
        
    def _map_finished(self) -> None:
        # ----------------
        if not self.map_finish_log_written:
            self.map_finish_log_written = True

            self.get_logger().warning(
                f"MAP FINISH DEBUG: "
                f"current_ned={self.local_position}, "
                f"route_index={self.map_route_index}, "
                f"last_route_target="
                f"{self.map_route[-1] if self.map_route else None}, "
                f"last_published_sp={self.last_published_position_sp}, "
                f"nav_state={self.vehicle_nav_state}, "
                f"in_offboard={self.vehicle_in_offboard}"
            )
        # ----------------

        # MAP完成后发布视觉启动结束
        self._publish_vision_mode("OFF", [])
        # MAP完成后持续保持静止
        self._publish_position_hold()

        # 等待主状态机发送TASK1 begin
        if self.begin_requests["TASK1"]:
            self._change_mission("TASK1", "WAIT_BEGIN")

    # ==================================================================
    # TASK1 状态机
    # ==================================================================
    def _run_task1_state_machine(self) -> None:
        if self.state == "WAIT_BEGIN":
            self._task1_wait_begin()
        elif self.state == "RECEIVE_TARGET":
            self._task1_receive_target()
        elif self.state == "FLY":
            self._task1_fly()
        elif self.state == "HOVER":
            self._task1_hover()
        elif self.state == "FOLLOW":
            self._task1_follow()
        #elif self.state == "FLY_TO_HOME":
            #self._task1_fly_to_home()
        elif self.state == "TASK1_FINISH":
            self._task1_finished()
        else:
            self._set_error(f"Unknown TASK1 state: {self.state}")

    def _task1_wait_begin(self) -> None:
        self._hold_if_control_started()
        if self.begin_requests["TASK1"]:
            self.task1_begin_time = time.monotonic()
            self._change_state("RECEIVE_TARGET")

    def _task1_receive_target(self) -> None:
        self._publish_position_hold()
        now = time.monotonic()
        elapsed = (
            now - self.task1_begin_time
            if self.task1_begin_time is not None
            else 0.0
        )

        if self.task1_buoy is None:
            if elapsed >= self.receive_timeout_s:
                self._set_error(
                    "No flashing blue buoy received within "
                    f"{self.receive_timeout_s:g} seconds"
                )
            return

        if not self._local_position_is_fresh(now):
            if elapsed >= self.local_position_timeout_s:
                self._set_error("PX4 local position is unavailable or stale")
            return

        position = self.task1_buoy["NED_position"]
        self.task1_target = [
            float(position[0]),
            float(position[1]),
            self.home_ned[2] - self.search_height_m,
        ]
        self._change_state("FLY")

    def _task1_fly(self) -> None:
        if not self._local_position_is_fresh(time.monotonic()):
            self._publish_velocity(0.0, 0.0, 0.0)
            self._set_error("PX4 local position is unavailable or stale")
            return

        if self._task1_has_arrived():
            self._publish_position_hold()
            self._change_state("HOVER")
            return

        self._publish_interpolated_position(
            self.task1_target,
            max_speed_m_s=self.fly_speed_m_s,
        )

    def _task1_hover(self) -> None:
        self._publish_position_hold()

        msg = String()
        msg.data = "uav_arrive_task1_start"
        self.task1_notify_pub.publish(msg)

        if self.task1_usv_at_start:
            self._change_state("FOLLOW")

    def _task1_follow(self) -> None:
        now = time.monotonic()

        if not self._local_position_is_fresh(now):
            self._publish_velocity(0.0, 0.0, 0.0)
            self._set_error(
                "PX4 local position is unavailable or stale"
            )
            return

        # USV已经到达终点
        if self.task1_usv_at_end:
            self._publish_position_hold()
            self._change_state("TASK1_FINISH")
            return

        # USV位置尚未收到或者已经过期
        if (
            self.usv_ned_position is None
            or self.last_usv_position_time is None
            or now - self.last_usv_position_time
            > self.follow_position_timeout_s
        ):
            self._publish_position_hold()
            return

        if self.home_ned is None:
            self._set_error(
                "Home NED is unavailable during TASK1 follow"
            )
            return

        # 跟随USV的XY，Z保持搜索高度
        target_position = [
            float(self.usv_ned_position[0]),
            float(self.usv_ned_position[1]),
            (
                self.home_ned[2]
                - self.search_height_m
            ),
        ]

        self._publish_interpolated_position(
            target_position,
            max_speed_m_s=self.follow_speed_m_s,
        )

    # 目前这里虽然有fly to home的机制，但是代码写的还是保持原地
    #def _task1_fly_to_home(self) -> None:
        # TODO: 后续在这里加入返航控制；当前保持零速度。
        #self._publish_velocity(0.0, 0.0, 0.0)
        #if self.finish_requests["TASK1"]:
            #self._change_state("TASK1_FINISH")

    def _task1_finished(self) -> None:
        # MAP完成后持续保持静止
        self._publish_position_hold()

        # 等待主状态机发送TASK3 begin
        if self.begin_requests["TASK3"]:
            self._change_mission("TASK3", "WAIT_BEGIN")

    # ==================================================================
    # TASK3 状态机骨架
    # ==================================================================
    def _run_task3_state_machine(self) -> None:
        if self.state == "WAIT_BEGIN":
            self._task3_wait_begin()
        elif self.state == "FLY_TO_PLATFORM":
            self._task3_fly_to_platform()
        elif self.state == "MOVE_TO_PLATFORM":
            self._task3_move_to_platform()
        elif self.state == "DETECT_TIN":
            self._task3_detect_tin()
        elif self.state == "CHECK_PLATFORM":
            self._task3_check_platform()
        elif self.state == "ALIGN_TIN_XY":
            self._task3_align_tin_xy()
        elif self.state == "DESCEND_TO_TIN":
            self._task3_descend_to_tin()
        elif self.state == "RECOVER_ASCEND":
            self._task3_recover_ascend()
        elif self.state == "PICK":
            self._pick()
        elif self.state == "WAIT_GRIPPER_CLOSE":
            self._wait_gripper() 
        elif self.state == "RETURN_SEARCH_HEIGHT":
            self._task3_return_search_height()           
        elif self.state == "VERIFY":
            self._verify() 
        elif self.state == "FLY_TO_OTHER_PLATFORM":
            self._task3_fly_to_other_platform()
        elif self.state == "PLACE_RESOURCE":
            self._task3_place_resource()
        elif self.state == "MOVE_TO_PLACE_POSITION":
            self._task3_move_to_place_position()
        elif self.state == "PLACE":
            self._place()
        elif self.state == "WAIT_FOR_PLACE":
            self._task3_wait_for_place()
        elif self.state == "ASCEND_AFTER_PLACE":
            self._task3_ascend_after_place()
        elif self.state == "FLY_TO_HOME":
            self._task3_fly_to_home()
        elif self.state == "WAIT_GEAR_DOWN":
            self._task3_wait_gear_down()
        elif self.state == "DESCEND":
            self._task3_descend()
        elif self.state == "LAND":
            self._task3_land()
        elif self.state == "WAIT_FOR_LAND":
            self._task3_wait_for_land()
        elif self.state == "FINISHED":
            self._task3_finished()
        else:
            self._set_error(f"Unknown TASK3 state: {self.state}")

    def _task3_wait_begin(self) -> None:
        self._publish_position_hold()

        if (
            self.task3_color_sequence_received
            and self.task3_platforms_received
        ):
            self._change_state("FLY_TO_PLATFORM")

    def _task3_fly_to_platform(self) -> None:
        now = time.monotonic()

        if not self._local_position_is_fresh(now):
            self._publish_velocity(0.0, 0.0, 0.0)
            self._set_error(
                "PX4 local position is unavailable or stale"
            )
            return

        if self.home_ned is None:
            self._set_error(
                "Home NED is unavailable during TASK3"
            )
            return

        if (
            self.task3_platform_1_position is None
            or self.task3_platform_2_position is None
        ):
            self._publish_position_hold()
            return

        # 第一次进入时，选择距离最近的平台
        if self.task3_target_platform is None:
            distance_1 = math.hypot(
                self.task3_platform_1_position[0]
                - self.local_position[0],
                self.task3_platform_1_position[1]
                - self.local_position[1],
            )

            distance_2 = math.hypot(
                self.task3_platform_2_position[0]
                - self.local_position[0],
                self.task3_platform_2_position[1]
                - self.local_position[1],
            )

            if distance_1 <= distance_2:
                nearest = self.task3_platform_1_position
                other = self.task3_platform_2_position
            else:
                nearest = self.task3_platform_2_position
                other = self.task3_platform_1_position

            self.task3_original_platform_position = list(
                nearest
            )
            self.task3_other_platform_position = list(
                other
            )

            # 固定为搜索高度，不使用不断变化的当前Z
            self.task3_target_platform = [
                float(nearest[0]),
                float(nearest[1]),
                (
                    self.home_ned[2]
                    - self.search_height_m
                ),
            ]

            self.task3_target_platform_role = "ORIGINAL"

            self.get_logger().info(
                f"Selected original platform: "
                f"{self.task3_original_platform_position}; "
                f"other platform: "
                f"{self.task3_other_platform_position}"
            )

        dx = (
            self.task3_target_platform[0]
            - self.local_position[0]
        )
        dy = (
            self.task3_target_platform[1]
            - self.local_position[1]
        )

        distance = math.hypot(dx, dy)

        # 到达平台的大致位置
        if distance <= self.task3_platform_tolerance_m:
            self._publish_position_hold()

            self.task3_current_platform_role = (
                self.task3_target_platform_role
            )

            self._change_state("MOVE_TO_PLATFORM")
            return

        # 插值位置控制
        self._publish_interpolated_position(
            self.task3_target_platform,
            max_speed_m_s=self.task3_fly_speed_m_s,
        )

    def _task3_move_to_platform(self) -> None:
        self._publish_vision_mode(
            "TASK3",
            ["platform"],
        )

        now = time.monotonic()

        if not self._local_position_is_fresh(now):
            self._publish_velocity(0.0, 0.0, 0.0)
            self._set_error(
                "PX4 local position is unavailable or stale"
            )
            return

        if (
            self.home_ned is None
            or self.task3_target_platform is None
        ):
            self._set_error(
                "TASK3 platform target is unavailable"
            )
            return

        # 没有新鲜视觉结果
        if (
            not self.latest_task3_objects
            or self.last_task3_vision_time is None
            or now - self.last_task3_vision_time
            > self.task3_vision_timeout_s
        ):
            self._publish_position_hold()
            return

        platforms = [
            item
            for item in self.latest_task3_objects
            if str(item.get("type", "")).lower()
            == "platform"
            and "gripper_position" in item
        ]

        if not platforms:
            self._publish_position_hold()
            return

        # 选择距离夹爪中心最近的平台
        target = min(
            platforms,
            key=lambda item: math.hypot(
                float(item["gripper_position"][0]),
                float(item["gripper_position"][1]),
            ),
        )

        gripper_position = target["gripper_position"]

        gripper_x = float(gripper_position[0])
        gripper_y = float(gripper_position[1])

        xy_error = math.hypot(
            gripper_x,
            gripper_y,
        )

        # 已经与平台中心对齐
        if (
            xy_error
            <= self.task3_gripper_xy_tolerance_m
        ):
            self._publish_position_hold()

            self.task3_platform_center_ned = [
                self.local_position[0],
                self.local_position[1],
                (
                    self.home_ned[2]
                    - self.search_height_m
                ),
            ]

            self.get_logger().info(
                f"Recorded TASK3 platform center "
                f"at search height: "
                f"{self.task3_platform_center_ned}"
            )

            self.latest_task3_objects = []
            self.last_task3_vision_time = None

            if (
                self.task3_after_alignment_state
                == "DETECT_TIN"
            ):
                self.task3_detect_tin_start_time = (
                    time.monotonic()
                )

            self._change_state(
                self.task3_after_alignment_state
            )
            return

        try:
            target_position = (
                self._visual_xy_target_ned(
                    target,
                    self.task3_target_platform[2],
                )
            )
        except (ValueError, TypeError, KeyError) as exc:
            self._publish_position_hold()
            self.get_logger().warning(
                f"Invalid platform visual target: {exc}"
            )
            return

        self._publish_interpolated_position(
            target_position,
            max_speed_m_s=(
                self.task3_visual_move_speed_m_s
            ),
        )

    def _task3_detect_tin(self) -> None:
        # 保持无人机静止
        self._publish_position_hold()

        # 开启TIN检测
        self._publish_vision_mode(
            "TASK3",
            ["tin"],
        )

        now = time.monotonic()

        # 持续保存最新一帧中的TIN信息
        tins: list[dict[str, Any]] = []

        for item in self.latest_task3_objects:
            if str(item.get("type", "")).lower() != "tin":
                continue

            position = item["gripper_position"]

            tins.append(
                {
                    "id": int(item["id"]),
                    "color": str(item["color"]).lower(),
                    "gripper_position": [
                        float(position[0]),
                        float(position[1]),
                        float(position[2]),
                    ],
                }
            )

        # 有检测结果时更新，避免偶尔漏检把结果清空
        if tins:
            self.task3_tin_objects = tins
            self.task3_tin_count = len(tins)

        # 防止计时变量意外没有设置
        if self.task3_detect_tin_start_time is None:
            self.task3_detect_tin_start_time = now

        # 在platform上停留并检测2秒
        if now - self.task3_detect_tin_start_time < 2.0:
            return

        self.get_logger().info(
            f"TIN detection finished: "
            f"{self.task3_tin_count} tin(s), "
            f"{self.task3_tin_objects}"
        )

        # 停止当前视觉检测
        self._publish_vision_mode("OFF", [])

        # 换成你接下来实际需要的状态名称
        self._change_state("CHECK_PLATFORM")

    def _task3_check_platform(self) -> None:
        self._publish_position_hold()

        resource_color = str(
            self.task3_resource_color
        ).strip().lower()

        matching_tins = [
            tin
            for tin in self.task3_tin_objects
            if str(tin.get("color", "")).strip().lower()
            == resource_color
        ]

        # 情况1：当前平台存在目标TIN
        if matching_tins:
            self.task3_selected_resource_tin = min(
                matching_tins,
                key=lambda tin: math.hypot(
                    float(tin["gripper_position"][0]),
                    float(tin["gripper_position"][1]),
                ),
            )

            self.task3_align_stable_start_time = None

            self.get_logger().info(
                f"Found resource tin: "
                f"{self.task3_selected_resource_tin}"
            )

            self._change_state("ALIGN_TIN_XY")
            return

        # 情况2：原平台没有，前往另一个平台
        if self.task3_current_platform_role == "ORIGINAL":
            if (
                self.task3_other_platform_position is None
                or self.local_position is None
            ):
                self._set_error(
                    "Other TASK3 platform is unavailable"
                )
                return

            self.task3_target_platform = [
                self.task3_other_platform_position[0],
                self.task3_other_platform_position[1],
                self.home_ned[2] - self.search_height_m,
            ]
            self.task3_target_platform_role = "OTHER"
            self.task3_after_alignment_state = "DETECT_TIN"

            self.task3_tin_objects = []
            self.task3_tin_count = 0
            self.latest_task3_objects = []
            self.last_task3_vision_time = None
            self.task3_detect_tin_start_time = None
            self.task3_selected_resource_tin = None

            self._change_state("FLY_TO_PLATFORM")
            return

        # 情况3：另一个平台也没有目标TIN
        if self.task3_current_platform_role == "OTHER":
            self._set_error(
                "Resource tin was not found on either TASK3 platform"
            )
            return

        # 平台身份异常
        self._set_error(
            "Current TASK3 platform role is unavailable"
        )
    
    def _task3_align_tin_xy(self) -> None:
        self._publish_vision_mode(
            "TASK3",
            ["tin"],
        )

        now = time.monotonic()
        tin = self._get_nearest_resource_tin(now)

        if tin is None:
            self._start_pick_recovery(
                "Resource tin lost during XY alignment"
            )
            return

        if not self._local_position_is_fresh(now):
            self._start_pick_recovery(
                "Local position unavailable during XY alignment"
            )
            return

        if self.vehicle_attitude_q is None:
            self._publish_position_hold()
            return

        if self.task3_platform_center_ned is None:
            self._set_error(
                "TASK3 platform center is unavailable"
            )
            return

        # 每帧重新选择距离最近、颜色正确的TIN
        self.task3_selected_resource_tin = tin

        position = tin["gripper_position"]

        xy_error = math.hypot(
            float(position[0]),
            float(position[1]),
        )

        target_z = self.task3_platform_center_ned[2]

        try:
            # 视觉相对坐标转换成绝对NED目标
            target_position = (
                self._visual_xy_target_ned(
                    tin,
                    target_z,
                )
            )
        except (
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            self._start_pick_recovery(str(exc))
            return

        # 使用位置setpoint插值向TIN中心移动
        self._publish_interpolated_position(
            target_position,
            max_speed_m_s=(
                self.task3_visual_move_speed_m_s
            ),
        )

        # 已经进入小容差，开始稳定计时
        if (
            xy_error
            <= self.task3_xy_small_tolerance_m
        ):
            if self.task3_align_stable_start_time is None:
                self.task3_align_stable_start_time = now
                return

            if (
                now - self.task3_align_stable_start_time
                >= self.task3_align_stable_duration_s
            ):
                self._change_state(
                    "DESCEND_TO_TIN"
                )

            return

        # 离开小容差，重新开始稳定计时
        self.task3_align_stable_start_time = None

    def _task3_descend_to_tin(self) -> None:
        self._publish_vision_mode("TASK3", ["tin"])

        now = time.monotonic()
        tin = self._get_nearest_resource_tin(now)

        if tin is None:
            self._start_pick_recovery(
                "Resource tin lost during descent"
            )
            return

        if self.vehicle_attitude_q is None:
            self._publish_position_hold()
            return

        self.task3_selected_resource_tin = tin

        try:
            position = tin["gripper_position"]

            xy_error = math.hypot(
                float(position[0]),
                float(position[1]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            self._start_pick_recovery(str(exc))
            return

        # 超过大容差：恢复
        if xy_error > self.task3_xy_large_tolerance_m:
            self._start_pick_recovery(
                f"TIN XY error too large: {xy_error:.3f} m"
            )
            return

        # 大小容差之间：暂停下降，只纠正XY
        if xy_error > self.task3_xy_small_tolerance_m:
            self.vertical_move_xy = None

            if self.task3_descent_hold_z is None:
                self.task3_descent_hold_z = (
                    self.local_position[2]
                )

            try:
                target_position = (
                    self._visual_xy_target_ned(
                        tin,
                        self.task3_descent_hold_z,
                    )
                )
            except ValueError as exc:
                self._start_pick_recovery(str(exc))
                return

            self._publish_interpolated_position(
                target_position,
                max_speed_m_s=(
                    self.task3_visual_move_speed_m_s
                ),
            )
            return

        # 小容差内：继续垂直下降
        self.task3_descent_hold_z = None

        if (
            self.home_ned is None
            or not self._local_position_is_fresh(now)
        ):
            self._start_pick_recovery(
                "Local position height unavailable"
            )
            return

        target_z = self.home_ned[2] - 0.2

        if abs(target_z - self.local_position[2]) <= 0.02:
            self._publish_position_hold()
            self._publish_vision_mode("OFF", [])
            self._change_state("PICK")
            return

        target_position = self._vertical_position_target(
            target_z
        )

        self._publish_interpolated_position(
            target_position,
            max_speed_m_s=(
                self.task3_descend_to_tin_speed_m_s
            ),
        )

    def _task3_recover_ascend(self) -> None:
        now = time.monotonic()

        if not self._local_position_is_fresh(now):
            self._publish_velocity(0.0, 0.0, 0.0)
            self._set_error(
                "PX4 local position unavailable during recovery"
            )
            return

        if self.task3_platform_center_ned is None:
            self._set_error(
                "TASK3 platform center unavailable"
            )
            return

        target_z = self.task3_platform_center_ned[2]
        dz = target_z - self.local_position[2]

        if abs(dz) <= self.z_tolerance_m:
            self._publish_position_hold()

            self.latest_task3_objects = []
            self.last_task3_vision_time = None
            self.task3_tin_objects = []
            self.task3_tin_count = 0
            self.task3_detect_tin_start_time = None
            self.task3_after_alignment_state = "DETECT_TIN"

            self._change_state("MOVE_TO_PLATFORM")
            return

        target_position = self._vertical_position_target(
            target_z
        )

        self._publish_interpolated_position(
            target_position,
            max_speed_m_s=self.task3_return_speed_m_s,
        )

    def _pick(self) -> None:
        self._publish_position_hold()
        # 已成功完成对准和下降，清零recover计数
        self.task3_recover_retry_count = 0
        # 发布一次夹爪闭合命令
        self._publish_gripper_command("close")
        self.task3_gripper_close_start_time = (
            time.monotonic()
        )
        self._change_state("WAIT_GRIPPER_CLOSE")

    def _wait_gripper(self) -> None:
        self._publish_position_hold()

        elapsed = (
            time.monotonic()
            - self.task3_gripper_close_start_time
        )

        if elapsed >= self.task3_gripper_close_wait_s:
            self._change_state("RETURN_SEARCH_HEIGHT")

    def _task3_return_search_height(self) -> None:
        dx = (
            self.task3_platform_center_ned[0]
            - self.local_position[0]
        )
        dy = (
            self.task3_platform_center_ned[1]
            - self.local_position[1]
        )
        dz = (
            self.task3_platform_center_ned[2]
            - self.local_position[2]
        )

        distance = math.sqrt(
            dx * dx + dy * dy + dz * dz
        )

        if distance <= self.task3_return_tolerance_m:
            self._publish_position_hold()
            # 清除抓取前遗留的视觉结果
            self.latest_task3_objects = []
            self.last_task3_vision_time = None

            self.task3_tin_count_after_pick = 0
            self.task3_verify_start_time = time.monotonic()

            self._change_state("VERIFY")
            return

        self._publish_interpolated_position(
            self.task3_platform_center_ned,
            max_speed_m_s=self.task3_return_speed_m_s,
        )

    def _verify(self) -> None:
        self._publish_position_hold()

        # 开启TIN视觉检测
        self._publish_vision_mode(
            "TASK3",
            ["tin"],
        )

        now = time.monotonic()

        if self.task3_verify_start_time is None:
            self.task3_verify_start_time = now

        # 等待VERIFY开始之后的新视觉结果
        if (
            self.last_task3_vision_time is None
            or self.last_task3_vision_time
            < self.task3_verify_start_time
            or now - self.last_task3_vision_time
            > self.task3_vision_timeout_s
        ):
            return

        tins_after_pick = [
            item
            for item in self.latest_task3_objects
            if str(item.get("type", "")).lower()
            == "tin"
        ]

        # 即使数量是0也要记录
        self.task3_tin_count_after_pick = len(
            tins_after_pick
        )

        # 检测指定时间
        if (
            now - self.task3_verify_start_time
            < self.task3_verify_duration_s
        ):
            return

        self._publish_vision_mode("OFF", [])

        expected_count = max(
            self.task3_tin_count - 1,
            0,
        )

        success = (
            self.task3_tin_count_after_pick
            == expected_count
        )

        self.get_logger().info(
            f"TASK3 verify: "
            f"before={self.task3_tin_count}, "
            f"after={self.task3_tin_count_after_pick}, "
            f"expected={expected_count}, "
            f"success={success}"
        )

        if success:
            self.get_logger().info(
                "TASK3 resource pick verification succeeded"
            )
            self.task3_verify_retry_count = 0
            self._change_state("FLY_TO_OTHER_PLATFORM")
            return

        # 本次验证失败
        if (
            self.task3_verify_retry_count
            >= self.task3_verify_max_retries
        ):
            self._set_error(
                "TASK3 resource pick failed after "
                f"{self.task3_verify_max_retries} retries"
            )
            return

        self.task3_verify_retry_count += 1

        self.get_logger().warning(
            f"TASK3 resource pick verification failed; "
            f"retry "
            f"{self.task3_verify_retry_count}/"
            f"{self.task3_verify_max_retries}"
        )

        # 重新抓取前打开夹爪
        self._publish_gripper_command("open")

        # 清除上一次检测、规划和验证结果
        self.latest_task3_objects = []
        self.last_task3_vision_time = None

        self.task3_tin_objects = []
        self.task3_tin_count = 0
        self.task3_tin_count_after_pick = 0

        self.task3_selected_resource_tin = None
        self.task3_pick_target_ned = None

        self.task3_verify_start_time = None
        self.task3_detect_tin_start_time = (
            time.monotonic()
        )

        # 回到TIN检测状态，重新检测、选择和抓取
        self._change_state("DETECT_TIN")
        return
        
    def _task3_fly_to_other_platform(self) -> None:
        now = time.monotonic()

        if not self._local_position_is_fresh(now):
            self._publish_velocity(0.0, 0.0, 0.0)
            self._set_error(
                "PX4 local position is unavailable or stale"
            )
            return

        # 当前在ORIGINAL，就前往OTHER
        if self.task3_current_platform_role == "ORIGINAL":
            destination = self.task3_other_platform_position
            destination_role = "OTHER"

        # 当前在OTHER，就返回ORIGINAL
        elif self.task3_current_platform_role == "OTHER":
            destination = (
                self.task3_original_platform_position
            )
            destination_role = "ORIGINAL"

        else:
            self._set_error(
                "Current TASK3 platform role is unavailable"
            )
            return

        if destination is None:
            self._set_error(
                "Destination TASK3 platform is unavailable"
            )
            return

        # 只移动XY，保持当前高度
        self.task3_target_platform = [
            float(destination[0]),
            float(destination[1]),
            self.home_ned[2] - self.search_height_m,
        ]
        self.task3_target_platform_role = destination_role

        dx = (
            self.task3_target_platform[0]
            - self.local_position[0]
        )
        dy = (
            self.task3_target_platform[1]
            - self.local_position[1]
        )

        distance = math.hypot(dx, dy)

        # 已经到达另一个平台附近
        if distance <= self.task3_platform_tolerance_m:
            self._publish_position_hold()

            self.task3_current_platform_role = (
                self.task3_target_platform_role
            )

            # 清除之前TIN视觉留下的数据
            self.latest_task3_objects = []
            self.last_task3_vision_time = None

            # 平台视觉对准完成后进入放置状态
            self.task3_after_alignment_state = (
                "PLACE_RESOURCE"
            )

            # 复用已有的平台视觉精确对准
            self._change_state("MOVE_TO_PLATFORM")
            return

        self._publish_interpolated_position(
            self.task3_target_platform,
            max_speed_m_s=self.task3_fly_speed_m_s,
        )

    def _task3_place_resource(self) -> None:
        self._publish_position_hold()

        # 开启circle视觉模型
        self._publish_vision_mode(
            "TASK3",
            ["circle"],
        )

        now = time.monotonic()

        # 等待新鲜的视觉结果
        if (
            self.last_task3_vision_time is None
            or now - self.last_task3_vision_time
            > self.task3_vision_timeout_s
        ):
            return

        destination_color = str(
            self.task3_destination_color
        ).strip().lower()

        circles = [
            item
            for item in self.latest_task3_objects
            if str(item.get("type", "")).strip().lower()
            == "circle"
            and str(item.get("color", "")).strip().lower()
            == destination_color
            and "gripper_position" in item
        ]

        if not circles:
            return

        # 如果存在多个同色circle，选择距离夹爪中心最近的
        target = min(
            circles,
            key=lambda item: math.hypot(
                float(item["gripper_position"][0]),
                float(item["gripper_position"][1]),
            ),
        )

        gripper_position = target["gripper_position"]

        gx = float(gripper_position[0])
        gy = float(gripper_position[1])
        gz = float(gripper_position[2])

        # 希望最终circle位于夹爪FLU的[0, 0, -0.1]
        # 所以UAV实际需要移动的在FLU相对位移为：
        offset_gripper_flu = [
            gx,
            gy,
            gz + 0.1,
        ]

        try:
            offset_ned = self.quat_rotate_gripper_flu_to_ned(
                self.vehicle_attitude_q,
                offset_gripper_flu,
            )
        except ValueError as exc:
            self._set_error(
                f"Failed to calculate place target: {exc}"
            )
            return

        # 只计算一次固定的绝对NED放置目标
        self.task3_place_target_ned = [
            self.local_position[0] + offset_ned[0],
            self.local_position[1] + offset_ned[1],
            self.local_position[2] + offset_ned[2],
        ]

        self.get_logger().info(
            f"Selected {destination_color} circle; "
            f"fixed place target NED: "
            f"{self.task3_place_target_ned}"
        )

        # 后续不持续进行视觉规划
        self._publish_vision_mode("OFF", [])
        self._change_state("MOVE_TO_PLACE_POSITION")

    def _task3_move_to_place_position(self) -> None:

        if self.task3_place_target_ned is None:
            self._publish_position_hold()
            self._set_error(
                "TASK3 place target is unavailable"
            )
            return

        dx = (
            self.task3_place_target_ned[0]
            - self.local_position[0]
        )
        dy = (
            self.task3_place_target_ned[1]
            - self.local_position[1]
        )
        dz = (
            self.task3_place_target_ned[2]
            - self.local_position[2]
        )

        distance = math.sqrt(
            dx * dx + dy * dy + dz * dz
        )

        # 可以暂时复用抓取阶段的容差
        if distance <= self.task3_pick_tolerance_m:
            self._publish_position_hold()
            self._change_state("PLACE")
            return

        self._publish_interpolated_position(
            self.task3_place_target_ned,
            max_speed_m_s=self.task3_pick_speed_m_s,
        )

    def _place(self) -> None:
        self._publish_position_hold()

        # 只发布一次夹爪打开命令
        self._publish_gripper_command("open")

        # 记录开始等待的时间
        self.task3_place_start_time = time.monotonic()

        # 立即离开PLACE，避免重复发送open
        self._change_state("WAIT_FOR_PLACE")

    def _task3_wait_for_place(self) -> None:
        self._publish_position_hold()

        if self.task3_place_start_time is None:
            self._set_error(
                "TASK3 place start time is unavailable"
            )
            return

        elapsed = (
            time.monotonic()
            - self.task3_place_start_time
        )

        if elapsed >= 2.0:
            self._change_state("ASCEND_AFTER_PLACE")

    def _task3_ascend_after_place(self) -> None:
        now = time.monotonic()

        if self.home_ned is None:
            self._publish_position_hold()
            self._set_error("Home NED is unavailable")
            return

        # Local NED中向上为负
        target_z = (
            self.home_ned[2]
            - self.search_height_m
        )

        dz = target_z - self.local_position[2]

        # 到达搜索高度
        if abs(dz) <= self.z_tolerance_m:
            self._publish_position_hold()
            self._change_state("FLY_TO_HOME")
            return

        target_position = self._vertical_position_target(
            target_z
        )

        self._publish_interpolated_position(
            target_position,
            max_speed_m_s=self.ascend_speed_m_s,
        )

    def _task3_fly_to_home(self) -> None:
        now = time.monotonic()

        if (
            self.home_ned is None
            or not self._local_position_is_fresh(now)
        ):
            self._publish_velocity(0.0, 0.0, 0.0)
            return

        dx = self.home_ned[0] - self.local_position[0]
        dy = self.home_ned[1] - self.local_position[1]
        distance = math.hypot(dx, dy)

        if distance <= self.task3_home_xy_tolerance_m:
            self._publish_position_hold()
            # 开始下降前放下起落架和关闭夹爪
            self._publish_gripper_command("close")
            self._publish_landing_gear_command("down")

            # 开始等待两个机械动作完成
            self.landing_gear_wait_start_time = time.monotonic()

            self._change_state("WAIT_GEAR_DOWN")
            return

        target_position = [
            self.home_ned[0],
            self.home_ned[1],
            self.home_ned[2] - self.search_height_m,
        ]

        self._publish_interpolated_position(
            target_position,
            max_speed_m_s=self.task3_fly_speed_m_s,
        )

    def _task3_wait_gear_down(self) -> None:
        # 等待期间保持悬停
        self._publish_position_hold()

        if self.landing_gear_wait_start_time is None:
            self.landing_gear_wait_start_time = time.monotonic()
            return

        elapsed = (
            time.monotonic()
            - self.landing_gear_wait_start_time
        )

        if elapsed < self.landing_gear_wait_s:
            return

        self.landing_gear_wait_start_time = None
        self._change_state("DESCEND")

    def _task3_descend(self) -> None:
        now = time.monotonic()

        if (
            self.home_ned is None
            or not self._local_position_is_fresh(now)
        ):
            self._publish_velocity(0.0, 0.0, 0.0)
            return

        dz = self.home_ned[2] - self.local_position[2]

        if abs(dz) <= self.z_tolerance_m:
            self._publish_position_hold()
            self._change_state("LAND")
            return

        target_position = [
            self.home_ned[0],
            self.home_ned[1],
            self.home_ned[2],
        ]

        self._publish_interpolated_position(
            target_position,
            max_speed_m_s=self.task3_descend_speed_m_s,
        )

    def _task3_land(self) -> None:
        self._request_land()
        self.get_logger().info("PX4 LAND command sent")
        self._change_state("WAIT_FOR_LAND")

    def _task3_wait_for_land(self) -> None:
        # 这里不要继续发布速度指令，让PX4执行LAND

        if self.vehicle_armed:
            return

        self.get_logger().info(
            "UAV landed and disarmed"
        )
        self._change_state("FINISHED")

    def _task3_finished(self) -> None:
        now = time.monotonic()

        # 第一次进入FINISHED时开始计时
        if self.task3_finished_start_time is None:
            self.task3_finished_start_time = now
            return

        # 保持TASK3/FINISHED状态2秒
        if now - self.task3_finished_start_time < 5.0:
            return

        self._change_mission(
            "FINISHED",
            "FINISHED",
        )

    # ==================================================================
    # TASK4 状态机骨架
    # ==================================================================
    def _run_task4_state_machine(self) -> None:
        if self.state == "WAIT_BEGIN":
            self._task4_wait_begin()

        elif self.state == "TASK4_ACK":
            self._task4_ack()

        elif self.state == "ASCEND_TO_SEARCH_HEIGHT":
            self._task4_ascend_to_search_height()

        elif self.state == "FLY_TO_POSITION_XY":
            self._task4_fly_to_position_xy()

        elif self.state == "TASK4_READY":
            self._task4_ready()

        elif self.state == "WAIT_RESUME":
            self._task4_wait_resume()

        else:
            self._set_error(
                f"Unknown TASK4 state: {self.state}"
            )

    def _task4_wait_begin(self) -> None:
        # 立即停止原任务的运动
        self._publish_position_hold()

        if not self.begin_requests["TASK4"]:
            return

        if self.task4_is_executor:
            self._change_state("TASK4_ACK")
        else:
            # UAV不是执行者，不发送ACK，只原地等待恢复
            self._change_state("WAIT_RESUME")

    def _task4_ack(self) -> None:
        self._publish_position_hold()

        now = time.monotonic()

        if self.task4_ack_start_time is None:
            self.task4_ack_start_time = now
            return

        if now - self.task4_ack_start_time < 2.0:
            return

        self._change_state(
            "ASCEND_TO_SEARCH_HEIGHT"
        )

    def _task4_ascend_to_search_height(self) -> None:
        now = time.monotonic()

        if self.home_ned is None:
            self._publish_position_hold()
            self._set_error(
                "Home NED is unavailable for TASK4"
            )
            return

        # Local NED中向上是负Z
        target_z = (
            self.home_ned[2]
            - self.search_height_m
        )

        dz = target_z - self.local_position[2]

        # 已经到达搜索高度
        if abs(dz) <= self.z_tolerance_m:
            self._publish_position_hold()
            self._change_state(
                "FLY_TO_POSITION_XY"
            )
            return

        target_position = self._vertical_position_target(
            target_z
        )

        self._publish_interpolated_position(
            target_position,
            max_speed_m_s=self.ascend_speed_m_s,
        )

    def _task4_fly_to_position_xy(self) -> None:
        now = time.monotonic()

        if not self._local_position_is_fresh(now):
            self._publish_velocity(0.0, 0.0, 0.0)
            self._set_error(
                "PX4 local position is unavailable or stale"
            )
            return

        if self.task4_target_position is None:
            self._publish_position_hold()
            self._set_error(
                "TASK4 target position is unavailable"
            )
            return

        # 测试阶段临时约定：
        # latitude代表NED x
        # longitude代表NED y
        target_x = self.task4_target_position[
            "latitude"
        ]
        target_y = self.task4_target_position[
            "longitude"
        ]

        dx = target_x - self.local_position[0]
        dy = target_y - self.local_position[1]

        distance = math.hypot(dx, dy)

        # 到达目标XY
        if distance <= self.task4_xy_tolerance_m:
            self._publish_position_hold()
            self._change_state("TASK4_READY")
            return

        target_position = [
            float(target_x),
            float(target_y),
            self.home_ned[2] - self.search_height_m,
        ]

        self._publish_interpolated_position(
            target_position,
            max_speed_m_s=self.task4_fly_speed_m_s,
        )

    def _task4_ready(self) -> None:
        # 已经到达TASK4指定位置，保持静止
        self._publish_position_hold()
        # 保持在TASK4_READY，等待主规划器发送resume

    def _task4_wait_resume(self) -> None:
        # UAV不是TASK4执行者，保持静止等待resume
        self._publish_position_hold()

    # ==============================================================
    # MAP辅助函数
    # ==============================================================
    
    # ==================================================================
    # TASK1 计算辅助函数
    # ==================================================================
    def _task1_has_arrived(self) -> bool:
        if self.task1_target is None or self.local_position is None:
            return False

        return (
            abs(self.local_position[0] - self.task1_target[0])
            <= self.x_tolerance_m
            and abs(self.local_position[1] - self.task1_target[1])
            <= self.y_tolerance_m
            and abs(self.local_position[2] - self.task1_target[2])
            <= self.z_tolerance_m
        )

    # ==================================================================
    # TASK3 计算辅助函数
    # ==================================================================
    # 每帧选择最近的同颜色TIN
    def _get_nearest_resource_tin(
        self,
        now: float,
    ) -> dict[str, Any] | None:
        if (
            self.last_task3_vision_time is None
            or now - self.last_task3_vision_time
            > self.task3_vision_timeout_s
        ):
            return None

        resource_color = str(
            self.task3_resource_color
        ).strip().lower()

        matching_tins = []

        for item in self.latest_task3_objects:
            if str(item.get("type", "")).lower() != "tin":
                continue

            if (
                str(item.get("color", "")).strip().lower()
                != resource_color
            ):
                continue

            position = item.get("gripper_position")

            if not isinstance(position, (list, tuple)):
                continue

            if len(position) < 2:
                continue

            try:
                gx = float(position[0])
                gy = float(position[1])
            except (TypeError, ValueError):
                continue

            if not math.isfinite(gx) or not math.isfinite(gy):
                continue

            matching_tins.append(item)

        if not matching_tins:
            return None

        return min(
            matching_tins,
            key=lambda tin: math.hypot(
                float(tin["gripper_position"][0]),
                float(tin["gripper_position"][1]),
            ),
        )

    # 异常恢复
    def _start_pick_recovery(self, reason: str) -> None:
        self.task3_recover_retry_count += 1

        self._publish_position_hold()
        self._publish_vision_mode("OFF", [])

        # “超过3次”表示第4次进入ERROR
        if (
            self.task3_recover_retry_count
            > self.task3_recover_max_retries
        ):
            self._set_error(
                "TASK3 pick recovery exceeded limit: "
                f"{reason}"
            )
            return

        self.get_logger().warning(
            f"TASK3 pick recovery "
            f"{self.task3_recover_retry_count}/"
            f"{self.task3_recover_max_retries}: "
            f"{reason}"
        )

        self.task3_selected_resource_tin = None
        self.task3_align_stable_start_time = None
        self.task3_descent_hold_z = None

        self._change_state("RECOVER_ASCEND")
    
    # ==================================================================
    # TASK4 辅助函数
    # ==================================================================
 
    def _handle_task4_resume(
        self,
        *,
        task: str,
        requested_state,
    ) -> None:
        if self.mission != "TASK4":
            self.get_logger().warning(
                "Ignored resume because UAV is not in TASK4"
            )
            return

        if not isinstance(self.task4_checkpoint, dict):
            self._set_error(
                "TASK4 checkpoint is unavailable"
            )
            return

        checkpoint_task = str(
            self.task4_checkpoint.get("task", "")
        ).strip().upper()

        checkpoint_state = str(
            self.task4_checkpoint.get("progress", "")
        ).strip().upper()

        if task != checkpoint_task:
            self._set_error(
                f"Resume task mismatch: "
                f"{task} != {checkpoint_task}"
            )
            return

        if (
            not isinstance(requested_state, str)
            or requested_state.strip().upper()
            != checkpoint_state
        ):
            self._set_error(
                "Resume progress does not match checkpoint"
            )
            return

        resume_state = (
            self.task4_resume_state_map
            .get(checkpoint_task, {})
            .get(checkpoint_state)
        )

        if resume_state is None:
            self._set_error(
                f"Unsupported resume point: "
                f"{checkpoint_task}/{checkpoint_state}"
            )
            return

        self._prepare_task4_resume(
            task=checkpoint_task,
            original_state=checkpoint_state,
        )

        # 清理TASK4数据
        self.begin_requests["TASK4"] = False
        self.finish_requests["TASK4"] = False
        self.task4_is_executor = False
        self.task4_target_vehicle = None
        self.task4_target_position = None
        self.task4_ack_start_time = None
        self.task4_last_event = None
        self.error = None

        self.get_logger().info(
            f"TASK4 resume: "
            f"{checkpoint_task}/{checkpoint_state} "
            f"-> {checkpoint_task}/{resume_state}"
        )

        # 恢复原任务
        self._change_mission(
            checkpoint_task,
            resume_state,
        )

        self.task4_checkpoint = None

    def _prepare_task4_resume(
        self,
        *,
        task: str,
        original_state: str,
    ) -> None:

        # MAP在等待起落架收起时被TASK4中断
        if (
            task == "MAP"
            and original_state == "WAIT_GEAR_UP"
        ):
            # 重新发送收起命令
            self._publish_landing_gear_command("up")

            # 重新开始等待计时
            self.landing_gear_wait_start_time = (
                time.monotonic()
            )
            return

        # TASK1重新计算接收目标的超时时间
        if (
            task == "TASK1"
            and original_state == "RECEIVE_TARGET"
        ):
            self.task1_begin_time = time.monotonic()
            return

        if task != "TASK3":
            return

        # 所有TASK3恢复都清除旧视觉帧
        self.latest_task3_objects = []
        self.last_task3_vision_time = None

        # 等待起落架下降时被TASK4中断
        if original_state == "WAIT_GEAR_DOWN":
            # 删除TASK4之前的旧等待时间
            self.landing_gear_wait_start_time = None
            return

        # 抓取前：重新对准并检测TIN
        if original_state in {
            "DETECT_TIN",
            "CHECK_PLATFORM",
            "ALIGN_TIN_XY",
            "DESCEND_TO_TIN",
            "RECOVER_ASCEND",
            "PICK",
        }:
            self.task3_after_alignment_state = "DETECT_TIN"
            self.task3_detect_tin_start_time = None
            self.task3_tin_objects = []
            self.task3_tin_count = 0
            self.task3_selected_resource_tin = None
            self.task3_pick_target_ned = None
            self.task3_align_stable_start_time = None
            self.task3_descent_hold_z = None
            self.vertical_move_xy = None
            return

        # 抓取后：返回平台中心并重新验证
        if original_state in {
            "WAIT_GRIPPER_CLOSE",
            "RETURN_SEARCH_HEIGHT",
            "VERIFY",
        }:
            self.task3_gripper_close_start_time = None
            self.task3_verify_start_time = None
            self.task3_tin_count_after_pick = 0
            return

        # 放置前：回目标平台并重新识别circle
        if original_state in {
            "PLACE_RESOURCE",
            "MOVE_TO_PLACE_POSITION",
        }:
            self.task3_after_alignment_state = "PLACE_RESOURCE"
            self.task3_place_target_ned = None
            self.task3_place_start_time = None
            return

        # MOVE_TO_PLATFORM可能属于抓取或放置阶段
        # 保留原来的task3_after_alignment_state
        if original_state == "MOVE_TO_PLATFORM":
            if self.task3_after_alignment_state == "DETECT_TIN":
                self.task3_detect_tin_start_time = None
                self.task3_tin_objects = []
                self.task3_tin_count = 0
                self.task3_selected_resource_tin = None
                self.task3_pick_target_ned = None

            elif self.task3_after_alignment_state == "PLACE_RESOURCE":
                self.task3_place_target_ned = None

            return

        # open已经发送，恢复后直接上升
        if original_state == "WAIT_FOR_PLACE":
            self.task3_place_start_time = None

    # task4 checkpoint的发布函数
    def _publish_checkpoint(self) -> None:
        if self.task4_checkpoint is None:
            return

        msg = String()
        msg.data = json.dumps(
            self.task4_checkpoint,
            ensure_ascii=False,
        )
        self.checkpoint_pub.publish(msg)

        self.get_logger().info(
            f"Published checkpoint: "
            f"{self.task4_checkpoint}"
        )

    # ==================================================================
    # 公共状态与安全辅助函数
    # ==================================================================
    def _change_state(self, new_state: str) -> None:
        if new_state == self.state:
            return

        if self.local_position is not None:
            self.position_interpolator.reset(
                self.local_position
            )

        self.vertical_move_xy = None
        self.hold_position_ned = None
        old_state = self.state
        self.state = new_state

        self.get_logger().info(
            f"{self.mission}: {old_state} -> {new_state}"
        )

    def _change_mission(
        self,
        new_mission: str,
        initial_state: str,
    ) -> None:
    # --------------
        self.get_logger().error(
            f"Offboard exit detected: "
            f"nav_state={self.vehicle_nav_state}, "
            f"current_ned={self.local_position}, "
            f"last_published_sp={self.last_published_position_sp}, "
            f"route_index={self.map_route_index}"
        )
    # --------------
        if self.local_position is not None:
            self.position_interpolator.reset(
                self.local_position
            )
        self.vertical_move_xy = None
        self.hold_position_ned = None
        old_mission = self.mission
        self.mission = new_mission
        self.state = initial_state

        self.get_logger().info(
            f"Mission: {old_mission} -> {new_mission}, "
            f"state={initial_state}"
        )

    def _set_error(self, message: str) -> None:
        if self.mission == "ERROR":
            return

        self.error = message
        self.error_handover_complete = False
        self.error_mode_request_time = None

        self.get_logger().error(message)
        self._publish_vision_mode("OFF", [])
        self._change_mission("ERROR", "ERROR")

    def _local_position_is_fresh(self, now: float) -> bool:
        return (
            self.local_position is not None
            and self.last_local_position_time is not None
            and now - self.last_local_position_time <= self.local_position_timeout_s
        )

    def _hold_if_control_started(self) -> None:
        if self.control_started:
            self._publish_position_hold()

    def _request_offboard(self) -> None:
        msg = VehicleCommand()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.param1 = 1.0
        msg.param2 = 6.0
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True

        self.vehicle_command_pub.publish(msg)

    def _request_position_mode(self) -> None:
        msg = VehicleCommand()
        msg.timestamp = (
            self.get_clock().now().nanoseconds // 1000
        )

        # PX4 Position模式：main mode = 3
        msg.param1 = 1.0
        msg.param2 = 3.0
        msg.param3 = 0.0

        msg.command = (
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        )
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True

        self.vehicle_command_pub.publish(msg)

    def _request_land(self) -> None:
        msg = VehicleCommand()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000

        msg.command = VehicleCommand.VEHICLE_CMD_NAV_LAND
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True

        self.vehicle_command_pub.publish(msg)

    def quat_rotate_gripper_flu_to_ned(
        self,
        quaternion,
        vector,
    ):
        x, y, z = vector

        body_frd = [
            x,
            -y,
            -z,
        ]

        return quat_rotate_frd_to_ned(
            quaternion,
            body_frd,
        )

    def _run_error_handover(self) -> None:
        # 已经交还飞手，不再发送Offboard指令
        if self.error_handover_complete:
            return

        # PX4已确认进入Position
        if self.vehicle_in_position:
            self.error_handover_complete = True
            self.get_logger().error(
                "Position mode confirmed; "
                "mission control stopped, pilot may take over"
            )
            return

        # 切换成功前继续保持零速度
        self._publish_position_hold()

        now = time.monotonic()

        # 每0.5秒重新请求一次Position
        if (
            self.error_mode_request_time is None
            or now - self.error_mode_request_time >= 0.5
        ):
            self._request_position_mode()
            self.error_mode_request_time = now

    def _visual_xy_target_ned(
        self,
        item: dict[str, Any],
        target_z: float,
    ) -> list[float]:
        if self.local_position is None:
            raise ValueError(
                "Local position is unavailable"
            )

        if self.vehicle_attitude_q is None:
            raise ValueError(
                "Vehicle attitude is unavailable"
            )

        position = item["gripper_position"]

        gx = float(position[0])
        gy = float(position[1])

        # 夹爪FLU相对位移转换为Local NED相对位移
        offset_ned = self.quat_rotate_gripper_flu_to_ned(
            self.vehicle_attitude_q,
            [gx, gy, 0.0],
        )

        # 当前绝对位置 + 视觉相对位移
        return [
            self.local_position[0] + offset_ned[0],
            self.local_position[1] + offset_ned[1],
            float(target_z),
        ]

    # 控制起落架的发布器
    def _publish_landing_gear_command(
        self,
        command: str,
    ) -> None:
        command = command.strip().lower()

        if command not in {"up", "down"}:
            self.get_logger().warning(
                f"Invalid landing gear command: {command}"
            )
            return

        msg = String()
        msg.data = command
        self.landing_gear_command_pub.publish(msg)

        self.get_logger().info(
            f"Landing gear command sent: {command}"
        )

    def _publish_gripper_command(
        self,
        command: str,
    ) -> None:
        command = command.strip().lower()

        if command not in {"open", "close"}:
            self.get_logger().warning(
                f"Invalid gripper command: {command}"
            )
            return

        msg = String()
        msg.data = json.dumps(
            {"command": command},
            ensure_ascii=False,
        )

        self.gripper_command_pub.publish(msg)

        self.get_logger().info(
            f"Gripper command sent: {command}"
        )

    # 控制开启视觉检测的发布函数
    def _publish_vision_mode(
        self,
        mode: str,
        object_types: list[str],
    ) -> None:
        msg = String()
        msg.data = json.dumps(
            {
                "mode": mode,
                "object_types": object_types,
            },
            ensure_ascii=False,
        )
        self.vision_control_pub.publish(msg)

    def _vertical_position_target(
        self,
        target_z: float,
    ) -> list[float]:
        if self.local_position is None:
            raise ValueError("Local position unavailable")

        if self.vertical_move_xy is None:
            self.vertical_move_xy = [
                self.local_position[0],
                self.local_position[1],
            ]

        return [
            self.vertical_move_xy[0],
            self.vertical_move_xy[1],
            float(target_z),
        ]

    def _publish_position_hold(self) -> None:
        if self.local_position is None:
            return

        # 第一次调用时锁定当前位置
        if self.hold_position_ned is None:
            self.hold_position_ned = [
                float(self.local_position[0]),
                float(self.local_position[1]),
                float(self.local_position[2]),
            ]

            self.position_interpolator.reset(
                self.hold_position_ned
            )

        self._publish_position_setpoint(
            self.hold_position_ned
        )

    # ==================================================================
    # 唯一的 PX4 速度 setpoint 发布函数
    # ==================================================================
    def _publish_velocity(self, vx: float, vy: float, vz: float) -> None:
        timestamp_us = self.get_clock().now().nanoseconds // 1000

        self.hold_position_ned = None

        control_mode = OffboardControlMode()
        control_mode.timestamp = timestamp_us
        control_mode.position = False
        control_mode.velocity = True
        control_mode.acceleration = False
        control_mode.attitude = False
        control_mode.body_rate = False
        self.offboard_control_mode_pub.publish(control_mode)

        setpoint = TrajectorySetpoint()
        setpoint.timestamp = timestamp_us
        setpoint.position = [math.nan, math.nan, math.nan]
        setpoint.velocity = [vx, vy, vz]
        setpoint.acceleration = [math.nan, math.nan, math.nan]
        if hasattr(setpoint, "jerk"):
            setpoint.jerk = [math.nan, math.nan, math.nan]
            
        # 整个任务使用记录下来的固定偏航角
        if self.fixed_mission_yaw is not None:
            setpoint.yaw = float(
                self.fixed_mission_yaw
            )
        else:
            setpoint.yaw = math.nan

        setpoint.yawspeed = math.nan
        self.trajectory_setpoint_pub.publish(setpoint)


    def _publish_interpolated_position(
        self,
        target_position: list[float],
        *,
        max_speed_m_s: float | None = None,
    ) -> None:
        if self.local_position is None:
            return
        
        self.hold_position_ned = None

        position_sp = self.position_interpolator.update(
            self.local_position,
            target_position,
            max_speed_m_s=max_speed_m_s,
        )

        self._publish_position_setpoint(position_sp)
    
    def _publish_position_setpoint(
        self,
        position: list[float],
    ) -> None:
        timestamp_us = (
            self.get_clock().now().nanoseconds // 1000
        )

        control_mode = OffboardControlMode()
        control_mode.timestamp = timestamp_us
        control_mode.position = True
        control_mode.velocity = False
        control_mode.acceleration = False
        control_mode.attitude = False
        control_mode.body_rate = False
        self.offboard_control_mode_pub.publish(control_mode)

        setpoint = TrajectorySetpoint()
        setpoint.timestamp = timestamp_us
        setpoint.position = [
            float(position[0]),
            float(position[1]),
            float(position[2]),
        ]
        setpoint.velocity = [
            math.nan,
            math.nan,
            math.nan,
        ]
        setpoint.acceleration = [
            math.nan,
            math.nan,
            math.nan,
        ]

        if hasattr(setpoint, "jerk"):
            setpoint.jerk = [
                math.nan,
                math.nan,
                math.nan,
            ]

        setpoint.yaw = (
            float(self.fixed_mission_yaw)
            if self.fixed_mission_yaw is not None
            else math.nan
        )
        setpoint.yawspeed = math.nan
        # ----------------
        self.last_published_position_sp = [
            float(position[0]),
            float(position[1]),
            float(position[2]),
        ]
        # ----------------
        self.trajectory_setpoint_pub.publish(setpoint)
    
    # ==================================================================
    # 统一进度发布
    # ==================================================================
    def _publish_progress(self) -> None:
        if not self.control_started:
            return

        msg = String()
        msg.data = json.dumps(
            {
                "vehicle": "uav",
                "task": self.mission,
                "progress": self.state,
                "error": self.error,
                "command": None,
            },
            ensure_ascii=False,
        )
        self.progress_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UavMissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

