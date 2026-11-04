# uav_mission

RobotX UAV 的单节点总状态机骨架。一个 ROS 2 节点统一管理 MAP、TASK1、TASK3、TASK4，并且只有这一处向 PX4 发布控制指令。

## 状态层级

```text
mission = MAP / TASK1 / TASK3 / TASK4 / FINISHED / ERROR
state   = 当前任务内部状态
```

MAP 管理巡航，TASK1 接收浮标目标，TASK3 管理平台对齐与物资操作。感知启停统一由当前任务控制。

## 构建和启动

```bash
cd /home/robotx/robotx_ws
source /opt/ros/humble/setup.bash
colcon build --packages-up-to uav_mission --symlink-install
source install/setup.bash
ros2 launch uav_mission all_uav_nodes.launch.py
```

## 任务与识别

| 当前任务收到 begin 后 | 开启的识别 | 输出用途 |
|---|---|---|
| MAP | dock、buoy、platform | 累计记录 local NED 位置 |
| TASK1 | buoy | 将蓝色闪烁浮标送入已有 TASK1 接收目标流程 |
| TASK3 | platform、tin | 使用原有夹爪坐标对齐、抓取流程 |

TASK3 的 `PLACE_RESOURCE` 阶段额外开启原有 circle 检测。任务结束、ERROR 或进入 TASK4 时停止识别；恢复任务时按该任务重新开启。进程和模型启动后常驻，OFF 时不启动新推理，已经开始的推理可以完成。任务模式切换保留灯标轨迹、光度历史、异步结果、RGB 和深度缓存；检测器原有的时间与几何检查继续决定数据是否可用。

`uav_mission_node.py` 是唯一任务模式来源，通过 `/tongji_mrl/tsky/downward_vision_control` 发布 `{mode, object_types}`。两套独立检测器和下视节点执行同一条指令，不再由 TASK3 子步骤分别切换相互冲突的识别集合。

`mission_perception_node.py` 负责三源结果适配和记录：

- 灯光 `/task1/beacons/results_json`：采用检测器已有的位置、灯状态有效标志，将 ENU 转为同一 PX4 原点的 NED。
- dock `/robotx/perception/uav/pontoon/observation`：记录主干中心的 NED 位置，并保留分支几何；主干中心不代表泊位中心。
- platform `/tongji_mrl/tsky/map_targets_ned`：使用下视节点已有的 NED 输出。
- MAP 累计结果发布到 `/tongji_mrl/tsky/map_records`，保存到 `~/.ros/uav_mission/maps/map_<session>.json`。同一次运行中暂停/恢复保留记录，每条记录保存首次、最新位置和最多64条最近观测。
- TASK1 本地浮标发布到 `/tongji_mrl/tsky/task1_buoy`。`mission.yaml` 的 `target.task1_buoy_source` 默认 `local`；设为 `external` 时只订阅原 `/tongji_mrl/twave/task1_buoy`，两路不会同时争用目标缓存。

结果适配按检测器的坐标字段和有效标志转换，保留源观测时间；没有源时间时记录 null，不按时间戳或任务标签额外拒收。TASK1 有有效蓝闪目标时更新候选；任务切换、暂无匹配目标或接收间隔变长均不清空已有候选。

任务接入和记录节点均位于 `uav_mission`，纯坐标适配和记录类位于 `uav_common`。`uav_map`、`uav_task1`、`uav_task3`、`uav_task4` 保持原样。

## 相机和位置配置

`mission_perception.launch.py` 启动灯光、dock、位姿桥和结果记录；默认灯光使用 Arducam，下视 platform/tin 和 dock 使用现有 OAK 数据。修改相机话题、模型文件和外参时使用相应配置：

- `config/downward_vision.yaml`：platform/tin/circle 模型路径、OAK 内参、`T_body_camera`、`T_gripper_camera`。
- `config/mission_perception.yaml`：结果话题、记录目录、PX4 时间匹配参数。
- `color_light_detector/config/task1_position_px4.yaml`：灯光定位使用的高度平面和相机标定。原有未完成标定的标志保持原值；检测器返回 `position.valid=false` 的估计不会录为有效坐标。

`px4_perception_pose_node.py` 把现有 PX4 位置、姿态按采样时间转换成 `/tongji_mrl/tsky/perception/odometry`，同时发布 `map_ned -> base_link_frd`。dock 相机变换复用下视的 `T_body_camera`，光学 frame 默认从 `/oak/rgb/camera_info` 获取。灯光使用同一位姿源，输出 ENU 后由适配器转换回 NED。

dock 默认接原始 RGB，使用 CameraInfo 的 K/D 修正射线；深度需要已对齐 RGB。时间偏移参数只用于已知的时钟差，不用消息到达时间替代采样时间。

单独运行感知接入（不启动飞控任务节点）：

```bash
ros2 launch uav_mission mission_perception.launch.py
```

已有 Arducam 图像发布器时，可加 `start_buoy_camera:=false`。下视节点仍通过 `downward_vision2.launch.py` 启动；总启动文件包含以上两者。

