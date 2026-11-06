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
| MAP | dock、buoy、platform | 通过原 MAP 接口逐批输出 local NED 目标，内部累计记录 |
| TASK1 | buoy | 将蓝色闪烁浮标送入已有 TASK1 接收目标流程 |
| TASK3 | platform、tin | 使用原有夹爪坐标对齐、抓取流程 |

TASK3 的 `PLACE_RESOURCE` 阶段额外开启原有 circle 检测。任务结束、ERROR 或进入 TASK4 时停止识别；恢复任务时按该任务重新开启。进程和模型启动后常驻，OFF 时不启动新推理，已经开始的推理可以完成。任务模式切换保留灯标轨迹、光度历史、异步结果、RGB 和深度缓存；检测器原有的时间与几何检查继续决定数据是否可用。

`uav_mission_node.py` 是唯一任务模式来源，通过 `/tongji_mrl/tsky/downward_vision_control` 发布 `{mode, object_types}`。两套独立检测器和下视节点执行同一条指令，不再由 TASK3 子步骤分别切换相互冲突的识别集合。

`mission_perception_node.py` 是统一识别结果发布节点。MAP、TASK1、TASK3 都通过 `/tongji_mrl/tsky/map_targets_ned` 输出实时识别结果，由 `task` 和 `frame` 区分；主任务只订阅这一处实时视觉接口。

内部识别输入及路由如下：

- 灯光 `/task1/beacons/results_json`：采用检测器已有的位置、灯状态有效标志，将 ENU 转为同一 PX4 原点的 NED。
- dock `/robotx/perception/uav/pontoon/observation`：记录主干中心的 NED 位置，并保留分支几何；主干中心不代表泊位中心。
- platform `/tongji_mrl/tsky/perception/platforms_ned`：下视节点的内部 NED 输出，由 `downward_vision.yaml` 的 `topics.map_platforms` 配置；只有汇总节点订阅它用于 MAP。
- TASK3 `/tongji_mrl/tsky/perception/task3_targets_gripper`：下视节点的内部夹爪坐标输出，由 `downward_vision.yaml` 的 `topics.task3_targets` 配置；汇总节点仅在 TASK3 转发当前阶段需要的类型。
- 唯一对外识别话题 `/tongji_mrl/tsky/map_targets_ned` 由 `mission_perception.yaml` 和 `mission.yaml` 的 `topics.map_result` 配置。总 launch 中只有汇总节点发布此话题；主任务通过 `_on_perception_result` 按当前任务、消息 `task/frame` 和目标类型分发。两份可独立启动的下视实现都使用内部输入；总 launch 只启动 `downward_vision2`。
- 每次检测发送该来源本次的有效目标，保留原 `task: MAP`、`frame: local_ned`、`objects` 协议。每个对象包含 `type/color/state/id/NED_position`；附带 `position_valid/state_valid`，dock 另带 `planning_ready/geometry`。顶层 `source` 区分 `buoy/platform/dock`，`stamp/rgb_stamp` 保留源采样时间，`sequence` 保留源序号（源未提供时为 null）。缺少采样时间时 stamp 为 null，不替换成接收时间。
- MAP 消息是各来源的观测更新，不是全场累计快照。某来源没有有效目标时发送该来源的 `objects: []`；坐标系不匹配时丢弃该批。接收方不能把一个来源的空批次当作其它来源目标消失。对象 `id` 保留源检测编号，结合 `type` 使用；它不是经过验证的物理目标编号。
- 累计历史只保存到 `~/.ros/uav_mission/maps/map_<session>.json`，不再创建 `/tongji_mrl/tsky/map_records` 发布者。同一次运行中暂停/恢复保留记录，每条记录保存首次、最新位置和最多64条最近观测。0.5 秒定时器仅保存变化，不重复发送旧观测；TASK1/TASK3 结果不会追加 MAP 历史。
- 内部输入与公开输出必须不同，汇总节点会检查 ROS remap 后的话题名，防止自身消息被再次录入。下视的 `map_platforms/task3_targets` 要与汇总器的 `platforms/task3_targets` 分别匹配，不要把公开 `map_targets_ned` 配回任一内部输入。
- TASK1 的有效蓝闪浮标通过同一公开话题发送，`task: TASK1`、`frame: local_ned`、`source: buoy`，对象使用 `NED_position`。没有有效蓝闪时不发新目标，已有候选保留。旧 `task1_buoy` 实时话题及 local/external 订阅选择已移除。
- TASK3 通过同一公开话题发送，`task: TASK3`、`frame: gripper`、`source: downward`，对象使用 `gripper_position`（FLU，相对夹爪，单位米）。仍保留源 `stamp/rgb_stamp/sequence`。旧 `task3_target_gripper` 对外话题已移除，坐标不会因统一话题而改为 NED。
- TASK3 导航所需的两个平台 NED 位置继续由外部通过 `/tongji_mrl/twave/task3_platform` 指定，资源／目的颜色继续来自 `/tongji_mrl/twave/task3`；这些是任务参数输入。TASK1 的 USV 起终点事件和船位置输入也继续保留。

| 公开话题中的消息 | 坐标字段 | 本机消费与保存 |
|---|---|---|
| `task=MAP, frame=local_ned` | `NED_position` | 对外发布本次观测，汇总节点内部累计保存；主任务不据此执行 TASK1/TASK3 |
| `task=TASK1, frame=local_ned` | `NED_position` | 主任务选择有效蓝闪浮标，不追加 MAP 历史 |
| `task=TASK3, frame=gripper` | `gripper_position` | 主任务取当前阶段的 platform/tin/circle，不追加 MAP 历史 |

任务切换时到达的其它任务结果、错误坐标系、坏结构或坏坐标会被忽略。TASK3 合法空批次仍发送；只有未请求类型的非空批次不会改写成空批次，防止把无关或异常数据变成“没有目标”。

结果适配按检测器的坐标字段和有效标志转换，并验证坐标恰好包含三个有限数；无效位置不对外输出或写入历史。浮标状态未确认时颜色和状态均为 unknown；码头可选几何点无效时置 null，几何不完整时 planning_ready 为 false。保留源观测时间，不按时间戳或任务标签额外拒收。TASK1 有有效蓝闪目标时更新候选；任务切换、暂无匹配目标或接收间隔变长均不清空已有候选。

任务接入和记录节点均位于 `uav_mission`，纯坐标适配和记录类位于 `uav_common`。`uav_map`、`uav_task1`、`uav_task3`、`uav_task4` 保持原样。

## 识别输出检查

| 输出 | 消费与用途 | 结论 |
|---|---|---|
| 灯标 `results_json` | 汇总节点生成 MAP 浮标与 TASK1 蓝闪目标 | 必需的原始输入 |
| 码头 `pontoon/observation` | 汇总节点生成 MAP dock 和几何 | 必需的原始输入；主干中心不代表泊位中心，分支标签仍是图像顺序 |
| 内部 `perception/platforms_ned` | 汇总节点生成 MAP platform | 必需，替代下视直接占用公开话题 |
| 内部 `perception/task3_targets_gripper` | 汇总节点转发 TASK3 目标 | 必需，内部夹爪坐标输入 |
| 公开 `map_targets_ned` | 所有任务的实时识别接口，主任务和外部订阅者按 task/frame 使用 | 唯一识别出口；本仓库不包含 USV 接收实现 |
| 公开话题的 TASK1 buoy | 主任务选择有效蓝闪浮标 | 必需；不再使用独立 task1_buoy 话题 |
| 公开话题的 TASK3 platform | 主任务的平台对齐 | 必需，夹爪坐标而非 NED |
| 公开话题的 TASK3 tin | 选取物资、抓取对齐和抓后计数 | 必需；深度无效会过滤对象，消费风险见下文 |
| 公开话题的 TASK3 circle | PLACE_RESOURCE 阶段的放置位置选择 | 必需，仅该阶段开启 |
| MAP/TASK3 debug、灯标 positions_json/调试图、码头 debug | 保留原始检测、深度、定位或几何诊断信息 | 诊断输出，不作为额外 MAP 接口 |
| 独立灯标 map_json | 单独灯标建图工具 | 集成 launch 已 `start_map=false`，没有重复启动 |

TASK3 业务消息现在也保留源 `stamp/rgb_stamp/sequence`。两个已有消费问题仍需独立修复：主任务尚以接收时刻判断 TASK3 观测新鲜度；VERIFY 按过滤后的 tin 数量判断抓取，无法定位的 tin 消失可能被误判为抓取成功。本次修改没有改变抓取状态机。接口测试不能证明真实相机识别精度、外参精度或 USV 实机接收。

回归测试：

```bash
python3 -m pytest -q uav_common/test uav_mission/test color_light_detector/test robotx_uav_pontoon_v37_planning_preview/ros2_src/robotx_uav_pontoon_bridge/test
```

真实 DDS 检查需先 source ROS 和已构建的 `robotx_uav_pontoon_interfaces` overlay，再在仓库根目录运行（构造检测输入，不启动相机、模型或飞控）：

```bash
PYTHONPATH="$PWD/uav_common/src:$PWD/uav_mission/src:$PYTHONPATH" \
  ROS_DOMAIN_ID=214 ROS_LOCALHOST_ONLY=1 \
  python3 uav_mission/test/map_routing_dds_smoke.py
```

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

## UAV JSON 状态

`uav_status_json_node.py` 从 PX4 读取控制状态、着陆状态、GNSS 位置、
水平速度、姿态航向和电池信息，以 5 Hz 发布 JSON 到
`/tongji_mrl/tsky/uav_status`。本机 Drone ID 在
`config/uav_status.yaml` 的 `identity.vehicle_id` 中配置。

```bash
ros2 launch uav_mission uav_status_json.launch.py
ros2 topic echo /tongji_mrl/tsky/uav_status
```

已有 Arducam 图像发布器时，可加 `start_buoy_camera:=false`。下视节点仍通过 `downward_vision2.launch.py` 启动；总启动文件包含以上两者。

