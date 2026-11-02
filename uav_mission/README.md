# uav_mission

RobotX UAV 的单节点总状态机骨架。一个 ROS 2 节点统一管理 MAP、TASK1、TASK3、TASK4，并且只有这一处向 PX4 发布控制指令。

## 状态层级

```text
mission = MAP / TASK1 / TASK3 / TASK4 / FINISHED / ERROR
state   = 当前任务内部状态
```

目前 TASK1 包含已有的接收目标、飞行、悬停、跟随和返航占位流程。MAP、TASK3、TASK4 的运行状态暂时持续发布零速度，等待后续填充具体逻辑。

## 构建和启动

```bash
cd /home/robotx/robotx_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select uav_mission --symlink-install
source install/setup.bash
ros2 launch uav_mission mission.launch.py
```

