uav_mission/src/uav_mission 这是主节点的位置，由这个主节点来控制map，task1，3，4的整个流程，通信按照比赛要求书写
其他的包里放置着特定任务对应的python类，在主节点运行中被调用
尚未完成的是：
1.task4只完成了优先级的考虑和接收命令，还没完成“去到特定的点”的代码，需要在现有框架下补充
2.通信按照比赛/主规划器要求书写，但是还不是用编码的版本

buoy、dock 已按任务接入；MAP 位置记录和配置见 uav_mission/README.md。

ground和air test需要的uav发布其相关状态的代码以及上传，详细见：uav_mission/README.md中的“UAV JSON 状态”

air test需要的自主take off，之后由飞手顺利接管的代码在uav_mission/src/uav_mission/ascend_hover_node.py，详细见：uav_mission/README.md中的“自主take off”
