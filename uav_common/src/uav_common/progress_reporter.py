"""Shared UAV progress publisher."""

from __future__ import annotations

import json
from typing import Any

from rclpy.node import Node
from std_msgs.msg import String


class UavProgressReporter:
    """Publish a task's current status and error in the common UAV format."""

    def __init__(
        self,
        node: Node,
        task: str,
        topic: str = "/tongji_mrl/tsky/progress",
    ) -> None:
        task = task.strip().upper()
        topic = topic.strip()

        if not task:
            raise ValueError("task must not be empty")
        if not topic:
            raise ValueError("topic must not be empty")

        self._task = task
        self._publisher = node.create_publisher(
            String,
            topic,
            10,
        )

    def publish(
        self,
        status: str | None,
        error: Any | None,
    ) -> None:
        """Publish the status and error currently held by the task node."""
        payload = {
            "vehicle": "uav",
            "task": self._task,
            "progress": status,
            "error": error,
            "command": None,
        }

        msg = String()
        msg.data = json.dumps(
            payload,
            ensure_ascii=False,
        )
        self._publisher.publish(msg)

