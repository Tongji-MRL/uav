"""Route mission perception to MAP records and the TASK1 buoy input."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
import yaml

from uav_common.mission_perception import (
    MissionMapRecorder, adapt_beacons, adapt_dock, adapt_platform, task1_blue_objects,
)


def stamp_dict(seconds):
    total = round(seconds * 1_000_000_000)
    return {"sec": total // 1_000_000_000, "nanosec": total % 1_000_000_000}


def point_list(point):
    return [float(point.x), float(point.y), float(point.z)]


class MissionPerceptionNode(Node):
    def __init__(self):
        super().__init__("mission_perception")
        self.declare_parameter("config_file", "")
        path = Path(self.get_parameter("config_file").value).expanduser()
        config = yaml.safe_load(path.read_text(encoding="utf-8"))["mission_perception"]
        topics = config["topics"]
        self.session_id = uuid4().hex
        self.mode, self.object_types = "OFF", set()
        self.recorder = MissionMapRecorder(
            match_radius_m=float(config.get("map_match_radius_m", 1.0)),
            history_limit=int(config.get("map_history_limit", 64)),
        )
        directory = Path(config["snapshot_directory"]).expanduser()
        self.snapshot_file = directory / ("map_" + self.session_id + ".json")
        self.dirty = self.have_map_session = False
        self.source_status = {}
        control_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(String, topics["map_records"], control_qos)
        self.buoy_pub = self.create_publisher(String, topics["task1_buoy"], 10)
        self.control_sub = self.create_subscription(
            String, topics["vision_control"], self._control, control_qos)
        self.beacon_sub = self.create_subscription(String, topics["beacons"], self._beacons, 10)
        self.platform_sub = self.create_subscription(String, topics["platforms"], self._platforms, 10)
        from robotx_uav_pontoon_interfaces.msg import PontoonObservation
        self.dock_sub = self.create_subscription(PontoonObservation, topics["dock"], self._dock, 10)
        self.timer = self.create_timer(0.5, self._tick)
        self.get_logger().info("MAP position records: " + str(self.snapshot_file))

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _publish(publisher, payload):
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        publisher.publish(message)

    def _control(self, message):
        command = json.loads(message.data)
        mode, objects = command["mode"], set(command["object_types"])
        if (mode, objects) == (self.mode, self.object_types):
            return
        self.mode, self.object_types = mode, objects
        now = self._now()
        self.recorder.set_mode(mode, now, session_id=self.session_id)
        self.have_map_session |= mode == "MAP"
        self.dirty = self.have_map_session
        self._publish_map(now)

    def _enabled(self, object_type, modes):
        return self.mode in modes and object_type in self.object_types

    def _record(self, source, result, now):
        self.source_status[source] = {"accepted": result["accepted"], "reason": result["reason"]}
        self.dirty |= self.recorder.ingest(source, result, now) > 0

    def _beacons(self, message):
        if not self._enabled("buoy", ("MAP", "TASK1")):
            return
        payload = json.loads(message.data)
        now = self._now()
        result = adapt_beacons(payload)
        if self.mode == "MAP":
            self._record("buoy", result, now)
        else:
            objects = task1_blue_objects(result)
            if objects:
                self._publish_task1(objects, result["observed_at_s"])

    def _platforms(self, message):
        if self._enabled("platform", ("MAP",)):
            now = self._now()
            self._record("platform", adapt_platform(json.loads(message.data)), now)

    def _dock(self, message):
        if not self._enabled("dock", ("MAP",)):
            return
        payload = {
            "stamp": {"sec": message.header.stamp.sec, "nanosec": message.header.stamp.nanosec},
            "world_frame_id": message.world_frame_id,
            "geometry_ready": message.geometry_ready,
            "xyz_valid": message.xyz_valid,
            "planning_ready": message.planning_ready,
            "trunk_center_world_valid": message.trunk_center_world_valid,
            "trunk_start_world_valid": message.trunk_start_world_valid,
            "trunk_end_world_valid": message.trunk_end_world_valid,
            **{key + "_world_m": point_list(getattr(message, key + "_world_m"))
               for key in ("trunk_start", "trunk_end", "trunk_center")},
            "branches": [
                {"index": branch.index, "label": branch.label,
                 **{key + "_world_m": point_list(getattr(branch, key + "_world_m"))
                    for key in ("tip", "root", "center")},
                 **{key + "_world_valid": getattr(branch, key + "_world_valid")
                    for key in ("tip", "root", "center")}}
                for branch in message.branches
            ],
        }
        now = self._now()
        self._record("dock", adapt_dock(payload), now)

    def _publish_task1(self, objects, stamp):
        self._publish(self.buoy_pub, {"task": "TASK1", "frame": "local_ned",
                                    "stamp": stamp_dict(stamp) if stamp is not None else None,
                                    "objects": objects})

    def _publish_map(self, now):
        if not self.have_map_session:
            return
        payload = self.recorder.snapshot(now)
        payload["sources"] = dict(self.source_status)
        payload["snapshot_file"] = str(self.snapshot_file)
        if self.dirty:
            self.recorder.write_snapshot(self.snapshot_file, now)
            self.dirty = False
        self._publish(self.map_pub, payload)

    def _tick(self):
        self._publish_map(self._now())

    def destroy_node(self):
        if self.have_map_session:
            now = self._now()
            self.recorder.set_mode("OFF", now, session_id=self.session_id)
            self.recorder.write_snapshot(self.snapshot_file, now)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MissionPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
