"""ROS adapter for live Task 1 map observations and durable map records."""

import json
import os
from pathlib import Path
import sqlite3

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
import yaml

from .task1_beacon_map import BeaconMap, storage_paths
from .task1_persistent_beacon_store import PersistentBeaconStore


class Task1BeaconMapNode(Node):
    def __init__(self):
        super().__init__("task1_beacon_map")
        self.declare_parameter("map_config", "")
        filename = str(self.get_parameter("map_config").value)
        if not filename:
            from ament_index_python.packages import get_package_share_directory
            filename = str(Path(get_package_share_directory("color_light_detector")) /
                           "config/task1_beacon_map_unconfigured.yaml")
        config = yaml.safe_load(Path(filename).expanduser().read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("simulation") is not False:
            raise ValueError("Live map requires simulation: false")
        self.map = BeaconMap(config)
        self.snapshot_file, database_file = storage_paths(
            config["snapshot_file"], config["persistent_db_file"])
        self.store = PersistentBeaconStore(database_file, config)
        self.persistence_error = None
        self.last_publish_time = float("-inf")
        self.last_warned = None
        self.last_saved_snapshot_key = None
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.publisher = self.create_publisher(String, config["output_topic"], qos)
        self.subscription = self.create_subscription(
            String, config["input_topic"], self._receive, qos)
        self.timer = self.create_timer(1.0, lambda: self._publish(save_snapshot=True))
        self.get_logger().info(
            "Task 1 durable beacon map started; live shared positions remain invalid "
            "until the source origin and transform are verified")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _receive(self, message):
        try:
            payload = json.loads(message.data)
            now = self._now()
            accepted = self.map.ingest(payload, now)
            if not accepted and self.map.last_rejection != self.last_warned:
                self.get_logger().warning("Map input rejected: " + self.map.last_rejection)
                self.last_warned = self.map.last_rejection
            if not accepted:
                return
            if self.map.last_rejection:
                if self.map.last_rejection != self.last_warned:
                    self.get_logger().warning(
                        "Map alignment degraded: " + self.map.last_rejection)
                    self.last_warned = self.map.last_rejection
            else:
                self.last_warned = None
            try:
                self.store.ingest(payload, self.map.snapshot(now))
                self.persistence_error = None
            except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
                current_error = str(error)
                if current_error != self.persistence_error:
                    self.get_logger().error(
                        "Durable map write failed: " + current_error)
                self.persistence_error = current_error
            if now - self.last_publish_time >= 0.2:
                self._publish()
        except (TypeError, ValueError, KeyError) as error:
            self.get_logger().warning("Map JSON rejected: " + str(error))

    def _publish(self, save_snapshot=False):
        now = self._now()
        snapshot = self.map.snapshot(now)
        snapshot["schema_version"] = 2
        snapshot["identity_scope"] = "PERSISTENT_MAP_RECORD_ID_NOT_PHYSICAL_ID"
        try:
            persistent = self.store.snapshot(now)
            if self.persistence_error is not None:
                persistent = {"status": "ERROR", "error": self.persistence_error,
                              "records": []}
            else:
                for beacon in snapshot["beacons"]:
                    beacon_id = self.store.link_for(
                        snapshot["session_id"], beacon["track_id"])
                    beacon["persistent_beacon_id"] = beacon_id
                    beacon["durably_recorded"] = beacon_id is not None
        except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
            persistent = {"status": "ERROR", "error": str(error), "records": []}
        for beacon in snapshot["beacons"]:
            beacon.setdefault("durably_recorded", False)
        snapshot["persistence_ok"] = persistent.get("status") == "OK"
        snapshot["persistence_enabled"] = persistent.get(
            "persistence_enabled", False)
        snapshot["persistent_map"] = persistent
        encoded = json.dumps(snapshot, separators=(",", ":"), allow_nan=False)
        message = String()
        message.data = encoded
        self.publisher.publish(message)
        self.last_publish_time = now
        if not save_snapshot:
            return
        semantic_snapshot = dict(snapshot)
        semantic_snapshot["generated_at_s"] = None
        snapshot_key = json.dumps(
            semantic_snapshot, separators=(",", ":"), allow_nan=False)
        if snapshot_key == self.last_saved_snapshot_key:
            return
        try:
            self.snapshot_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.snapshot_file.with_name(self.snapshot_file.name + ".tmp")
            temporary.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n",
                                 encoding="utf-8")
            os.replace(temporary, self.snapshot_file)
            self.last_saved_snapshot_key = snapshot_key
        except OSError as error:
            self.get_logger().warning("Map snapshot not saved: " + str(error))

    def destroy_node(self):
        self.store.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Task1BeaconMapNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
