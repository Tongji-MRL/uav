"""Mission objects and accumulated MAP positions for the integrated PX4 pipeline.

Beacon map uses ENU; pontoon map_ned and downward local_ned use NED. The
integrated pose bridge gives these frames the same origin. Detectors retain
ownership of their position/state validity and calibration checks.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile
from uuid import uuid4

BEACON_STATES = {
    "BLINK_RED": ("red", "flashing"), "BLINK_GREEN": ("green", "flashing"),
    "BLINK_BLUE": ("blue", "flashing"), "SOLID_BLUE": ("blue", "solid"),
    "OFF": ("unknown", "off"),
}
MAP_TYPES = {"dock", "buoy", "platform"}


def _point(value):
    """Only admit finite 3-D positions to JSON and distance matching."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        point = [float(axis) for axis in value]
    except (TypeError, ValueError, OverflowError):
        return None
    return point if all(math.isfinite(axis) for axis in point) else None


def observation_stamp(payload):
    """Read acquisition time from the detector's JSON representation."""
    if "observed_at_s" in payload:
        return payload["observed_at_s"]
    stamp = payload.get("stamp", payload.get("rgb_stamp", payload.get("header", {}).get("stamp")))
    if stamp is None:
        return None
    return stamp["sec"] + stamp["nanosec"] * 1e-9


def _envelope(payload):
    # Preserve the original integer nanoseconds for the public protocol. Float
    # seconds are sufficient for history, but lose precision at epoch times.
    stamp = None if "observed_at_s" in payload else payload.get(
        "stamp", payload.get("rgb_stamp", payload.get("header", {}).get("stamp")))
    return {"objects": [], "observed_at_s": observation_stamp(payload),
            "stamp": deepcopy(stamp),
            "sequence": payload.get("sequence"), "accepted": True, "reason": ""}


def _reject(result, reason):
    result.update(accepted=False, reason=reason)
    return result


def adapt_beacons(payload, source_frame="map"):
    """Convert valid color_light ENU positions into local NED buoy objects."""
    result = _envelope(payload)
    for track in payload.get("tracks", []):
        position = track.get("position", {})
        if (not track.get("visible") or position.get("valid") is not True
                or position.get("world_valid") is not True
                or position.get("frame_id") != source_frame):
            continue
        point = _point(position.get("world_xyz_m"))
        if point is None:
            continue
        raw_state = track.get("committed_state", track.get("state", "UNKNOWN"))
        state_valid = track.get("state_valid") is True
        color, state = BEACON_STATES.get(raw_state, ("unknown", "unknown"))
        if not state_valid:
            color, state = "unknown", "unknown"
        result["objects"].append({
            "type": "buoy", "id": track["track_id"], "color": color, "state": state,
            "beacon_state": raw_state, "state_valid": state_valid, "position_valid": True,
            "NED_position": [point[1], point[0], -point[2]], "frame_id": "local_ned",
            "observed_at_s": result["observed_at_s"],
        })
    return result


def task1_blue_objects(payload):
    return [deepcopy(item) for item in payload["objects"]
            if item["type"] == "buoy" and item["beacon_state"] == "BLINK_BLUE"
            and item["state_valid"] and item["position_valid"]]


def adapt_platform(payload):
    """Select platform positions from the internal downward MAP output."""
    result = _envelope(payload)
    if payload.get("frame", payload.get("frame_id")) != "local_ned":
        return _reject(result, "SOURCE_FRAME_MISMATCH")
    for item in payload.get("objects", []):
        if item.get("type") != "platform" or item.get("position_valid") is False:
            continue
        point = _point(item.get("NED_position"))
        if point is None:
            continue
        observation = deepcopy(item)
        observation.update(NED_position=point, state="static", state_valid=True,
                           position_valid=True, frame_id="local_ned",
                           observed_at_s=result["observed_at_s"])
        result["objects"].append(observation)
    return result


def adapt_dock(payload, source_frame="map_ned"):
    """Represent a pontoon as one dock, retaining its branch geometry."""
    result = _envelope(payload)
    if payload.get("world_frame_id") != source_frame:
        return _reject(result, "SOURCE_FRAME_MISMATCH")
    if not all(payload.get(key) is True for key in
               ("geometry_ready", "xyz_valid", "trunk_center_world_valid")):
        result["reason"] = "DOCK_GEOMETRY_OR_POSITION_INVALID"
        return result
    center = _point(payload.get("trunk_center_world_m"))
    if center is None:
        result["reason"] = "DOCK_GEOMETRY_OR_POSITION_INVALID"
        return result
    geometry = {"source_frame_id": source_frame,
                "branch_identity_scope": "IMAGE_ORDER_NOT_PHYSICAL_BAY_ID", "branches": []}
    for key in ("trunk_start", "trunk_end", "trunk_center"):
        geometry[key + "_NED_position"] = (
            _point(payload.get(key + "_world_m")) if payload.get(key + "_world_valid") else None)
    for branch in payload.get("branches", []):
        item = {"index": branch["index"], "label": branch["label"]}
        for key in ("tip", "root", "center"):
            item[key + "_NED_position"] = (
                _point(branch.get(key + "_world_m")) if branch.get(key + "_world_valid") else None)
        geometry["branches"].append(item)
    complete_geometry = (
        all(geometry[key + "_NED_position"] is not None
            for key in ("trunk_start", "trunk_end", "trunk_center"))
        and bool(geometry["branches"])
        and all(branch[key + "_NED_position"] is not None
                for branch in geometry["branches"] for key in ("tip", "root", "center"))
    )
    metadata = {key: payload[key] for key in (
        "detector_version", "status", "invalid_reason", "geometry_score", "branch_count",
        "xyz_method", "geometry_ready", "xyz_valid", "planning_ready") if key in payload}
    result["objects"] = [{
        "type": "dock", "id": payload.get("id", 0), "color": "unknown", "state": "static",
        "position_valid": True, "state_valid": True,
        "planning_ready": bool(payload.get("planning_ready", False) and complete_geometry),
        "NED_position": center, "frame_id": "local_ned", "observed_at_s": result["observed_at_s"],
        "geometry": geometry, "metadata": metadata,
    }]
    return result


def map_objects(payload):
    """Project one source batch to the USV object contract, without history."""
    objects = []
    for item in payload["objects"]:
        obj = {"type": item["type"], "color": item.get("color", "unknown"),
               "state": item.get("state", "unknown"), "id": item["id"],
               "NED_position": list(item["NED_position"]),
               "position_valid": item["position_valid"], "state_valid": item["state_valid"]}
        # Dock center is a detected structure, not a verified physical berth.
        if item["type"] == "dock":
            obj["planning_ready"] = item["planning_ready"]
            obj["geometry"] = deepcopy(item["geometry"])
        objects.append(obj)
    return objects


class MissionMapRecorder:
    """Accumulate MAP positions; pause/resume keeps the same survey records."""

    def __init__(self, match_radius_m=1.0, history_limit=64):
        if match_radius_m <= 0 or history_limit < 1:
            raise ValueError("map match radius and history limit must be positive")
        self.match_radius_m, self.history_limit = match_radius_m, history_limit
        self.mode, self.session_id = "OFF", None
        self.started_at_s = self.completed_at_s = None
        self.records = []

    def set_mode(self, mode, now_s, session_id=None):
        new_session = mode == "MAP" and (self.session_id is None or (
            session_id is not None and session_id != self.session_id))
        changed = mode != self.mode or new_session
        if new_session:
            self.session_id = session_id or str(uuid4())
            self.started_at_s, self.records = now_s, []
        if self.mode == "MAP" and mode != "MAP":
            self.completed_at_s = now_s
        elif mode == "MAP":
            self.completed_at_s = None
        self.mode = mode
        return changed

    def ingest(self, source, payload, now_s):
        """Consume adapter output with source timestamps and local NED coordinates."""
        if self.mode != "MAP" or not payload["accepted"]:
            return 0
        stamp, used, seen_ids, count = payload["observed_at_s"], set(), set(), 0
        for item in payload["objects"]:
            if item["type"] not in MAP_TYPES:
                continue
            key = (item["type"], item["id"])
            if key in seen_ids:
                continue
            seen_ids.add(key)
            point = item["NED_position"]
            candidates = [(math.dist(record["NED_position"], point), index)
                          for index, record in enumerate(self.records)
                          if index not in used and record["source"] == source and record["type"] == item["type"]]
            candidates = sorted(pair for pair in candidates if pair[0] <= self.match_radius_m)
            if candidates:
                index = candidates[0][1]
                record = self.records[index]
            else:
                index = len(self.records)
                record = {"id": index, "record_id": f"{self.session_id}:{index}",
                          "type": item["type"], "source": source, "first_seen_s": stamp,
                          "observations": [], "observation_count": 0}
                self.records.append(record)
            used.add(index)
            observation = deepcopy(item)
            observation.update(source=source, observed_at_s=stamp)
            record.setdefault("first", deepcopy(observation))
            record["observations"].append(observation)
            del record["observations"][:-self.history_limit]
            record.update(last_seen_s=stamp, NED_position=list(point),
                          color=item.get("color", "unknown"), state=item.get("state", "unknown"),
                          source_object_id=item["id"], latest=observation,
                          geometry=observation.get("geometry"), observation_count=record["observation_count"] + 1)
            count += 1
        return count

    def snapshot(self, now_s):
        return {"schema_version": 1, "mode": self.mode, "active": self.mode == "MAP",
                "session_id": self.session_id, "started_at_s": self.started_at_s,
                "completed_at_s": self.completed_at_s, "generated_at_s": now_s,
                "frame_id": "local_ned", "identity_scope": "SURVEY_SPATIAL_RECORD_NOT_PHYSICAL_ID",
                "observation_history_limit": self.history_limit, "objects": deepcopy(self.records)}

    def write_snapshot(self, path, now_s):
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(self.snapshot(now_s), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=destination.name + ".", delete=False) as stream:
            temporary = stream.name
            stream.write(encoded)
        try:
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
