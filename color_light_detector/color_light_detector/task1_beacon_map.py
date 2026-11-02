"""Run-local Task 1 beacon map built only from verified world positions.

The upstream localizer owns optical -> body -> UAV-local ENU geometry. This
module only applies a measured UAV-local ENU -> fixed shared-map transform.
"""

import math
from pathlib import Path


STATES = {"OFF", "BLINK_RED", "BLINK_GREEN", "BLINK_BLUE", "SOLID_BLUE"}


def config_string(config, key):
    value = config[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("INVALID_" + key.upper())
    return value.strip()


def storage_paths(snapshot_file, database_file):
    snapshot = Path(snapshot_file).expanduser().resolve()
    database = Path(database_file).expanduser().resolve()
    snapshot_outputs = {snapshot, snapshot.with_name(snapshot.name + ".tmp")}
    database_files = {
        database,
        database.with_name(database.name + "-wal"),
        database.with_name(database.name + "-shm"),
    }
    if snapshot_outputs & database_files:
        raise ValueError("SNAPSHOT_AND_DATABASE_PATHS_COLLIDE")
    return snapshot, database


def xyz(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("EXPECTED_XYZ_M")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) for v in value):
        raise ValueError("NONFINITE_XYZ_M")
    return [float(v) for v in value]


def image_stamp(payload):
    stamp = payload["stamp"]
    sec, nanosec = stamp["sec"], stamp["nanosec"]
    if type(sec) is not int or type(nanosec) is not int or not 0 <= nanosec < 10**9:
        raise ValueError("INVALID_IMAGE_STAMP")
    return sec + nanosec * 1e-9


class BeaconMap:
    """Keep session-track observations; persistent physical IDs need association."""

    def __init__(self, config):
        self.source_frame = config_string(config, "source_frame_id")
        self.shared_frame = config_string(config, "shared_frame_id")
        self.source_origin = config_string(config, "source_origin_id")
        self.shared_origin = config_string(config, "shared_origin_id")
        self.transform_id = config_string(config, "transform_id")
        self.transform_confirmed = config["transform_confirmed"] is True
        self.simulation = config.get("simulation", False) is True
        self.expected_reset = config.get("expected_pose_reset_count")
        self.translation = xyz(config["translation_xyz_m"])
        self.yaw_deg = float(config["yaw_deg"])
        self.position_ttl = float(config["position_ttl_s"])
        self.state_ttl = float(config["state_ttl_s"])
        self.max_input_age = float(config["max_input_age_s"])
        self.max_position_sigma = float(config.get(
            "max_position_sigma_m",
            config.get("persistent_max_sigma_m", 0.75),
        ))
        self.track_retention = float(config.get("track_retention_s", 10.0))
        if (not self.source_frame or not self.shared_frame or
                not math.isfinite(self.yaw_deg) or
                any(not math.isfinite(v) or v <= 0 for v in
                    (self.position_ttl, self.state_ttl, self.max_input_age,
                     self.max_position_sigma,
                     self.track_retention)) or
                self.track_retention < max(self.position_ttl, self.state_ttl)):
            raise ValueError("INVALID_MAP_CONFIG")
        if self.expected_reset is not None and (
                type(self.expected_reset) is not int or self.expected_reset < 0):
            raise ValueError("INVALID_EXPECTED_RESET_COUNT")
        angle = math.radians(self.yaw_deg)
        self.cos_yaw, self.sin_yaw = math.cos(angle), math.sin(angle)
        self.session_id = None
        self.last_sequence = -1
        self.last_image_time = float("-inf")
        self.closed_sessions = set()
        self.active_origin_id = None
        self.active_reset_count = None
        self.source_coordinates_valid = False
        self.records = {}
        self.last_rejection = None
        self.timestamp_source = None

    def alignment_ready(self, origin_id, reset_count):
        return (self.source_coordinates_valid and self.transform_confirmed and
                self.source_origin != "UNCONFIRMED" and
                self.shared_origin != "UNCONFIRMED" and
                self.transform_id != "UNCONFIRMED" and
                self.expected_reset is not None and
                origin_id == self.source_origin and
                self.active_origin_id == self.source_origin and
                self.active_reset_count == self.expected_reset and
                reset_count == self.expected_reset)

    def to_shared(self, point):
        x, y, z = point
        return [self.cos_yaw*x - self.sin_yaw*y + self.translation[0],
                self.sin_yaw*x + self.cos_yaw*y + self.translation[1],
                z + self.translation[2]]

    def ingest(self, payload, now_s):
        """Take one complete results_json frame; reject stale or reordered frames."""
        try:
            if not isinstance(payload, dict):
                raise ValueError("RESULT_OBJECT_REQUIRED")
            if type(payload.get("schema_version")) is not int or payload["schema_version"] < 4:
                raise ValueError("POSITION_SCHEMA_REQUIRED")
            session, sequence = payload["session_id"], payload["sequence"]
            if not isinstance(session, str) or not session or type(sequence) is not int:
                raise ValueError("INVALID_SESSION_OR_SEQUENCE")
            stamp = image_stamp(payload)
            if (not math.isfinite(now_s) or stamp <= 0 or
                    not -0.05 <= now_s-stamp <= self.max_input_age):
                raise ValueError("IMAGE_TIME_STALE_OR_FUTURE")
            if not isinstance(payload["tracks"], list):
                raise ValueError("TRACKS_REQUIRED")
            track_ids = []
            for track in payload["tracks"]:
                track_id = track.get("track_id") if isinstance(track, dict) else None
                if type(track_id) is not int or track_id < 0:
                    raise ValueError("INVALID_TRACK_ID")
                track_ids.append(track_id)
            if len(track_ids) != len(set(track_ids)):
                raise ValueError("DUPLICATE_TRACK_ID")
            localization = payload.get("localization", {})
            if not isinstance(localization, dict):
                raise ValueError("LOCALIZATION_REQUIRED")
        except (KeyError, TypeError, ValueError) as error:
            self.last_rejection = str(error)
            return False
        if session != self.session_id:
            if session in self.closed_sessions or stamp <= self.last_image_time:
                self.last_rejection = "OLD_OR_REORDERED_SESSION"
                return False
            if self.session_id is not None:
                self.closed_sessions.add(self.session_id)
            self.session_id, self.last_sequence = session, -1
            self.active_origin_id = None
            self.active_reset_count = None
            self.records.clear()
        if sequence <= self.last_sequence or stamp <= self.last_image_time:
            self.last_rejection = "OUT_OF_ORDER_SEQUENCE"
            return False
        self.last_sequence, self.last_image_time = sequence, stamp
        self.timestamp_source = payload.get("timestamp_source")
        origin_id = localization.get("map_origin_id")
        self.last_rejection = None
        if self.active_origin_id is not None and origin_id != self.active_origin_id:
            for record in self.records.values():
                record["position_time"] = None
            self.last_rejection = "MAP_ORIGIN_CHANGED_REQUIRES_ALIGNMENT"
        self.active_origin_id = origin_id
        # Older schema-4 producers omitted these fields; their contract is ENU.
        # An explicit incompatible declaration must revoke cached positions too.
        self.source_coordinates_valid = (
            localization.get("world_convention", "ENU") == "ENU"
            and localization.get("world_frame_id", self.source_frame) == self.source_frame)
        if not self.source_coordinates_valid:
            for record in self.records.values():
                record["position_time"] = None
            self.last_rejection = "SOURCE_COORDINATE_FRAME_MISMATCH"
        pose_alignment = localization.get("pose_alignment")
        reset = pose_alignment.get("reset_count") if isinstance(pose_alignment, dict) else None
        self.active_reset_count = reset
        if self.expected_reset is not None and reset is not None and reset != self.expected_reset:
            for record in self.records.values():
                record["position_time"] = None
            self.last_rejection = "POSE_RESET_REQUIRES_NEW_MAP_ALIGNMENT"
        for track in payload["tracks"]:
            track_id = track.get("track_id") if isinstance(track, dict) else None
            if type(track_id) is not int or track_id < 0:
                continue
            record = self.records.setdefault(track_id, {
                "track_id": track_id, "state": "UNKNOWN", "state_time": None,
                "state_revision": None, "change_pending": False,
                "source_xyz_m": None, "position_time": None,
                "horizontal_sigma_m": None, "origin_id": None, "reset_count": None,
                "last_seen_time": stamp,
            })
            record["last_seen_time"] = stamp
            record["change_pending"] = track.get("change_pending") is True
            state = track.get("committed_state", track.get("state"))
            if (track.get("visible") is True and
                    track.get("state_valid") is True and state in STATES):
                record["state"] = state
                record["state_time"] = stamp
                record["state_revision"] = track.get("state_revision")
            position = track.get("position")
            if (track.get("visible") is not True or
                    not isinstance(position, dict) or
                    localization.get("status") != "READY" or
                    not self.source_coordinates_valid or
                    position.get("valid") is not True or
                    position.get("world_valid") is not True or
                    position.get("frame_id") != self.source_frame):
                continue
            try:
                point = xyz(position["world_xyz_m"])
                if xyz(position["xyz_m"]) != point:
                    continue
            except (KeyError, ValueError):
                continue
            alignment = position.get("pose_alignment", {})
            reset_count = alignment.get("reset_count") if isinstance(alignment, dict) else None
            if type(reset_count) is not int or reset_count < 0:
                continue
            sigma = position.get("horizontal_sigma_m")
            if (isinstance(sigma, bool) or
                    not isinstance(sigma, (int, float)) or
                    not math.isfinite(sigma) or
                    not 0 <= sigma <= self.max_position_sigma):
                continue
            sigma = float(sigma)
            record.update(source_xyz_m=point, position_time=stamp,
                          horizontal_sigma_m=sigma, origin_id=origin_id,
                          reset_count=reset_count)
        return True

    def snapshot(self, now_s):
        """Expose independent position, state and identity validity."""
        if not math.isfinite(now_s):
            raise ValueError("INVALID_SNAPSHOT_TIME")
        expired = [
            track_id for track_id, record in self.records.items()
            if (record["last_seen_time"] is not None
                and now_s >= record["last_seen_time"]
                and now_s-record["last_seen_time"] > self.track_retention)
        ]
        for track_id in expired:
            del self.records[track_id]
        entries = []
        for track_id, record in sorted(self.records.items()):
            position_time, state_time = record["position_time"], record["state_time"]
            position_fresh = (position_time is not None and
                              0 <= now_s-position_time <= self.position_ttl)
            state_fresh = (state_time is not None and
                           0 <= now_s-state_time <= self.state_ttl and
                           not record["change_pending"])
            aligned = self.alignment_ready(record["origin_id"], record["reset_count"])
            shared = (self.to_shared(record["source_xyz_m"])
                      if position_fresh and aligned else None)
            entries.append({
                "observation_id": f"{self.session_id}:{track_id}",
                "track_id": track_id,
                "persistent_beacon_id": None,
                "identity_valid": False,
                "state": record["state"],
                "state_valid": bool(state_fresh and not self.simulation),
                "state_observed_at_s": state_time,
                "state_revision": record["state_revision"],
                "change_pending": record["change_pending"],
                "source_xyz_m": record["source_xyz_m"] if position_fresh else None,
                "source_frame_id": self.source_frame,
                "shared_xyz_m": shared,
                "shared_frame_id": self.shared_frame,
                "position_valid": bool(shared is not None and not self.simulation),
                "position_observed_at_s": position_time,
                "horizontal_sigma_m": record["horizontal_sigma_m"] if position_fresh else None,
                "position_status": (
                    "SIMULATION_ONLY" if self.simulation and shared is not None else
                    "VALID" if shared is not None else
                    "STALE" if position_time is not None and not position_fresh else
                    "SHARED_FRAME_UNCONFIRMED" if position_time is not None else
                    "NO_VALID_POSITION"),
            })
        return {
            "schema_version": 1, "simulation": self.simulation,
            "generated_at_s": now_s, "session_id": self.session_id,
            "source_frame_id": self.source_frame,
            "source_convention": "ENU", "shared_convention": "ENU",
            "source_origin_id": self.source_origin,
            "shared_frame_id": self.shared_frame,
            "shared_origin_id": self.shared_origin,
            "transform_id": self.transform_id,
            "active_source_origin_id": self.active_origin_id,
            "active_pose_reset_count": self.active_reset_count,
            "timestamp_source": self.timestamp_source,
            "identity_scope": "SESSION_TRACK_ONLY",
            "alignment_valid": self.alignment_ready(
                self.active_origin_id, self.active_reset_count),
            "alignment_status": (
                "READY" if self.alignment_ready(
                    self.active_origin_id, self.active_reset_count)
                else self.last_rejection or "UNCONFIRMED"),
            "last_rejection": self.last_rejection,
            "beacons": entries,
        }
