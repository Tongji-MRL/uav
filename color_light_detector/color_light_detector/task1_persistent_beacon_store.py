"""Durable, conservative Task 1 beacon observations and map records.

A map-record ID is an archive key, not a confirmed physical beacon identity.
No stored observation is promoted to live validity after a process restart.
"""
import json
import math
import sqlite3
import statistics
from contextlib import contextmanager
from pathlib import Path

from .task1_beacon_map import STATES, config_string, image_stamp, xyz


APPLICATION_ID = 0x5258314D  # "RX1M"
SCHEMA_VERSION = 2


class PersistentBeaconStore:
    def __init__(self, path, config):
        self.path = Path(path).expanduser().resolve()
        if str(path) == ":memory:":
            raise ValueError("DATABASE_MUST_BE_OUTSIDE_SOURCE_TREE")
        try:
            self.path.relative_to(Path(__file__).resolve().parents[1])
        except ValueError:
            pass
        else:
            raise ValueError("DATABASE_MUST_BE_OUTSIDE_SOURCE_TREE")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.simulation = config.get("simulation", False) is True
        self.namespace = (
            config_string(config, "shared_frame_id"),
            config_string(config, "shared_origin_id"),
            config_string(config, "transform_id"),
        )
        self.manifest = self._map_manifest(config)
        self.persistence_enabled = (
            self.manifest["transform_confirmed"]
            and self.manifest["source_origin_id"] != "UNCONFIRMED"
            and all(value != "UNCONFIRMED" for value in self.namespace)
            and self.manifest["expected_pose_reset_count"] is not None
        )
        self.max_sigma = float(config.get("persistent_max_sigma_m", 0.75))
        self.min_observations = int(config.get("persistent_min_observations", 2))
        self.window = float(config.get("persistent_stability_window_s", 3.0))
        self.radius = float(config.get("persistent_match_radius_m", 1.5))
        self.margin = float(config.get("persistent_match_margin_m", 0.5))
        self.stability_radius = float(
            config.get("persistent_stability_radius_m", 0.35))
        self.min_stability_duration = float(
            config.get("persistent_min_stability_duration_s", 0.5))
        self.linked_observation_interval = float(
            config.get("persistent_linked_observation_interval_s", 0.5))
        self.position_ttl = float(config["position_ttl_s"])
        self.state_ttl = float(config["state_ttl_s"])
        if (any(not value for value in self.namespace)
                or any(not math.isfinite(value) or value <= 0 for value in (
                    self.max_sigma, self.window, self.radius, self.margin,
                    self.stability_radius, self.min_stability_duration,
                    self.linked_observation_interval,
                    self.position_ttl, self.state_ttl))
                or self.min_observations < 2
                or self.stability_radius > self.radius
                or self.min_stability_duration > self.window
                or self.linked_observation_interval > min(
                    self.position_ttl, self.state_ttl)):
            raise ValueError("INVALID_PERSISTENT_MAP_CONFIG")
        self._live_positions = {}
        self._live_states = {}
        self._active_session = None
        self._active_source_alignment = None
        self.db = sqlite3.connect(str(self.path), timeout=10)
        try:
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA foreign_keys=ON")
            self._initialize()
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
        except Exception:
            self.db.close()
            raise

    @staticmethod
    def _map_manifest(config):
        reset_count = config.get("expected_pose_reset_count")
        if reset_count is not None and (
                type(reset_count) is not int or reset_count < 0):
            raise ValueError("INVALID_EXPECTED_RESET_COUNT")
        yaw_deg = float(config["yaw_deg"])
        if not math.isfinite(yaw_deg):
            raise ValueError("INVALID_MAP_CONFIG")
        return {
            "source_frame_id": config_string(config, "source_frame_id"),
            "shared_frame_id": config_string(config, "shared_frame_id"),
            "source_origin_id": config_string(config, "source_origin_id"),
            "shared_origin_id": config_string(config, "shared_origin_id"),
            "transform_id": config_string(config, "transform_id"),
            "transform_confirmed": config.get("transform_confirmed") is True,
            "translation_xyz_m": xyz(config["translation_xyz_m"]),
            "yaw_deg": yaw_deg,
            "expected_pose_reset_count": reset_count,
        }

    def _initialize(self):
        application_id = self.db.execute("PRAGMA application_id").fetchone()[0]
        user_version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if user_version > SCHEMA_VERSION:
            raise ValueError("UNSUPPORTED_DATABASE_SCHEMA_VERSION")
        if application_id not in (0, APPLICATION_ID):
            raise ValueError("DATABASE_APPLICATION_ID_MISMATCH")
        existing_tables = {
            row["name"] for row in self.db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")
            if not row["name"].startswith("sqlite_")
        }
        known_tables = {
            "metadata", "frames", "records", "observations", "track_links",
            "transform_manifests",
        }
        if application_id == 0 and existing_tables-known_tables:
            raise ValueError("UNRECOGNIZED_DATABASE_SCHEMA")
        statements = (
            """CREATE TABLE IF NOT EXISTS metadata (
                   key TEXT PRIMARY KEY, value TEXT NOT NULL
               )""",
            """CREATE TABLE IF NOT EXISTS frames (
                   session_id TEXT NOT NULL, sequence INTEGER NOT NULL,
                   stamp_s REAL NOT NULL,
                   PRIMARY KEY (session_id, sequence)
               )""",
            """CREATE TABLE IF NOT EXISTS records (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   shared_frame_id TEXT NOT NULL,
                   shared_origin_id TEXT NOT NULL,
                   transform_id TEXT NOT NULL,
                   last_shared_xyz_json TEXT,
                   position_stamp_s REAL,
                   horizontal_sigma_m REAL,
                   last_state TEXT NOT NULL DEFAULT 'UNKNOWN',
                   state_stamp_s REAL,
                   last_session_id TEXT,
                   last_track_id INTEGER,
                   last_source_origin_id TEXT,
                   last_pose_reset_count INTEGER,
                   last_timestamp_source TEXT,
                   observation_count INTEGER NOT NULL DEFAULT 0
               )""",
            """CREATE TABLE IF NOT EXISTS observations (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   session_id TEXT NOT NULL, sequence INTEGER NOT NULL,
                   track_id INTEGER NOT NULL, stamp_s REAL NOT NULL,
                   shared_frame_id TEXT NOT NULL,
                   shared_origin_id TEXT NOT NULL,
                   transform_id TEXT NOT NULL,
                   shared_xyz_json TEXT, horizontal_sigma_m REAL,
                   state TEXT, record_id INTEGER REFERENCES records(id),
                   association_status TEXT NOT NULL,
                   source_origin_id TEXT,
                   pose_reset_count INTEGER,
                   timestamp_source TEXT,
                   UNIQUE (session_id, sequence, track_id),
                   FOREIGN KEY (session_id, sequence)
                       REFERENCES frames(session_id, sequence)
               )""",
            """CREATE INDEX IF NOT EXISTS observation_candidate_idx
                   ON observations(session_id, track_id, stamp_s)""",
            """CREATE INDEX IF NOT EXISTS observation_record_idx
                   ON observations(record_id, stamp_s DESC, id DESC)""",
            """CREATE TABLE IF NOT EXISTS track_links (
                   session_id TEXT NOT NULL, track_id INTEGER NOT NULL,
                   shared_frame_id TEXT NOT NULL,
                   shared_origin_id TEXT NOT NULL,
                   transform_id TEXT NOT NULL,
                   record_id INTEGER NOT NULL REFERENCES records(id),
                   PRIMARY KEY (session_id, track_id, shared_frame_id,
                                shared_origin_id, transform_id)
               )""",
            """CREATE TABLE IF NOT EXISTS transform_manifests (
                   shared_frame_id TEXT NOT NULL,
                   shared_origin_id TEXT NOT NULL,
                   transform_id TEXT NOT NULL,
                   manifest_json TEXT NOT NULL,
                   PRIMARY KEY (shared_frame_id, shared_origin_id, transform_id)
               )""",
        )
        with self.db:
            for statement in statements:
                self.db.execute(statement)
            self._validate_schema()
            mode = "simulation" if self.simulation else "real"
            row = self.db.execute(
                "SELECT value FROM metadata WHERE key='mode'").fetchone()
            if row is None:
                self.db.execute(
                    "INSERT INTO metadata(key,value) VALUES('mode',?)", (mode,))
            elif row["value"] != mode:
                raise ValueError("SIMULATION_REAL_DATABASE_MODE_MISMATCH")
            rebuild_counts = user_version < SCHEMA_VERSION
            self._ensure_counter("frame_count", "frames", rebuild_counts)
            self._ensure_counter(
                "observation_count", "observations", rebuild_counts)
            self._ensure_transform_manifest()
            self.db.execute(f"PRAGMA application_id={APPLICATION_ID}")
            self.db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        if (self.db.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID
                or self.db.execute("PRAGMA user_version").fetchone()[0]
                != SCHEMA_VERSION):
            raise RuntimeError("DATABASE_SCHEMA_HEADER_WRITE_FAILED")

    def _validate_schema(self):
        required = {
            "metadata": {"key", "value"},
            "frames": {"session_id", "sequence", "stamp_s"},
            "records": {
                "id", "shared_frame_id", "shared_origin_id", "transform_id",
                "last_shared_xyz_json", "position_stamp_s", "horizontal_sigma_m",
                "last_state", "state_stamp_s", "last_session_id", "last_track_id",
                "last_source_origin_id", "last_pose_reset_count",
                "last_timestamp_source", "observation_count",
            },
            "observations": {
                "id", "session_id", "sequence", "track_id", "stamp_s",
                "shared_frame_id", "shared_origin_id", "transform_id",
                "shared_xyz_json", "horizontal_sigma_m", "state", "record_id",
                "association_status", "source_origin_id", "pose_reset_count",
                "timestamp_source",
            },
            "track_links": {
                "session_id", "track_id", "shared_frame_id", "shared_origin_id",
                "transform_id", "record_id",
            },
            "transform_manifests": {
                "shared_frame_id", "shared_origin_id", "transform_id",
                "manifest_json",
            },
        }
        for table, columns in required.items():
            present = {
                row["name"] for row in self.db.execute(
                    f"PRAGMA table_info({table})")
            }
            if not columns <= present:
                raise ValueError("INCOMPATIBLE_DATABASE_SCHEMA")

    def _ensure_counter(self, key, table, rebuild=False):
        row = self.db.execute(
            "SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        try:
            valid = row is not None and int(row["value"]) >= 0
        except (TypeError, ValueError):
            valid = False
        if rebuild or not valid:
            count = self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            self.db.execute(
                "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
                (key, str(count)))

    def _archive_has_data(self):
        return bool(self.db.execute(
            """SELECT EXISTS(SELECT 1 FROM frames LIMIT 1)
                      OR EXISTS(SELECT 1 FROM observations LIMIT 1)
                      OR EXISTS(SELECT 1 FROM records LIMIT 1)"""
        ).fetchone()[0])

    def _ensure_transform_manifest(self):
        encoded = json.dumps(
            self.manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
        rows = self.db.execute(
            """SELECT shared_frame_id,shared_origin_id,transform_id,manifest_json
               FROM transform_manifests""").fetchall()
        if not rows:
            if self._archive_has_data():
                raise ValueError(
                    "LEGACY_DATABASE_WITH_DATA_HAS_NO_TRANSFORM_MANIFEST")
            self.db.execute(
                """INSERT INTO transform_manifests(
                       shared_frame_id,shared_origin_id,transform_id,manifest_json)
                   VALUES(?,?,?,?)""", (*self.namespace, encoded))
        else:
            row = rows[0]
            try:
                stored_manifest = json.loads(row["manifest_json"])
            except (TypeError, ValueError):
                stored_manifest = None
            stored_namespace = tuple(
                row[key] for key in self._namespace_keys())
            matches = (
                len(rows) == 1 and stored_namespace == self.namespace
                and stored_manifest == self.manifest
            )
            if not matches and self._archive_has_data():
                raise ValueError("TRANSFORM_MANIFEST_MISMATCH")
            if not matches:
                self.db.execute("DELETE FROM transform_manifests")
                self.db.execute(
                    """INSERT INTO transform_manifests(
                           shared_frame_id,shared_origin_id,transform_id,
                           manifest_json)
                       VALUES(?,?,?,?)""", (*self.namespace, encoded))
            elif row["manifest_json"] != encoded:
                self.db.execute(
                    """UPDATE transform_manifests SET manifest_json=?
                       WHERE shared_frame_id=? AND shared_origin_id=?
                         AND transform_id=?""",
                    (encoded, *self.namespace))

    def _increment_counter(self, key, amount=1):
        self.db.execute(
            """UPDATE metadata SET value=CAST(value AS INTEGER)+?
               WHERE key=?""", (amount, key))

    def _counter_value(self, key):
        row = self.db.execute(
            "SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        try:
            value = int(row["value"])
        except (TypeError, ValueError):
            raise RuntimeError("DATABASE_COUNTER_CORRUPT") from None
        if value < 0:
            raise RuntimeError("DATABASE_COUNTER_CORRUPT")
        return value

    @contextmanager
    def _transaction(self):
        try:
            with self.db:
                yield
        except Exception:
            self._live_positions.clear()
            self._live_states.clear()
            raise

    @staticmethod
    def _id(number):
        return f"B{number:06d}"

    @staticmethod
    def _distance(first, second):
        return math.dist(first, second)

    def link_for(self, session_id, track_id):
        row = self.db.execute(
            """SELECT record_id FROM track_links WHERE session_id=? AND track_id=?
               AND shared_frame_id=? AND shared_origin_id=? AND transform_id=?""",
            (session_id, track_id, *self.namespace)).fetchone()
        return self._id(row["record_id"]) if row else None

    def _other_live_links(self, session, track_id, number, stamp, visible_tracks):
        """Historical track links do not reserve an archive record forever."""
        rows = self.db.execute(
            """SELECT l.track_id,MAX(o.stamp_s) AS last_observed
               FROM track_links l LEFT JOIN observations o
                 ON o.session_id=l.session_id AND o.track_id=l.track_id
                AND o.record_id=l.record_id
               WHERE l.session_id=? AND l.shared_frame_id=?
                 AND l.shared_origin_id=? AND l.transform_id=?
                 AND l.record_id=? AND l.track_id<>?
               GROUP BY l.track_id""",
            (session, *self.namespace, number, track_id)).fetchall()
        lifetime = max(self.position_ttl, self.state_ttl)
        return [row["track_id"] for row in rows
                if row["track_id"] in visible_tracks or (
                    row["last_observed"] is not None
                    and stamp-row["last_observed"] <= lifetime)]

    def _qualified_observation(self, track, entry, stamp):
        visible = track.get("visible") is True
        state = None
        if (visible and track.get("state_valid") is True
                and track.get("change_pending") is not True and entry is not None
                and entry.get("state") in STATES
                and entry.get("state_observed_at_s") is not None
                and abs(entry["state_observed_at_s"] - stamp) < 1e-6):
            state = entry["state"]
        point, sigma = None, None
        position = track.get("position")
        if (visible and all(value != "UNCONFIRMED" for value in self.namespace)
                and isinstance(position, dict)
                and position.get("valid") is True
                and position.get("world_valid") is True
                and entry is not None
                and entry.get("position_observed_at_s") is not None
                and abs(entry["position_observed_at_s"] - stamp) < 1e-6
                and (entry.get("position_valid") is True or (
                    self.simulation and entry.get("position_status") == "SIMULATION_ONLY"))):
            try:
                point = xyz(entry["shared_xyz_m"])
                raw_sigma = entry.get("horizontal_sigma_m")
                if (isinstance(raw_sigma, bool)
                        or not isinstance(raw_sigma, (int, float))
                        or not math.isfinite(raw_sigma)
                        or not 0 <= raw_sigma <= self.max_sigma):
                    point = None
                else:
                    sigma = float(raw_sigma)
            except (KeyError, TypeError, ValueError):
                point = None
        return point, sigma, state

    def ingest(self, payload, live_snapshot):
        """Append one accepted frame and associate only current verified observations."""
        if not isinstance(payload, dict) or not isinstance(live_snapshot, dict):
            raise ValueError("FRAME_AND_SNAPSHOT_REQUIRED")
        session, sequence = payload["session_id"], payload["sequence"]
        stamp = image_stamp(payload)
        if (not isinstance(session, str) or not session or type(sequence) is not int
                or sequence < 0 or not math.isfinite(stamp)
                or live_snapshot.get("session_id") != session
                or live_snapshot.get("simulation") is not self.simulation
                or tuple(live_snapshot.get(key) for key in (
                    "shared_frame_id", "shared_origin_id", "transform_id"))
                != self.namespace):
            raise ValueError("FRAME_SNAPSHOT_MISMATCH")
        tracks = payload.get("tracks")
        if not isinstance(tracks, list):
            raise ValueError("TRACKS_REQUIRED")
        if not self.persistence_enabled:
            return
        entries = {
            row.get("track_id"): row for row in live_snapshot.get("beacons", [])
            if isinstance(row, dict) and type(row.get("track_id")) is int
        }
        if self._active_session != session:
            self._live_positions.clear()
            self._live_states.clear()
            self._active_session = session
        source_origin = live_snapshot.get("active_source_origin_id")
        pose_reset = live_snapshot.get("active_pose_reset_count")
        alignment = (source_origin, pose_reset)
        if self._active_source_alignment is not None and alignment != self._active_source_alignment:
            self._live_positions.clear()
        if live_snapshot.get("alignment_valid") is not True:
            self._live_positions.clear()
        self._active_source_alignment = alignment
        timestamp_source = live_snapshot.get("timestamp_source")
        prepared = []
        pending = set()
        seen = set()
        visible_tracks = {
            track["track_id"] for track in tracks
            if isinstance(track, dict) and type(track.get("track_id")) is int
            and track.get("visible") is True
        }
        for track in tracks:
            if not isinstance(track, dict):
                continue
            track_id = track.get("track_id")
            if type(track_id) is not int or track_id < 0 or track_id in seen:
                continue
            seen.add(track_id)
            point, sigma, state = self._qualified_observation(
                track, entries.get(track_id), stamp)
            link = self.db.execute(
                """SELECT l.record_id,r.last_shared_xyz_json,r.last_state,
                          r.position_stamp_s,r.state_stamp_s
                   FROM track_links l JOIN records r ON r.id=l.record_id
                   WHERE l.session_id=? AND l.track_id=?
                     AND l.shared_frame_id=? AND l.shared_origin_id=?
                     AND l.transform_id=?""",
                (session, track_id, *self.namespace)).fetchone()
            is_pending = track.get("change_pending") is True
            if is_pending and link is not None:
                self._live_states.pop(link["record_id"], None)
            if link is not None and self._other_live_links(
                    session, track_id, link["record_id"], stamp, visible_tracks):
                # Revoke even when there is no qualified new measurement or
                # the archive rate limiter would skip writing this frame.
                self._live_positions.pop(link["record_id"], None)
                self._live_states.pop(link["record_id"], None)
            if point is None and (state is None or link is None):
                continue
            if link is not None:
                jump = (
                    point is not None
                    and link["last_shared_xyz_json"] is not None
                    and self._distance(
                        point, json.loads(link["last_shared_xyz_json"]))
                    > self.radius
                )
                state_changed = (
                    state is not None and state != link["last_state"])
                timestamps = [
                    value for value in (
                        link["position_stamp_s"], link["state_stamp_s"])
                    if value is not None
                ]
                latest = max(timestamps) if timestamps else None
                sample_due = (
                    latest is None or stamp < latest
                    or stamp-latest+1e-9 >= self.linked_observation_interval
                )
                if not (jump or state_changed or sample_due):
                    continue
            prepared.append((track_id, point, sigma, state))
            if is_pending:
                pending.add(track_id)
        if not prepared:
            return
        accepted = []
        with self._transaction():
            cursor = self.db.execute(
                "INSERT OR IGNORE INTO frames(session_id,sequence,stamp_s) VALUES(?,?,?)",
                (session, sequence, stamp))
            if cursor.rowcount == 0:
                return
            self._increment_counter("frame_count")
            inserted_observations = 0
            for track_id, point, sigma, state in prepared:
                self.db.execute(
                    """INSERT INTO observations(
                        session_id,sequence,track_id,stamp_s,
                        shared_frame_id,shared_origin_id,transform_id,
                        shared_xyz_json,horizontal_sigma_m,state,
                        association_status,source_origin_id,pose_reset_count,
                        timestamp_source)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (session, sequence, track_id, stamp, *self.namespace,
                     json.dumps(point) if point is not None else None,
                     sigma, state,
                     "PENDING_STABILITY" if point is not None else
                     "STATE_ONLY" if state is not None else "NO_VERIFIED_OBSERVATION",
                     source_origin, pose_reset, timestamp_source))
                inserted_observations += 1
                accepted.append((track_id, point, state))
            if inserted_observations:
                self._increment_counter(
                    "observation_count", inserted_observations)
            # Existing links are stable within a session while geometry remains near.
            for track_id, point, state in accepted:
                row = self.db.execute(
                    """SELECT l.record_id,r.last_shared_xyz_json FROM track_links l
                       JOIN records r ON r.id=l.record_id
                       WHERE l.session_id=? AND l.track_id=?
                         AND l.shared_frame_id=? AND l.shared_origin_id=?
                         AND l.transform_id=?""",
                    (session, track_id, *self.namespace)).fetchone()
                if row is None:
                    continue
                if self._other_live_links(
                        session, track_id, row["record_id"], stamp, visible_tracks):
                    self.db.execute(
                        """UPDATE observations
                           SET association_status='AMBIGUOUS_SAME_FRAME'
                           WHERE session_id=? AND sequence=? AND track_id=?""",
                        (session, sequence, track_id))
                    self._live_positions.pop(row["record_id"], None)
                    self._live_states.pop(row["record_id"], None)
                    continue
                if (point is not None and row["last_shared_xyz_json"] is not None
                        and self._distance(point, json.loads(row["last_shared_xyz_json"]))
                        > self.radius):
                    self.db.execute(
                        """DELETE FROM track_links WHERE session_id=? AND track_id=?
                           AND shared_frame_id=? AND shared_origin_id=?
                           AND transform_id=?""",
                        (session, track_id, *self.namespace))
                    self.db.execute(
                        """UPDATE observations SET association_status='POSITION_JUMP_UNASSOCIATED'
                           WHERE session_id=? AND sequence=? AND track_id=?""",
                        (session, sequence, track_id))
                    self._live_positions.pop(row["record_id"], None)
                    self._live_states.pop(row["record_id"], None)
                    continue
                self.db.execute(
                    """UPDATE observations SET record_id=?,
                       association_status='LINKED_SESSION_TRACK'
                       WHERE session_id=? AND sequence=? AND track_id=?""",
                    (row["record_id"], session, sequence, track_id))
                self._append_record_observation(
                    row["record_id"], point, state, stamp, session, track_id,
                    source_origin, pose_reset, timestamp_source, sigma)
                if track_id in pending:
                    self._live_states.pop(row["record_id"], None)
                if point is not None:
                    self._live_positions[row["record_id"]] = stamp
                if state is not None:
                    self._live_states[row["record_id"]] = stamp
            # Candidate tracks require repeated stable positions in this short window.
            candidates = []
            for track_id, point, state in accepted:
                if point is None or self.link_for(session, track_id):
                    continue
                recent = self.db.execute(
                    """SELECT stamp_s,shared_xyz_json FROM observations
                       WHERE session_id=? AND track_id=? AND record_id IS NULL
                         AND shared_frame_id=? AND shared_origin_id=?
                         AND transform_id=? AND shared_xyz_json IS NOT NULL
                         AND stamp_s>=? ORDER BY stamp_s DESC""",
                    (session, track_id, *self.namespace, stamp-self.window)).fetchall()
                if len(recent) < self.min_observations:
                    continue
                points = [json.loads(row["shared_xyz_json"]) for row in recent]
                center = [
                    statistics.median(old[axis] for old in points)
                    for axis in range(3)
                ]
                if any(self._distance(center, old) > self.stability_radius
                       for old in points):
                    self.db.execute(
                        """UPDATE observations SET association_status='UNSTABLE_POSITION'
                           WHERE session_id=? AND sequence=? AND track_id=?""",
                        (session, sequence, track_id))
                    continue
                stamps = [row["stamp_s"] for row in recent]
                if (max(stamps)-min(stamps) + 1e-9
                        < self.min_stability_duration):
                    continue
                candidates.append((track_id, center, state, min(stamps)))
            records = self.db.execute(
                """SELECT id,last_shared_xyz_json FROM records
                   WHERE shared_frame_id=? AND shared_origin_id=?
                     AND transform_id=? AND last_shared_xyz_json IS NOT NULL""",
                self.namespace).fetchall()
            proposed = {}
            for track_id, point, state, first_stamp in candidates:
                near = sorted(
                    (self._distance(point, json.loads(row["last_shared_xyz_json"])),
                     row["id"]) for row in records)
                near = [(distance, number) for distance, number in near
                        if distance <= self.radius]
                if near and (len(near) == 1 or near[1][0]-near[0][0] >= self.margin):
                    proposed[track_id] = (near[0][1], first_stamp)
                elif not near:
                    proposed[track_id] = (None, first_stamp)
                else:
                    self.db.execute(
                        """UPDATE observations SET association_status='AMBIGUOUS_SPATIAL_MATCH'
                           WHERE session_id=? AND sequence=? AND track_id=?""",
                        (session, sequence, track_id))
            # An existing record may have only one track in the current frame.
            claimed = {}
            for track_id, (number, _) in proposed.items():
                if number is not None:
                    claimed.setdefault(number, []).append(track_id)
            for track_id, point, state, _ in candidates:
                if track_id not in proposed:
                    continue
                number, first_stamp = proposed[track_id]
                is_new = number is None
                if number is not None:
                    competing = [other_id for other_id, other_point, _ in accepted
                                 if other_id != track_id and other_point is not None
                                 and self._distance(
                                     other_point,
                                     next(json.loads(row["last_shared_xyz_json"])
                                          for row in records if row["id"] == number))
                                 <= self.radius]
                    linked = self._other_live_links(
                        session, track_id, number, stamp, visible_tracks)
                    if competing or linked or len(claimed[number]) != 1:
                        self.db.execute(
                            """UPDATE observations SET association_status='AMBIGUOUS_SAME_FRAME'
                               WHERE session_id=? AND sequence=? AND track_id=?""",
                            (session, sequence, track_id))
                        continue
                else:
                    competing = [
                        other_id for other_id, other_point, _ in accepted
                        if other_id != track_id and other_point is not None
                        and self._distance(other_point, point) <= self.radius
                    ]
                    if competing:
                        self.db.execute(
                            """UPDATE observations SET association_status='AMBIGUOUS_NEW_RECORD'
                               WHERE session_id=? AND sequence=? AND track_id=?""",
                            (session, sequence, track_id))
                        continue
                    cursor = self.db.execute(
                        """INSERT INTO records(shared_frame_id,shared_origin_id,transform_id)
                           VALUES(?,?,?)""", self.namespace)
                    number = cursor.lastrowid
                self.db.execute(
                    """INSERT INTO track_links(
                        session_id,track_id,shared_frame_id,shared_origin_id,
                        transform_id,record_id) VALUES(?,?,?,?,?,?)""",
                    (session, track_id, *self.namespace, number))
                self.db.execute(
                    """UPDATE observations SET record_id=?,
                       association_status=CASE WHEN ? THEN 'NEW_MAP_RECORD'
                           ELSE 'SPATIAL_REASSOCIATED' END
                       WHERE session_id=? AND track_id=? AND shared_frame_id=?
                         AND shared_origin_id=? AND transform_id=?
                         AND record_id IS NULL AND stamp_s>=?""",
                    (number, is_new, session, track_id, *self.namespace, first_stamp))
                self._refresh_record(number)
                self._live_positions[number] = stamp
                if state is not None:
                    self._live_states[number] = stamp

    def _append_record_observation(self, number, point, state, stamp,
                                   session, track_id, source_origin,
                                   pose_reset, timestamp_source, sigma):
        row = self.db.execute(
            "SELECT * FROM records WHERE id=?", (number,)).fetchone()
        anchor = row["last_shared_xyz_json"]
        anchor_sigma = row["horizontal_sigma_m"]
        if anchor is None and point is not None:
            anchor = json.dumps(point)
            anchor_sigma = sigma
        self.db.execute(
            """UPDATE records SET last_shared_xyz_json=?,position_stamp_s=?,
               horizontal_sigma_m=?,last_state=?,state_stamp_s=?,
               last_session_id=?,last_track_id=?,last_source_origin_id=?,
               last_pose_reset_count=?,last_timestamp_source=?,
               observation_count=? WHERE id=?""",
            (anchor,
             stamp if point is not None else row["position_stamp_s"],
             anchor_sigma,
             state if state is not None else row["last_state"],
             stamp if state is not None else row["state_stamp_s"],
             session if point is not None or state is not None else row["last_session_id"],
             track_id if point is not None or state is not None else row["last_track_id"],
             source_origin if point is not None else row["last_source_origin_id"],
             pose_reset if point is not None else row["last_pose_reset_count"],
             timestamp_source if point is not None else row["last_timestamp_source"],
             row["observation_count"] + 1, number))

    def _refresh_record(self, number):
        where = "record_id=?"
        current = self.db.execute(
            """SELECT last_shared_xyz_json,horizontal_sigma_m
               FROM records WHERE id=?""",
            (number,)).fetchone()
        count = self.db.execute(
            f"SELECT COUNT(*) FROM observations WHERE {where}", (number,)).fetchone()[0]
        position_rows = self.db.execute(
            f"""SELECT id,shared_xyz_json,horizontal_sigma_m,stamp_s,
                       session_id,track_id,source_origin_id,
                       pose_reset_count,timestamp_source FROM observations
                WHERE {where} AND shared_xyz_json IS NOT NULL
                ORDER BY id""", (number,)).fetchall()
        position = position_rows[-1] if position_rows else None
        state = self.db.execute(
            f"""SELECT id,state,stamp_s,session_id,track_id FROM observations
                WHERE {where} AND state IS NOT NULL
                ORDER BY id DESC LIMIT 1""", (number,)).fetchone()
        anchor = current["last_shared_xyz_json"]
        anchor_sigma = current["horizontal_sigma_m"]
        if anchor is None:
            points = [json.loads(row["shared_xyz_json"])
                      for row in position_rows]
            if points:
                center = [
                    statistics.median(point[axis] for point in points)
                    for axis in range(3)
                ]
                anchor = json.dumps(center)
                uncertainty = [
                    float(row["horizontal_sigma_m"])
                    for row in position_rows
                    if row["horizontal_sigma_m"] is not None
                ]
                uncertainty.extend(
                    math.hypot(point[0]-center[0], point[1]-center[1])
                    for point in points
                )
                anchor_sigma = max(uncertainty) if uncertainty else None
        source = max(
            (row for row in (position, state) if row is not None),
            key=lambda row: row["id"], default=None)
        self.db.execute(
            """UPDATE records SET last_shared_xyz_json=?,position_stamp_s=?,
               horizontal_sigma_m=?,last_state=?,state_stamp_s=?,
               last_session_id=?,last_track_id=?,last_source_origin_id=?,
               last_pose_reset_count=?,last_timestamp_source=?,
               observation_count=? WHERE id=?""",
            (anchor,
             position["stamp_s"] if position else None,
             anchor_sigma,
             state["state"] if state else "UNKNOWN",
             state["stamp_s"] if state else None,
             source["session_id"] if source else None,
             source["track_id"] if source else None,
             position["source_origin_id"] if position else None,
             position["pose_reset_count"] if position else None,
             position["timestamp_source"] if position else None,
             count, number))

    def snapshot(self, now_s):
        if isinstance(now_s, bool) or not isinstance(now_s, (int, float)) or not math.isfinite(now_s):
            raise ValueError("INVALID_SNAPSHOT_TIME")
        records = []
        for row in self.db.execute("SELECT * FROM records ORDER BY id"):
            number = row["id"]
            current_namespace = tuple(row[key] for key in self._namespace_keys()) == self.namespace
            position_stamp = row["position_stamp_s"]
            state_stamp = row["state_stamp_s"]
            position_fresh = (
                current_namespace and self._live_positions.get(number) == position_stamp
                and position_stamp is not None
                and 0 <= now_s-position_stamp <= self.position_ttl)
            state_fresh = (
                current_namespace and self._live_states.get(number) == state_stamp
                and state_stamp is not None
                and 0 <= now_s-state_stamp <= self.state_ttl)
            records.append({
                "persistent_beacon_id": self._id(number),
                "identity_valid": False,
                "identity_status": "UNCONFIRMED_MAP_RECORD",
                "shared_frame_id": row["shared_frame_id"],
                "shared_origin_id": row["shared_origin_id"],
                "transform_id": row["transform_id"],
                "last_shared_xyz_m": (
                    json.loads(row["last_shared_xyz_json"])
                    if row["last_shared_xyz_json"] is not None else None),
                "position_observed_at_s": position_stamp,
                "horizontal_sigma_m": row["horizontal_sigma_m"],
                "position_valid": bool(position_fresh and not self.simulation),
                "historical_state": row["last_state"],
                "state_observed_at_s": state_stamp,
                "state_valid": bool(state_fresh and not self.simulation),
                "last_session_id": row["last_session_id"],
                "last_track_id": row["last_track_id"],
                "last_source_origin_id": row["last_source_origin_id"],
                "last_pose_reset_count": row["last_pose_reset_count"],
                "last_timestamp_source": row["last_timestamp_source"],
                "observation_count": row["observation_count"],
                "planning_valid": False,
            })
        return {
            "status": "OK",
            "simulation": self.simulation,
            "persistence_enabled": self.persistence_enabled,
            "database_file": str(self.path),
            "shared_frame_id": self.namespace[0],
            "shared_origin_id": self.namespace[1],
            "transform_id": self.namespace[2],
            "frame_count": self._counter_value("frame_count"),
            "observation_count": self._counter_value("observation_count"),
            "recent_association_status_counts": {
                row["association_status"]: row["total"] for row in self.db.execute(
                    """SELECT association_status,COUNT(*) AS total FROM
                       (SELECT association_status FROM observations
                        ORDER BY id DESC LIMIT 1000)
                       GROUP BY association_status""")
            },
            "recent_association_status_window": 1000,
            "records": records,
        }

    @staticmethod
    def _namespace_keys():
        return ("shared_frame_id", "shared_origin_id", "transform_id")

    def close(self):
        self.db.close()
